import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================================================================
# 1. APART Adapter 单体 (修改版：增加最小 Scale 保底)
# =========================================================================
class APART_Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=256, dropout=0.1): 
        super().__init__()
        
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
        # === 核心设置 ===
        # 初始 logits 设置为 0.0，对应的 sigmoid 约为 0.5
        # 这样初始 scale 会在 (min + max) / 2 附近
        self.scale_logits = nn.Parameter(torch.tensor(-5.0))
        
        # [关键修改] 定义 Scale 的活动范围
        self.min_scale_val = 0.07  # 设置底薪：最少也要有 1% 的贡献
        self.max_scale_val = 0.10  # 设置上限：最多 20% (之前建议放宽到 0.2)

        with torch.no_grad():
            # 使用较小的正态分布初始化，防止一开始是纯 0 被 Weight Decay 吸住
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.normal_(self.up_proj.weight, mean=0.0, std=1e-4) # 给一点点初始扰动
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        # 1. 混合精度保护
        origin_dtype = x.dtype
        x = x.to(self.down_proj.weight.dtype)
        
        # 2. Adapter 计算
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = self.dropout(down)
        up = self.up_proj(down)
        
        # 3. [关键修改] 计算带保底的 Scale
        # 逻辑：scale = min + (max - min) * sigmoid(logits)
        # 这样无论 logits 怎么变，scale 永远不会小于 self.min_scale_val
        probs = torch.sigmoid(self.scale_logits)
        current_scale = self.min_scale_val + (self.max_scale_val - self.min_scale_val) * probs
        
        out = up * current_scale
        
        # 4. 转回原精度
        return out.to(origin_dtype)


# =========================================================================
# 2. Pool Wrapper (保持不变)
# =========================================================================
class APART_PoolWrapper(nn.Module):
    def __init__(self, original_mlp, input_dim, pool_size=5):
        super().__init__()
        self.original_mlp = original_mlp 
        
        # 创建专家池：包含 pool_size 个独立的 Adapter
        self.adapters = nn.ModuleList([
            APART_Adapter(input_dim, bottleneck_dim=256) 
            for _ in range(pool_size)
        ])
        
        # 当前激活的专家索引 (默认 0)
        # 如果设为 -1，表示不使用 Adapter (纯 CLIP 模式)
        self.active_idx = 0 

    def forward(self, x):
        # 1. 原始路径 (Frozen, 永远计算)
        with torch.no_grad():
            original_out = self.original_mlp(x)
        
        # 2. 状态判断
        # 如果 active_idx < 0，说明是为了计算 Task Embedding，只需返回原始特征
        if self.active_idx < 0:
            return original_out

        # 3. 选中指定的专家
        # 这个索引由外部的 Router 决定并设置
        selected_adapter = self.adapters[self.active_idx]
        adapter_out = selected_adapter(x)
        
        # 4. 并行相加
        return original_out + adapter_out.type(x.dtype)


# =========================================================================
# 3. Router (保持不变)
# =========================================================================
class AdapterRouter(nn.Module):
    def __init__(self, input_dim, pool_size=5):
        super().__init__()
        self.pool_size = pool_size
        self.input_dim = input_dim
        
        self.prompt_key = nn.Parameter(torch.randn(pool_size, input_dim))
        # === [新增] 初始化标志 ===
        # 标记是否已经用真实数据清洗过
        self.is_initialized = False
        
        # 记录当前 Epoch (需要外部更新)
        self.current_epoch = 0

    def get_best_expert_idx(self, task_embedding, training=True):
        # task_embedding shape: [Batch_Size, Dim] (Batch_Size 这里通常等于 N_way)
        
        # 1. 计算相似度 [Batch_Size, Pool_Size]
        keys_norm = F.normalize(self.prompt_key, p=2, dim=1)
        task_norm = F.normalize(task_embedding, p=2, dim=1)
        similarity = torch.matmul(task_norm, keys_norm.t()) 
        
        # 2. 策略选择
        if training:
            # 加上噪声防抖
            noise = torch.randn_like(similarity) * 0.1 
            noisy_similarity = similarity + noise
            topk_values, topk_indices = torch.topk(similarity, k=2, dim=1)
        
        # 计算 Softmax 权重，用于加权求和
        routing_weights = F.softmax(topk_values, dim=1) 
        
        # 统计 (只统计第一名，或者都统计)
        if training and hasattr(self, 'expert_usage_counts'):
            for idx in topk_indices[:, 0]: # 统计 Top-1
                self.expert_usage_counts[idx.item()] += 1
                
        return topk_indices, routing_weights