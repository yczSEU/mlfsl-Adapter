import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================================================================
# 1. APART Adapter 单体 (保持之前调试好的最佳逻辑不变)
# =========================================================================
class APART_Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64, dropout=0.1): 
        super().__init__()
        
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
        # === 核心设置 (保持不变) ===
        # 使用 Logits 控制 Scale
        # 初始值 0.0 -> Sigmoid(0.0)=0.5 -> Scale=0.025 (平稳起步)
        self.scale_logits = nn.Parameter(torch.tensor(0.0))
        
        # 强制最大 Scale 不超过 0.05 (甜点位)
        self.max_scale_val = 0.05

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
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
        
        # 3. 计算受限 Scale (Sigmoid 机制)
        current_scale = self.max_scale_val * torch.sigmoid(self.scale_logits)
        
        out = up * current_scale
        
        # 4. 转回原精度
        return out.to(origin_dtype)


# =========================================================================
# 2. Pool Wrapper (并行池化包装器)
# =========================================================================
class APART_PoolWrapper(nn.Module):
    def __init__(self, original_mlp, input_dim, pool_size=5):
        super().__init__()
        self.original_mlp = original_mlp 
        
        # 创建专家池：包含 pool_size 个独立的 Adapter
        self.adapters = nn.ModuleList([
            APART_Adapter(input_dim, bottleneck_dim=64) 
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
# 3. Router (路由器)
# =========================================================================
class AdapterRouter(nn.Module):
    def __init__(self, input_dim, pool_size=5):
        super().__init__()
        self.pool_size = pool_size
        self.input_dim = input_dim
        
        # 专家 Key：形状 [pool_size, input_dim]
        # 初始化为正交矩阵，让专家初始状态尽量“互不相同”
        self.prompt_key = nn.Parameter(torch.randn(pool_size, input_dim))
        nn.init.orthogonal_(self.prompt_key)

    def get_best_expert_idx(self, task_embedding):
        """
        根据 Support Set 的任务特征，选择最匹配的专家。
        task_embedding: [1, input_dim]
        返回: int (最佳专家 ID)
        """
        # 归一化 (Cosine Similarity 前置步骤)
        keys_norm = F.normalize(self.prompt_key, p=2, dim=1)
        task_norm = F.normalize(task_embedding, p=2, dim=1)
        
        # 计算相似度: [1, dim] @ [dim, pool_size] -> [1, pool_size]
        similarity = torch.matmul(task_norm, keys_norm.t())
        
        # 选分最高的专家
        best_idx = torch.argmax(similarity, dim=1).item()
        return best_idx