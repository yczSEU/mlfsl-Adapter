import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import open_clip
except ImportError:
    pass 

# =========================================================================
# 1. APART Adapter 单体
# =========================================================================
class APART_Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64, dropout=0.3): 
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.channel_scale_logits = nn.Parameter(torch.ones(input_dim) * -1.0) 
        self.post_norm = nn.LayerNorm(input_dim)

        self.max_scale = 0.10


        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.normal_(self.up_proj.weight,std=0.01)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        origin_dtype = x.dtype
        target_dtype = self.down_proj.weight.dtype
        x_in = x.to(target_dtype)
        down = self.down_proj(x_in)
        down = self.non_linear_func(down)
        down = self.dropout(down)
        up = self.up_proj(down)
        up = self.post_norm(up) 
        probs = torch.sigmoid(self.channel_scale_logits)
        min_s, max_s = 0.01, 0.10
        scale = min_s + (self.max_scale - min_s) * probs
        return (up * scale).to(origin_dtype)

# =========================================================================
# 2. Pool Wrapper (恢复纯净的 Top-1 路由注入 - 包含混合精度修复)
# =========================================================================
class APART_PoolWrapper(nn.Module):
    def __init__(self, original_mlp, input_dim, pool_size=5):
        super().__init__()
        self.original_mlp = original_mlp 
        self.adapters = nn.ModuleList([
            # 这里的 bottleneck_dim 建议保持在 16 或 32，对抗过拟合
            APART_Adapter(input_dim, bottleneck_dim=16) 
            for _ in range(pool_size)
        ])
        # 恢复为单体索引和权重
        self.active_idx = 0 
        self.active_weight = 1.0 

    def forward(self, x):
        # 记录原始输入精度 (通常是 FP16)
        origin_dtype = x.dtype 
        
        with torch.no_grad():
            orig = self.original_mlp(x)
        
        # 🚨 如果是 -1，说明是 pure_clip 阶段，原封不动返回原特征
        if self.active_idx < 0:
            return orig
            
        # ✅ 只有唯一胜出的专家被激活，保证特征绝对纯洁
        selected_adapter = self.adapters[self.active_idx]
        adapt = selected_adapter(x)
        w = self.active_weight
        
        if isinstance(w, torch.Tensor):
            if w.ndim == 1:
                if w.shape[0] != x.shape[0]: w = w.mean() 
                else: w = w.view(-1, 1, 1) if x.ndim == 3 else w.view(-1, 1)
            elif w.ndim > 1:
                w = w.squeeze() 
                if w.ndim == 1: w = w.view(-1, 1, 1) if x.ndim == 3 else w.view(-1, 1)

            # 在 FP32 下高精度融合，然后打回原精度返回
            out = orig.float() + adapt.float() * w.float()
            return out.to(origin_dtype)
            
        # 处理原生 float 权重的情况
        out = orig.float() + adapt.float() * float(w)
        return out.to(origin_dtype)

# =========================================================================
# 3. CoCoOp Router (强制 FP32 与 温度锐化)
# =========================================================================
class CoCoOpRouter(nn.Module):
    def __init__(self, clip_model, input_dim=512, pool_size=5, context_dim=512):
        super().__init__()
        self.pool_size = pool_size
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.clip_model = clip_model
        
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # ✅ 补回丢失的：拆除检查点与装载起搏器
        if hasattr(self.clip_model, 'set_grad_checkpointing'):
            self.clip_model.set_grad_checkpointing(False)
            
        if hasattr(self.clip_model, 'transformer'):
            self.clip_model.transformer.grad_checkpointing = False
            if hasattr(self.clip_model.transformer, 'resblocks'):
                for block in self.clip_model.transformer.resblocks:
                    if hasattr(block, 'grad_checkpointing'):
                        block.grad_checkpointing = False
                # 起搏器：强行解冻第一层的 LN
                self.clip_model.transformer.resblocks[0].ln_1.weight.requires_grad = True

        if hasattr(self.clip_model, 'ln_final'):
            # 起搏器：强行解冻最终的 LN
            self.clip_model.ln_final.weight.requires_grad = True
            self.clip_model.ln_final.bias.requires_grad = True

        if hasattr(self.clip_model, 'transformer'):
            self.clip_model.transformer = self.clip_model.transformer.float()
        if hasattr(self.clip_model, 'ln_final'):
            self.clip_model.ln_final = self.clip_model.ln_final.float()
        if hasattr(self.clip_model, 'text_projection'):
            if isinstance(self.clip_model.text_projection, torch.Tensor):
                self.clip_model.text_projection.data = self.clip_model.text_projection.data.float()
        if hasattr(self.clip_model, 'positional_embedding'):
            if isinstance(self.clip_model.positional_embedding, torch.Tensor):
                self.clip_model.positional_embedding.data = self.clip_model.positional_embedding.data.float()

        self.embed_dtype = torch.float32
        self.device = clip_model.token_embedding.weight.device if hasattr(clip_model, 'token_embedding') else 'cuda'
        self.transformer_dtype = torch.float32

        print("🧠 [CoCoOpRouter] Pacemaker ON, Graph Unblocked, FP32 Forced.")
            
        self.n_ctx = 8
        ctx_vectors = torch.empty(self.n_ctx, self.context_dim, dtype=self.embed_dtype)
        prompt_prefix = "a photo of a"
        
        try:
            import open_clip
            tokenized_prefix = open_clip.get_tokenizer('ViT-B-16')(prompt_prefix).to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.token_embedding(tokenized_prefix).type(self.embed_dtype)
                n_tokens = min(self.n_ctx, embedding.shape[1] - 2) 
                ctx_vectors[:n_tokens] = embedding[0, 1:1+n_tokens, :]
                if n_tokens < self.n_ctx:
                    ctx_vectors[n_tokens:] = embedding[0, n_tokens, :]
        except:
            nn.init.normal_(ctx_vectors, std=0.02)
            
        self.ctx = nn.Parameter(ctx_vectors) 
        
        self.meta_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 16, context_dim)
        ).to(dtype=self.embed_dtype)

        nn.init.normal_(self.meta_net[-1].weight, std=0.001)
        nn.init.zeros_(self.meta_net[-1].bias)

        # self.expert_descriptions = [
        #     "salient foreground object and main subject",            # 专家 0：专注提取显著的前景目标主体
        #     "background context, environment, and scene layout",     # 专家 1：专注提取背景环境和场景布局
        #     "texture, material, color, and surface details",         # 专家 2：专注提取纹理、材质、颜色等细粒度表面特征
        #     "interaction, spatial relationship, and action state",   # 专家 3：专注提取物体间的互动、空间位置和动作状态
        #     "shape, geometry, structure, and distinct boundaries"    # 专家 4：专注提取几何形状、结构和明显的轮廓边界
        # ]
        # self.expert_descriptions = [
        #     "person, pedestrian, and personal accessory",       # 专家 0
        #     "animal, bird, and natural wildlife",               # 专家 1
        #     "vehicle, transportation, and outdoor street",      # 专家 2
        #     "furniture, home appliance, and indoor scene",      # 专家 3
        #     "food, kitchenware, and sports equipment"           # 专家 4
        # ]
        # =================================================================
        # 🎯 ML-FSL 终极解：基于空间频率与功能属性的交叉锚点（COCO）
        # =================================================================
        
        self.expert_descriptions = [
            "a close-up focus on individual objects and their intricate details", # 专家0：专注局部细节与独立物体 (微观)
            "a wide view of outdoor environments, landscapes, and architecture",  # 专家1：专注户外大场景与建筑 (宏观-外)
            "an indoor setting with structured layouts and domestic items",       # 专家2：专注室内布局与家居 (宏观-内)
            "active elements involving motion, transport, and dynamic processes", # 专家3：专注运动、交通与动态过程 (动态)
            "organic life, nature, and biological entities in their habitats"     # 专家4：专注有机生命与自然生物 (生态)
        ]
        # =================================================================
        # 🎯 COCO 聚类优化版：绝对正交的 5 大实体域
        # =================================================================
        # self.expert_descriptions = [
        #     "dog, horse, bear, cow, and cat",                    # 专家 0：纯粹的动物与自然生物
        #     "car, bus, truck, traffic light, and stop sign",     # 专家 1：交通工具与街道标识 (合并了原簇2和3)
        #     "pizza, sandwich, cake, oven, and knife",            # 专家 2：食物与厨房用具 (清理了缝合怪)
        #     "surfboard, skateboard, kite, skis, and racket",     # 专家 3：纯粹的户外运动装备
        #     "tv, book, chair, bed, and laptop"                   # 专家 4：室内物品与家具家电
        # ]
        # =================================================================
        # 🎯 VG 聚类优化版：打破垄断的密集场景解构
        # =================================================================
        # self.expert_descriptions = [
        #     "man, boy, face, hand, and finger",                  # 专家 0：纯粹的人物主体与身体解剖部位 (拆解黑洞)
        #     "shirt, jacket, hat, shoe, and glove",               # 专家 1：衣物与穿戴饰品 (从人身上剥离，建立独立特长)
        #     "table, room, bed, chair, and cabinet",              # 专家 2：室内场景与木制家具
        #     "car, train, street, building, and sign",            # 专家 3：户外建筑、街道与交通网络
        #     "bag, box, bottle, bowl, and fruit"                  # 专家 4：独立的小型物品、容器与食物
        # ]
        self.init_fixed_tokens()
        self.expert_usage_counts = torch.zeros(pool_size)



    def init_fixed_tokens(self):
        prompts = [ " ".join(["X"] * self.n_ctx) + " " + name + "." for name in self.expert_descriptions ]
        tokenized = open_clip.get_tokenizer('ViT-B-16')(prompts).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized).type(self.embed_dtype)
        self.tokenized_prompts = tokenized
        self.register_buffer("prefix", embedding[:, :1, :])   
        self.register_buffer("suffix", embedding[:, 1+self.n_ctx:, :]) 

    def construct_prompts(self, image_features):
        image_features = image_features.to(dtype=self.embed_dtype)
        batch_size = image_features.shape[0]
        bias = self.meta_net(image_features).unsqueeze(1) 
        ctx = self.ctx.unsqueeze(0) 
        ctx_shifted = ctx + bias 
        ctx_expanded = ctx_shifted.unsqueeze(1).expand(-1, self.pool_size, -1, -1)
        prefix = self.prefix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        suffix = self.suffix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return torch.cat([prefix, ctx_expanded, suffix], dim=2)

    def get_dynamic_keys(self, prompts):
        b, n, l, d = prompts.shape
        x = prompts.view(b * n, l, d).float() 
        
        pos_emb = self.clip_model.positional_embedding[:l].float()
        x = x + pos_emb  # 此时形状: [Batch=50, Seq=77, Dim=512]
        
        # 🛡️ 绝对安全的 Causal Mask
        mask = torch.empty(l, l, device=x.device, dtype=x.dtype)
        mask.fill_(float("-inf"))
        mask.triu_(1) 
            
        if hasattr(self.clip_model, 'transformer') and hasattr(self.clip_model.transformer, 'resblocks'):
            # 💡 终极自适应策略：动态探测 Batch First 属性
            x_try = x.permute(1, 0, 2) # 先尝试标准老版本格式: [L, N, D] (即 77, 50, 512)
            try:
                for block in self.clip_model.transformer.resblocks:
                    x_try = block(x_try, attn_mask=mask)
                x = x_try.permute(1, 0, 2) # 如果没报错，说明跑通了，翻转回 [N, L, D]
            except RuntimeError as e:
                # 抓捕维度误判异常
                if "attn_mask" in str(e) or "shape" in str(e):
                    # 如果报错，说明模型底层是 batch_first=True，把 50 当成了句子长度！
                    # 我们直接使用原始的未翻转张量 x [N, L, D] (即 50, 77, 512)
                    for block in self.clip_model.transformer.resblocks:
                        x = block(x, attn_mask=mask)
                else:
                    raise e
        else:
            x = self.clip_model.transformer(x)

        x = self.clip_model.ln_final(x)
        
        tokens = self.tokenized_prompts.unsqueeze(0).expand(b, -1, -1).reshape(b*n, -1)
        x_out = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] 
        
        if hasattr(self.clip_model, 'text_projection'):
            x_out = x_out @ self.clip_model.text_projection
            
        return x_out.view(b, n, d)

    def get_best_expert_idx(self, task_embedding, training=True):
        prompts = self.construct_prompts(task_embedding)
        dynamic_keys = self.get_dynamic_keys(prompts)
        
        keys_feat = F.normalize(dynamic_keys.float(), p=2, dim=2)
        img_feat = F.normalize(task_embedding.float().unsqueeze(1), p=2, dim=2)
        
        # 获取原始余弦相似度 (通常在 0.1 ~ 0.4 之间)
        similarity = (img_feat * keys_feat).sum(dim=2)
        # 🛡️ 修复：在温度放大【之前】加入探索噪声！
        if training:
            # 引入更大的随机性，打破早期的同质化僵局
            noise = torch.randn_like(similarity) * 0.05
            similarity = similarity + noise
            
        # 🛡️ 修复：更极端的温度系数。将微小的差异成百倍放大，逼迫网络站队！
        temperature = 0.02
        scaled_sim = similarity / temperature
            
        full_weights = F.softmax(scaled_sim, dim=1)
        topk_values, topk_indices = torch.topk(scaled_sim, k=2, dim=1)
        routing_weights = F.softmax(topk_values, dim=1)
        
        if training:
            with torch.no_grad():
                self.expert_usage_counts = self.expert_usage_counts.to(topk_indices.device)
                counts = torch.bincount(topk_indices[:, 0], minlength=self.pool_size)
                self.expert_usage_counts += counts
                
        return topk_indices, routing_weights, full_weights