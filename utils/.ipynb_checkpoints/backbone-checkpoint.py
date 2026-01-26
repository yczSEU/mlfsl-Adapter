from torchvision.models.resnet import resnet50, resnet101
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import os
import open_clip

# === 【关键修改】引入 Adapter Pool 模块 ===
# 确保你已经创建了 utils/adapter_pool.py 文件
from utils.adapter_pool import APART_PoolWrapper, AdapterRouter

# -------------------------------------------------------------------------
# Backbone 定义
# -------------------------------------------------------------------------

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

class OpenCLIPBackbone(nn.Module):
    def __init__(self, model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', 
                 use_adapter=False, pool_size=5): # 新增 pool_size 参数
        super(OpenCLIPBackbone, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading OpenCLIP: {model_name}")
        if use_adapter:
            print(f"Mode: Adapter Pool (Size={pool_size}, Task-Aware)")
        else:
            print(f"Mode: Pure CLIP (Baseline)")

        self.model = open_clip.create_model(model_name, pretrained=pretrained, 
                                            precision='fp16' if self.device == 'cuda' else 'fp32',
                                            device=self.device)
        
        # 自动获取精度
        if hasattr(self.model.visual, 'conv1'):
            self.target_dtype = self.model.visual.conv1.weight.dtype
        else:
            self.target_dtype = next(self.model.visual.parameters()).dtype

        # 1. 全局冻结
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.use_adapter = use_adapter
        
        # 2. 注入 Adapter Pool
        if use_adapter:
            embed_dim = 768 
            if hasattr(self.model.visual, 'width'): embed_dim = self.model.visual.width
            
            # === [修改 1] 初始化 Router ===
            # Router 负责根据 Support Set 特征选专家
            self.router = AdapterRouter(input_dim=embed_dim, pool_size=pool_size).to(self.device)
            
            print(f"Injecting Adapter Pool (Size={pool_size}, Dim=64, Limit=0.05, Init=0.01)...")
            
            # === [修改 2] 替换 MLP 为 PoolWrapper ===
            for i, block in enumerate(self.model.visual.transformer.resblocks):
                old_mlp = block.mlp
                # 这里的 pool_size 必须和 Router 一致
                block.mlp = APART_PoolWrapper(old_mlp, input_dim=embed_dim, pool_size=pool_size).to(self.device)
            
            # 打印可训练参数
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"Injection Done! Trainable Params: {trainable} ({trainable/total:.2%})")

        # 计算 Feature Dim
        if hasattr(self.model.visual, 'image_size'):
            img_size = self.model.visual.image_size
            if isinstance(img_size, tuple): img_size = img_size[0]
        else: img_size = 224
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size, device=self.device, dtype=self.target_dtype)
            # 这里调用原始 forward 也没关系，因为默认 adapter_idx=0
            out = self.model.encode_image(dummy)
            self.final_feat_dim = out.shape[1]

    def get_task_embedding(self, support_images):
        """
        新功能：计算 Support Set 的任务向量 (Task Embedding)
        使用【纯 CLIP 模式】提取特征，不经过 Adapter。
        """
        with torch.no_grad():
            # 设为 True，跳过 Adapter 计算
            features = self.forward(support_images, use_pure_clip=True)
            # 平均得到任务向量 [1, dim]
            task_emb = features.mean(dim=0, keepdim=True) 
            return task_emb

    def forward(self, x, adapter_idx=0, use_pure_clip=False):
        """
        x: 图片 Tensor
        adapter_idx: 选定的专家 ID (由 Router 计算得出)
        use_pure_clip: 如果为 True，则跳过 Adapter (用于计算 Task Embedding)
        """
        if x.device != self.device: x = x.to(self.device)
        if x.dtype != self.target_dtype: x = x.type(self.target_dtype)
        
        # === 核心逻辑：设置全局专家状态 ===
        if self.use_adapter:
            # 决定目标索引：如果是纯 CLIP 模式，设为 -1
            target_idx = -1 if use_pure_clip else adapter_idx
            
            # 遍历所有层，设置当前激活的专家
            for block in self.model.visual.transformer.resblocks:
                if hasattr(block.mlp, 'active_idx'):
                    block.mlp.active_idx = target_idx

        # 调用 OpenCLIP 原始推理流程 (它会调用我们修改过的 block.mlp)
        return self.model.encode_image(x).float()

# -------------------------------------------------------------------------
# 模型入口函数
# -------------------------------------------------------------------------

def vit_b_openclip():
    # 纯净版 ViT-B (Baseline)
    return OpenCLIPBackbone(model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', use_adapter=False)

def vit_b_adapter():
    # APART Adapter Pool 版 ViT-B
    # 默认 pool_size=5 (你可以根据显存大小调整，显存不够改小点)
    return OpenCLIPBackbone(model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', use_adapter=True, pool_size=5)

def vit_h_openclip():
    return OpenCLIPBackbone(model_name='ViT-H-14-378-quickgelu', pretrained='dfn5b')
    
# -------------------------------------------------------------------------
# 旧 ResNet 代码 (保留占位，防止报错)
# -------------------------------------------------------------------------
class Conv64F(nn.Module):
    def __init__(self, **kwargs): super().__init__()
    def forward(self, x): return x
def Conv4NP(): return Conv64F()
def ResNet18(flatten=True): return Conv64F()
def ResNet18R(flatten=False): return Conv64F()
def ResNet50(): return Conv64F()
def ResNet101(): return Conv64F()

model_dict = {
    'Conv4': Conv64F,
    'Conv4R': Conv4NP,
    'ResNet18': ResNet18,
    'ResNet18R': ResNet18R,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ViT-H-CLIP': vit_h_openclip,
    'ViT-B-CLIP': vit_b_openclip,
    'ViT-B-Adapter': vit_b_adapter,
}