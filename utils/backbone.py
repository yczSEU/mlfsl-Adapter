from torchvision.models.resnet import resnet50, resnet101
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import os
import open_clip

from utils.adapter_pool import APART_PoolWrapper, AdapterRouter

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

# utils/backbone.py

class OpenCLIPBackbone(nn.Module):
    def __init__(self, model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', 
                 use_adapter=False, pool_size=5):
        super(OpenCLIPBackbone, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading OpenCLIP: {model_name}")
        self.model = open_clip.create_model(model_name, pretrained=pretrained, 
                                            precision='fp16' if self.device == 'cuda' else 'fp32',
                                            device=self.device)
        
        if hasattr(self.model.visual, 'conv1'):
            self.target_dtype = self.model.visual.conv1.weight.dtype
        else:
            self.target_dtype = next(self.model.visual.parameters()).dtype

        # 维度计算
        transformer_width = 768 
        if hasattr(self.model.visual, 'width'): 
            transformer_width = self.model.visual.width
            
        if hasattr(self.model.visual, 'image_size'):
            img_size = self.model.visual.image_size
            if isinstance(img_size, tuple): img_size = img_size[0]
        else: img_size = 224
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size, device=self.device, dtype=self.target_dtype)
            out = self.model.encode_image(dummy)
            self.final_feat_dim = out.shape[1]

        for param in self.model.parameters():
            param.requires_grad = False
            
        self.use_adapter = use_adapter
        
        if use_adapter:
            print(f"Mode: Adapter Pool (Size={pool_size})")
            # Router 的 bottleneck_dim 也要传，如果 Router 内部用了线性层的话 (你的 Router 代码看起来没用 bottleneck，但传进去无妨)
            self.router = AdapterRouter(input_dim=self.final_feat_dim, pool_size=pool_size).to(self.device)
            
            # 2. [修改这里] 把死数字 64 改成变量 {bottleneck_dim}
            print(f"Injecting Adapter Pool (Size={pool_size}, Dim=256, Limit=0.05, Init=0.01)...")
            
            for i, block in enumerate(self.model.visual.transformer.resblocks):
                old_mlp = block.mlp
                # 3. [关键] 把 bottleneck_dim 传给 Wrapper，Wrapper 再传给 Adapter
                block.mlp = APART_PoolWrapper(old_mlp, input_dim=transformer_width, 
                                              pool_size=pool_size).to(self.device)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"Injection Done! Trainable Params: {trainable} ({trainable/total:.2%})")

    def get_task_embedding(self, support_images):
        with torch.no_grad():
            # 这里必须显式传 use_pure_clip=True，这会触发 forward 里的 -1 逻辑
            features = self.forward(support_images, use_pure_clip=True)
            task_emb = features.mean(dim=0, keepdim=True) 
            return task_emb

    def forward(self, x, adapter_idx=None, use_pure_clip=False): # <--- [修改1] 默认改为 None
        """
        x: 图片 Tensor
        adapter_idx: 指定专家 ID。如果为 None，则保持当前状态不变！
        use_pure_clip: 强制使用纯 CLIP (Active Idx = -1)
        """
        if x.device != self.device: x = x.to(self.device)
        if x.dtype != self.target_dtype: x = x.type(self.target_dtype)
        
        if self.use_adapter:
            # [修改2] 只有在明确指定了 idx 或者要求纯 CLIP 时，才去修改状态
            # 如果 adapter_idx 是 None，说明外部已经通过 _set_active_adapter 设置好了，这里不要动！
            if use_pure_clip:
                target_idx = -1
                for block in self.model.visual.transformer.resblocks:
                    if hasattr(block.mlp, 'active_idx'):
                        block.mlp.active_idx = target_idx
            elif adapter_idx is not None:
                for block in self.model.visual.transformer.resblocks:
                    if hasattr(block.mlp, 'active_idx'):
                        block.mlp.active_idx = adapter_idx

        return self.model.encode_image(x).float()
# -------------------------------------------------------------------------
# 模型入口函数
# -------------------------------------------------------------------------

def vit_b_openclip():
    return OpenCLIPBackbone(model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', use_adapter=False)

def vit_b_adapter():
    return OpenCLIPBackbone(model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', use_adapter=True, pool_size=5)

def vit_h_openclip():
    return OpenCLIPBackbone(model_name='ViT-H-14-378-quickgelu', pretrained='dfn5b')
    
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