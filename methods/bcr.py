from torch import nn
from methods.template import MLLTemplate
import torch
import torch.nn.functional as F

class BCR(MLLTemplate):
    def __init__(self, model_func, n_way, n_shot, n_query, eta=0.01, gamma=0.01,
                 hidden_dim=100, device='cuda:0', verbose=False):
        super(BCR, self).__init__(model_func=model_func, n_way=n_way, n_shot=n_shot, n_query=n_query,
                                  device=device, verbose=verbose)
        self.eta = eta
        self.gamma = gamma
        
        # === 原始组件 ===
        self.encoder_x = nn.Linear(self.feat_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(self.feat_dim) # 核心
        self.encoder_y = nn.Linear(self.n_way, hidden_dim)
        self.encoder_z = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # === 优化器 (为了 Adapter 安全，我保留了分层学习率) ===
        # 如果你想完全复原，可以改回统一的 1e-3，但我强烈建议 Adapter 用小一点
        backbone_params = []
        scale_params = []
        head_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            
            if 'scale' in name:
                scale_params.append(param)
            elif 'feature_extractor' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        # 这里的 lr 设置为 1e-5 是为了防止 Adapter 崩盘
        # BCR 头部依然是 1e-3
        self.optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': 1e-5}, 
            {'params': scale_params,    'lr': 1e-3, 'weight_decay': 1e-3}, 
            {'params': head_params,     'lr': 1e-3}
        ])
        
        # 加上调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=30, eta_min=1e-6
        )
        self.to(self.device)

    # 安全阈值
    EPS = 1e-6

    def clean_multi_labels(self, x_support, y_support, y_query=None):
        """
        保留这个清洗函数，防止 VG 数据集报错 (维度不匹配问题)
        """
        n_shot = x_support.shape[0] // self.n_way
        target_indices = []
        
        for i in range(self.n_way):
            start_idx = i * n_shot
            end_idx = (i + 1) * n_shot
            if start_idx >= y_support.shape[0]: 
                target_indices.append(0)
                continue
            
            chunk = y_support[start_idx : end_idx]
            if chunk.shape[0] > 0:
                class_idx = chunk.sum(dim=0).argmax().item()
                target_indices.append(class_idx)
            else:
                target_indices.append(0)
        
        # 截断或补齐
        if len(target_indices) > self.n_way:
            target_indices = target_indices[:self.n_way]
        elif len(target_indices) < self.n_way:
            target_indices.extend([0] * (self.n_way - len(target_indices)))
            
        y_support_clean = y_support[:, target_indices]
        y_query_clean = y_query[:, target_indices] if y_query is not None else None
        
        return y_support_clean, y_query_clean

    def set_forward(self, x_support, y_support, x_query):
        # 这里的清洗主要是为了防止推理阶段 VG 数据集报错
        if y_support.shape[1] > self.n_way:
            y_support, _ = self.clean_multi_labels(x_support, y_support)

        y_support = y_support.float()
        z_support = self.feature_extractor(x_support)
        z_query = self.feature_extractor(x_query)
        
        # LayerNorm + Normalize
        z_support = self.input_norm(z_support)
        z_query = self.input_norm(z_query)
        z_support = F.normalize(z_support, p=2, dim=-1)
        z_query = F.normalize(z_query, p=2, dim=-1)
        
        weight = y_support / torch.sum(y_support, dim=0, keepdim=True)
        proto = torch.transpose(weight, 0, 1) @ z_support
        
        sim = torch.relu(self.cosine_similarity(z_support, proto))
        attention = y_support * sim + 1e-7
        attention = attention / torch.sum(attention, dim=0, keepdim=True)
        proto = torch.transpose(attention, 0, 1) @ z_support
        
        # 原始公式：不转 float，保持 FP16 (如果你显存够，转 float 更稳，不转也行)
        scores = -self.euclidean_dist(z_query, proto) / 1.0
        scores = torch.sigmoid(scores) * 2
        
        # 基本的数值保护
        scores = torch.clamp(scores, min=self.EPS, max=1.0-self.EPS)
        return scores

    def set_forward_loss(self, x_support, y_support, x_query, y_query):
        # VG 清洗逻辑
        if y_support.shape[1] > self.n_way:
            y_support, y_query = self.clean_multi_labels(x_support, y_support, y_query)

        y_support = y_support.float()
        z_support = self.feature_extractor(x_support)
        z_query = self.feature_extractor(x_query)
        
        z_support = self.input_norm(z_support)
        z_query = self.input_norm(z_query)
        z_support = F.normalize(z_support, p=2, dim=-1)
        z_query = F.normalize(z_query, p=2, dim=-1)
        
        weight = y_support / torch.sum(y_support, dim=0, keepdim=True)
        proto = torch.transpose(weight, 0, 1) @ z_support
        sim = torch.relu(self.cosine_similarity(z_support, proto))
        attention = y_support * sim + 1e-7
        attention = attention / torch.sum(attention, dim=0, keepdim=True)
        proto = torch.transpose(attention, 0, 1) @ z_support
        
        # 原始公式
        scores = -self.euclidean_dist(z_query, proto) / 1.0
        scores = torch.sigmoid(scores) * 2
        scores = torch.clamp(scores, min=self.EPS, max=1.0-self.EPS)
        
        loss_cls = nn.BCELoss()(scores, y_query)
        
        # ------------------------ LE loss ------------------------
        # 唯一保留的“现代”修改：Detach
        # 这是防止 Adapter 被辅助任务带偏的关键
        z_support_detached = z_support.detach()
        z_query_detached = z_query.detach()
        
        x = torch.cat([z_support_detached, z_query_detached], dim=0)
        y = torch.cat([y_support, y_query], dim=0)
        
        dx = self.encoder_x(x)
        dy = self.encoder_y(y)
        dz = self.encoder_z(torch.concat([dx, dy], dim=1))
        
        S = self.cosine_similarity(dz, dz)
        yy = S @ y
        loss_cl = nn.BCEWithLogitsLoss()(yy, y)
        
        weight = y / torch.sum(y, dim=0, keepdim=True)
        proto = torch.transpose(weight, 0, 1) @ x
        sim = torch.relu(self.cosine_similarity(x, proto))
        attention = y * sim + 1e-7
        attention = attention / torch.sum(attention, dim=0, keepdim=True)
        proto = torch.transpose(attention, 0, 1) @ x
        
        sscores = -self.euclidean_dist(x, proto) / 64.0
        loss_li = nn.CrossEntropyLoss()(sscores, torch.softmax(yy, dim=1))
        
        return loss_cls + loss_cl * self.eta + loss_li * self.gamma