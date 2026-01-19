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
        
        # === BCR 特有的 Projection Heads (随机初始化) ===
        # 这些层需要较大的 LR (1e-3) 来快速学习
        self.encoder_x = nn.Linear(self.feat_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(self.feat_dim) 
        self.encoder_y = nn.Linear(self.n_way, hidden_dim)
        self.encoder_z = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # === 核心修改：分层学习率 (Layer-wise Learning Rate) ===
        # 1. 区分 Adapter 参数 (Backbone) 和 BCR 头参数 (Heads)
        backbone_params = []
        scale_params = []
        head_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue # 跳过冻结参数
            
            # 判断逻辑：如果参数属于 feature_extractor，那就是 Adapter
            # 否则就是 encoder_x, encoder_y 等 BCR 头部
           # 1. 如果参数名包含 'scale'，放入 scale_params (学习率要大)
            if 'scale' in name:
                scale_params.append(param)
            # 2. 其他属于 feature_extractor 的是 Adapter 内部参数 (学习率要小)
            elif 'feature_extractor' in name:
                backbone_params.append(param)
            # 3. 剩下的就是 encoder_x, encoder_y 等 BCR 头部
            else:
                head_params.append(param)

        # 2. 构建优化器参数组
        # - Adapter (Backbone): 1e-4 (0.1x, 稳健微调，参考 APART)
        # - Heads (BCR): 1e-3 (1.0x, 快速收敛，参考 BCR 原文)
        self.optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': 1e-4}, 
            {'params': scale_params, 'lr': 1e-2}, 
            {'params': head_params, 'lr': 1e-3}
        ])
        
        self.to(self.device)

    # === 保持之前的安全设置 ===
    EPS = 1e-5

    def set_forward(self, x_support, y_support, x_query):
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
        
        # 强制 FP32 计算
        z_query_f = z_query.float()
        proto_f = proto.float()
        
        scores = -self.euclidean_dist(z_query_f, proto_f) / 1.0
        scores = torch.sigmoid(scores) * 2
        
        # === 修改点 3：软截断 (Soft Clamp) ===
        # 即使模型想输出 1.0，我们强制它只能输出 0.99999
        # 这样 log(1-x) 永远不会遇到 log(0)
        scores = torch.nan_to_num(scores, nan=0.0)
        scores = torch.clamp(scores, min=self.EPS, max=1.0-self.EPS) 
        
        return scores

    def set_forward_loss(self, x_support, y_support, x_query, y_query):
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
        
        # 强制 FP32 计算
        z_query_f = z_query.float()
        proto_f = proto.float()
        
        scores = -self.euclidean_dist(z_query_f, proto_f) / 1.0
        scores = torch.sigmoid(scores) * 2
        
        # === 修改点 4：软截断 (Soft Clamp) ===
        scores = torch.nan_to_num(scores, nan=0.0)
        scores = torch.clamp(scores, min=self.EPS, max=1.0-self.EPS)
        
        loss_cls = nn.BCELoss()(scores, y_query)
        
        # LE loss calculation
        x = torch.cat([z_support, z_query], dim=0)
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
        
        x_f = x.float()
        proto_2_f = proto.float()
        
        sscores = -self.euclidean_dist(x_f, proto_2_f) / 1.0 
        
        loss_li = nn.CrossEntropyLoss()(sscores, torch.softmax(yy, dim=1))
        
        return loss_cls + loss_cl * self.eta + loss_li * self.gamma