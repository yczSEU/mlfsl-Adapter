from torch import nn
from methods.template import MLLTemplate
import torch
import torch.nn.functional as F
import numpy as np

class BCR(MLLTemplate):
    def __init__(self, model_func, n_way, n_shot, n_query, eta=0.01, gamma=0.01,
                 hidden_dim=512, device='cuda:0', verbose=False):
        super(BCR, self).__init__(model_func=model_func, n_way=n_way, n_shot=n_shot, n_query=n_query,
                                  device=device, verbose=verbose)
        self.eta = eta
        self.gamma = gamma
        
        self.encoder_x = nn.Linear(self.feat_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(self.feat_dim) 
        self.encoder_y = nn.Linear(self.n_way, hidden_dim)
        self.encoder_z = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.fidelity_history = []
        self.print_counter = 0
        self.EPS = 1e-6
        self.test_expert_history = [] 

        self.beta = 8.0          
        self.alpha_balance = 0.1 
        self.ortho_weight = 0.5  
        self.router_weight = 1.0 
        
        # =========================================================
        # 优化器配置
        # =========================================================
        router_params = []
        adapter_weights = []
        adapter_scalars = []
        head_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            
            if 'router' in name:
                router_params.append(param)
            elif 'adapter' in name or 'scale' in name or 'feature_extractor' in name:
                if 'bias' in name or 'norm' in name or 'scale' in name:
                    adapter_scalars.append(param)
                else:
                    adapter_weights.append(param)
            else:
                head_params.append(param)

        print(f"🔧 Optimizer Groups:")
        print(f"   Router Params: {len(router_params)} (LR=0.002)")
        print(f"   Adapter Weights: {len(adapter_weights)} (LR=5e-4, WD=1e-3)")
        print(f"   Adapter Scalars: {len(adapter_scalars)} (LR=5e-4, WD=0.0)")
        print(f"   Head Params: {len(head_params)} (LR=1e-3)")

        self.optimizer = torch.optim.Adam([
            {'params': adapter_weights, 'lr': 0.0005, 'weight_decay': 1e-3},
            {'params': adapter_scalars, 'lr': 0.0005, 'weight_decay': 0.0},
            {'params': router_params,   'lr': 0.002,  'weight_decay': 1e-4}, 
            {'params': head_params,     'lr': 0.001}
        ])
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[3, 6], gamma=0.1
        )
        self.to(self.device)
        print("✅ BCR Model Initialized. Using CoCoOp Router (Auto-Initialized).")

    def reset_test_stats(self):
        self.test_expert_history = []
        print("[Stats] Test history reset.")

    def print_test_stats(self):
        if len(self.test_expert_history) == 0:
            print("[Stats] No data recorded.")
            return
        all_indices = torch.cat(self.test_expert_history).cpu().numpy()
        if len(all_indices.shape) > 1:
            all_indices = all_indices[:, 0]
            
        total_decisions = len(all_indices)
        print("\n" + "="*40)
        print(f"📊 [Test Phase Router Statistics (Top-1)]")
        print(f"   Total Class Decisions: {total_decisions}")
        print("-" * 40)
        unique, counts = np.unique(all_indices, return_counts=True)
        stats = dict(zip(unique, counts))
        pool_size = self.feature_extractor.router.pool_size if hasattr(self.feature_extractor, 'router') else 5
        for i in range(pool_size):
            count = stats.get(i, 0)
            ratio = count / total_decisions * 100
            bar = "█" * int(ratio // 2) 
            print(f"   Expert {i}: {count:6d} ({ratio:5.1f}%) | {bar}")
        print("="*40 + "\n")

    def _set_active_adapter(self, idx, weight=1.0):
        # ✅ 恢复接收单体 idx 和 weight
        if hasattr(self.feature_extractor, 'use_adapter') and self.feature_extractor.use_adapter:
            model = self.feature_extractor.model
            if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
                blocks = model.visual.transformer.resblocks
            elif hasattr(model, 'blocks'):
                blocks = model.blocks
            else: return
            
            for block in blocks:
                if hasattr(block.mlp, 'active_idx'):
                    block.mlp.active_idx = idx
                    block.mlp.active_weight = weight

    # =========================================================================
    # 测试逻辑 (回归 Top-1 Hard Routing)
    # =========================================================================
    def set_forward(self, x_support, y_support, x_query):
        n_way = self.n_way
        n_shot = self.n_shot
        n_query = x_query.shape[0]

        x_support_view = x_support.view(n_way, n_shot, *x_support.shape[-3:])
        z_support_all = torch.zeros(n_way * n_shot, self.feat_dim, device=x_support.device)

        with torch.no_grad():
            flat_support = x_support.reshape(-1, *x_support.shape[-3:])
            raw_feats = self.feature_extractor(flat_support, use_pure_clip=True)
            class_embs = raw_feats.view(n_way, n_shot, -1).mean(dim=1)
            
            best_indices_supp, routing_weights_supp, _ = self.feature_extractor.router.get_best_expert_idx(class_embs, training=False)
            
            # ✅ 坚决只取第一名
            top1_indices_supp = best_indices_supp[:, 0]
            top1_w_supp = routing_weights_supp[:, 0] 
            
            if not self.training:
                self.test_expert_history.append(top1_indices_supp.detach().cpu())

        # Support 提取 
        for class_id in range(n_way):
            idx = top1_indices_supp[class_id] 
            w_supp = top1_w_supp[class_id] 
            imgs = x_support_view[class_id]
            
            self._set_active_adapter(idx.item(), weight=w_supp) 
            feat = self.feature_extractor(imgs)
            start = class_id * n_shot
            z_support_all[start : start+n_shot] = feat

        # Query 提取
        z_query_all = torch.zeros(n_query, self.feat_dim, device=x_query.device)
        with torch.no_grad():
             query_feats_clip = self.feature_extractor(x_query, use_pure_clip=True)
             proto_norm = F.normalize(class_embs, p=2, dim=-1)
             query_norm = F.normalize(query_feats_clip, p=2, dim=-1)
             sim = torch.matmul(query_norm, proto_norm.t())
             assigned_class_idx = sim.argmax(dim=1) 
             
             query_top1_idx = top1_indices_supp[assigned_class_idx] 
             query_top1_w = top1_w_supp[assigned_class_idx]
        
        unique_experts = torch.unique(query_top1_idx)
        for exp_id in unique_experts:
            mask = (query_top1_idx == exp_id)
            if mask.any():
                imgs_sub = x_query[mask]
                w_sub = query_top1_w[mask] 
                
                self._set_active_adapter(exp_id.item(), weight=w_sub)
                f = self.feature_extractor(imgs_sub)
                z_query_all[mask] = f

        # --- BCR Head Forward ---
        z_support = self.input_norm(z_support_all)
        z_query = self.input_norm(z_query_all)
        z_support = F.normalize(z_support, p=2, dim=-1) 
        z_query = F.normalize(z_query, p=2, dim=-1)
        
        y_support = y_support.float()
        weight = y_support / torch.sum(y_support, dim=0, keepdim=True)
        proto = torch.transpose(weight, 0, 1) @ z_support
        sim = torch.relu(self.cosine_similarity(z_support, proto))
        attention = y_support * sim + 1e-7
        attention = attention / torch.sum(attention, dim=0, keepdim=True)
        proto = torch.transpose(attention, 0, 1) @ z_support
        
        scores = -self.euclidean_dist(z_query, proto) / 1.0 
        scores = torch.sigmoid(scores) * 2
        scores = torch.clamp(scores, min=self.EPS, max=1.0-self.EPS)
        return scores

    # =========================================================================
    # 训练逻辑 (回归 Top-1，允许长尾分布，放宽 Fidelity)
    # =========================================================================
    def set_forward_loss(self, x_support, y_support, x_query, y_query):
        n_way = self.n_way
        n_shot = self.n_shot
        n_query = x_query.shape[0]
        
        x_support_view = x_support.view(n_way, n_shot, *x_support.shape[-3:])
        with torch.no_grad():
            flat_support = x_support.reshape(-1, *x_support.shape[-3:])
            raw_feats = self.feature_extractor(flat_support, use_pure_clip=True)
            class_embs = raw_feats.view(n_way, n_shot, -1).mean(dim=1)
        
        best_indices, routing_weights, full_router_probs = self.feature_extractor.router.get_best_expert_idx(class_embs, training=True)
        
        # ✅ 坚决只提取第一名
        top1_indices = best_indices[:, 0]   
        top1_weights = routing_weights[:, 0] 
        
        pool_size = self.feature_extractor.router.pool_size
        avg_usage = full_router_probs.mean(dim=0) + 1e-6 # <--- 补回这一行！计算当前 Batch 的专家使用率

        
        # target_usage = torch.full_like(avg_usage, 1.0 / pool_size)
        # load_balance_loss = torch.sum(avg_usage * torch.log(avg_usage / target_usage))
        # ✅ 彻底砸碎 KL 均分枷锁！允许长尾分布的自然演化
        load_balance_loss = torch.tensor(0.0, device=self.device)
        
        # ---------------- Support 提取 ----------------
        z_support_list = []
        for class_id in range(n_way):
            idx = top1_indices[class_id]
            w = top1_weights[class_id] 
            imgs = x_support_view[class_id]
            
            self._set_active_adapter(idx.item(), weight=w)
            feat = self.feature_extractor(imgs)
            z_support_list.append(feat)
            
        z_support_all = torch.cat(z_support_list, dim=0)

        # ---------------- Query 提取 ----------------
        z_query_list = [None] * n_query 
        query_class_ids = y_query.argmax(dim=1) 
        query_top1_idx = top1_indices[query_class_ids]
        query_top1_w = top1_weights[query_class_ids] 
        
        unique_experts = torch.unique(query_top1_idx)
        for exp_id in unique_experts:
            mask = (query_top1_idx == exp_id)
            if mask.any():
                imgs_sub = x_query[mask]
                w_sub = query_top1_w[mask]
                
                self._set_active_adapter(exp_id.item(), weight=w_sub)
                f = self.feature_extractor(imgs_sub)
                
                indices = torch.nonzero(mask).squeeze(1)
                for i, idx_val in enumerate(indices):
                    z_query_list[idx_val] = f[i].unsqueeze(0)
                    
        z_query_all = torch.cat(z_query_list, dim=0)

        # 5. Fidelity Loss (保真度损失)
        sim_scores = F.cosine_similarity(z_support_all.view(-1, self.feat_dim), raw_feats, dim=-1)
        mean_sim = sim_scores.mean()
        self.fidelity_history.append(mean_sim.item())
        
        # ✅ 极限压榨操作：放宽 5-shot 的保真度底线，允许模型大胆改变特征！
        if self.n_shot == 1:
            fidelity_threshold = getattr(self, 'fidelity_threshold_1shot', 0.90)
        else:
            fidelity_threshold = getattr(self, 'fidelity_threshold_5shot', 0.80) # 以前是 0.85

        loss_fidelity = torch.relu(torch.tensor(fidelity_threshold, device=self.device) - mean_sim)

        # --- BCR Head 计算 ---
        z_support = self.input_norm(z_support_all)
        z_query = self.input_norm(z_query_all)
        z_support = F.normalize(z_support, p=2, dim=-1) 
        z_query = F.normalize(z_query, p=2, dim=-1)
        
        y_support = y_support.float()
        weight = y_support / torch.sum(y_support, dim=0, keepdim=True)
        proto = torch.transpose(weight, 0, 1) @ z_support
        sim = torch.relu(self.cosine_similarity(z_support, proto))
        attention = y_support * sim + 1e-7
        attention = attention / torch.sum(attention, dim=0, keepdim=True)
        proto = torch.transpose(attention, 0, 1) @ z_support
        
        scores = -self.euclidean_dist(z_query, proto) / 1.0 
        scores = torch.sigmoid(scores) * 2
        scores = torch.clamp(scores, min=self.EPS, max=1.0-self.EPS)
        
        loss_cls = nn.BCELoss()(scores, y_query)

        # =====================================================================
        # 💉 [深度诊断探针 1 & 2]：特征偏移量与原型分离度监控
        # =====================================================================
        with torch.no_grad():
            # 1. 监控 Adapter 对特征的修改幅度 (L2范数距离)
            # 如果 < 0.1: Adapter 基本没工作; 如果 > 5.0: 预训练特征被严重破坏
            feat_shift = torch.norm(z_support_all.view(-1, self.feat_dim) - raw_feats, dim=-1).mean().item()
            
            # 2. 监控各类别原型在空间中的平均距离 (分离度)
            # proto 的形状是 [feat_dim, n_way]，转置后计算两两之间的欧氏距离
            proto_t = proto.t()
            if proto_t.shape[0] > 1:
                proto_dist = torch.pdist(proto_t).mean().item()
            else:
                proto_dist = 0.0
        # =====================================================================
        
        x = torch.cat([z_support, z_query], dim=0) 
        y = torch.cat([y_support, y_query], dim=0)
        dx = self.encoder_x(x); dy = self.encoder_y(y); dz = self.encoder_z(torch.concat([dx, dy], dim=1))
        S = self.cosine_similarity(dz, dz); yy = S @ y; loss_cl = nn.BCEWithLogitsLoss()(yy, y)
        weight = y / torch.sum(y, dim=0, keepdim=True); proto = torch.transpose(weight, 0, 1) @ x
        sim = torch.relu(self.cosine_similarity(x, proto)); attention = y * sim + 1e-7
        attention = attention / torch.sum(attention, dim=0, keepdim=True); proto = torch.transpose(attention, 0, 1) @ x
        sscores = -self.euclidean_dist(x, proto) / 64.0; loss_li = nn.CrossEntropyLoss()(sscores, torch.softmax(yy, dim=1))
        
        w_bal    = getattr(self, 'alpha_balance', 0.05) # 保持 0.0
        w_beta   = getattr(self, 'beta', 8.0)
        
        total_loss = loss_cls + \
                     loss_cl * self.eta + \
                     loss_li * self.gamma + \
                     w_bal * load_balance_loss + \
                     w_beta * loss_fidelity
        
        # --- Logging ---
        self.print_counter += 1
        if self.print_counter % 50 == 0:
             recent_history = self.fidelity_history[-50:]
             avg_fid = sum(recent_history)/len(recent_history) if recent_history else 0.0
             min_fid = min(recent_history) if recent_history else 0.0
             if len(self.fidelity_history) > 2000: self.fidelity_history = self.fidelity_history[-50:]

             w1_avg = routing_weights[:, 0].mean().item()
             w2_avg = routing_weights[:, 1].mean().item()

             print(f"\n[Batch {self.print_counter}] Top-1 Indices: {top1_indices.tolist()[:10]}...")
             print(f"\033[91m    👉 Weight Split: Top1={w1_avg:.3f} | Top2={w2_avg:.3f} \033[0m", end="")
             print(f"\n\033[92m[Debug] Total: {total_loss.item():.2f}| "
                   f"KL: {(w_bal * load_balance_loss).item():.3f} | "
                   f"Fid(Avg/Min): {avg_fid:.4f}/{min_fid:.4f}\033[0m", end="")

            # =====================================================================
             # 💉 [深度诊断探针 3]：Loss 占比分析与诊断结果输出
             # =====================================================================
             l_cls_val = loss_cls.item()
             l_cl_val = (loss_cl * self.eta).item()
             l_li_val = (loss_li * self.gamma).item()
             
             print(f"\033[95m🔍 [Deep Debug] Feat Shift: {feat_shift:.3f} | Proto Dist: {proto_dist:.3f} | Loss(Cls/CL/Li): {l_cls_val:.3f} / {l_cl_val:.3f} / {l_li_val:.3f}\033[0m")
             
             if l_cl_val > l_cls_val or l_li_val > l_cls_val:
                 print("\033[93m   ⚠️ 异常预警: 辅助Loss(CL/Li)大过主分类Loss(Cls)！梯度可能已被带偏，建议调小 eta 和 gamma。\033[0m")
             if proto_dist < 0.2:
                 print("\033[93m   ⚠️ 异常预警: 原型距离(Proto Dist)极低！不同类别的特征在空间中坍缩在一起了。\033[0m")
             if feat_shift < 0.05:
                 print("\033[93m   ⚠️ 异常预警: 特征偏移极小！Adapter 并没有有效注入新特征，可能是降维太多或学习率太低。\033[0m")
             # =====================================================================
             avg_probs = full_router_probs.mean(dim=0)
             entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-9)).item()
             max_entropy = np.log(pool_size) 
             entropy_ratio = entropy / max_entropy
             dead_experts = (avg_probs < 0.05).sum().item()
             gap = w1_avg - w2_avg

             print(f"\n\033[94m🔬 [Router Health] 路由熵: {entropy:.3f} (占比: {entropy_ratio*100:.1f}%) | "
                   f"死区数: {dead_experts}/{pool_size} | 果断度: {gap:.3f}\033[0m")
             
             if entropy_ratio > 0.95:
                 print("\033[93m   ⚠️ 警告: 路由熵极高！专家正在吃大锅饭，建议更换为正交属性Prompt或降低Temperature。\033[0m")
             elif dead_experts >= pool_size * 0.4:
                 print(f"\033[93m   ⚠️ 警告: {dead_experts} 个专家饿死！建议减少Adapter数量或增加 alpha_balance。\033[0m")
             elif gap < 0.15:
                 print("\033[93m   ⚠️ 警告: 决策果断度过低！Router 极度纠结，专家间职能可能严重重叠。\033[0m")
             else:
                 print("\033[92m   ✅ 状态极佳: 专家分工明确，资源利用率健康。\033[0m")

        return total_loss