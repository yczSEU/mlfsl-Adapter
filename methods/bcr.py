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
        
        # [保持原样] 不修改 Hidden Dim，使用传入的维度 (通常是 512)
        self.encoder_x = nn.Linear(self.feat_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(self.feat_dim) 
        self.encoder_y = nn.Linear(self.n_way, hidden_dim)
        self.encoder_z = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 监控容器
        self.fidelity_history = []
        self.print_counter = 0
        self.EPS = 1e-6
        self.test_expert_history = [] 

        # 默认参数 (会被 Search 脚本覆盖)
        self.beta = 4.5
        self.alpha_balance = 2.0
        self.ortho_weight = 0.5
        self.router_weight = 1.0
        
        # =========================================================
        # 优化器配置
        # =========================================================
        backbone_params = []
        adapter_weights = [] 
        adapter_scalars = [] 
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            
            if 'scale' in name or 'router' in name or 'adapter' in name:
                if 'bias' in name or 'scale' in name or 'prompt_key' in name:
                    adapter_scalars.append(param)
                else:
                    adapter_weights.append(param)
            elif 'feature_extractor' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        # 学习率配置：Adapter 5e-4, Head 1e-3
        self.optimizer = torch.optim.Adam([
            {'params': adapter_weights, 'lr': 0.001, 'weight_decay': 1e-3}, 
            {'params': adapter_scalars, 'lr': 0.001,  'weight_decay': 0.0}, 
            {'params': head_params,     'lr': 0.0009} 
        ])
        
        # [恢复] Cosine 调度器 (比 MultiStep 更平滑，防止 Adapter 震荡)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[3, 6], gamma=0.1
        )
        self.to(self.device)
    

    # ... (辅助函数保持不变) ...
    def reset_test_stats(self):
        self.test_expert_history = []
        print("[Stats] Test history reset.")

    def print_test_stats(self):
        if len(self.test_expert_history) == 0:
            print("[Stats] No data recorded.")
            return
        all_indices = torch.cat(self.test_expert_history).cpu().numpy()
        total_decisions = len(all_indices)
        print("\n" + "="*40)
        print(f"📊 [Test Phase Router Statistics]")
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

    def _set_active_adapter(self, idx):
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

    # =========================================================================
    # 测试逻辑
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
            best_indices_support, _ = self.feature_extractor.router.get_best_expert_idx(class_embs, training=False)
            
            if not self.training:
                self.test_expert_history.append(best_indices_support.detach().cpu())

        for class_id in range(n_way):
            exp_idx = best_indices_support[class_id].item()
            self._set_active_adapter(exp_idx)
            imgs = x_support_view[class_id]
            feats = self.feature_extractor(imgs)
            start = class_id * n_shot
            z_support_all[start : start+n_shot] = feats

        z_query_all = torch.zeros(n_query, self.feat_dim, device=x_query.device)
        with torch.no_grad():
             query_feats_clip = self.feature_extractor(x_query, use_pure_clip=True)
             proto_norm = F.normalize(class_embs, p=2, dim=-1)
             query_norm = F.normalize(query_feats_clip, p=2, dim=-1)
             sim = torch.matmul(query_norm, proto_norm.t())
             assigned_class_idx = sim.argmax(dim=1)
             best_idx_query = best_indices_support[assigned_class_idx]
        
        unique_experts_q = torch.unique(best_idx_query)
        for exp_idx in unique_experts_q:
            self._set_active_adapter(exp_idx.item())
            img_mask = (best_idx_query == exp_idx)
            if img_mask.any():
                z_query_all[img_mask] = self.feature_extractor(x_query[img_mask])

        # BCR Core
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
        return scores

    # =========================================================================
    # 训练逻辑 (修复版)
    # =========================================================================
    def set_forward_loss(self, x_support, y_support, x_query, y_query):
        # 1. Router Init
        if hasattr(self.feature_extractor, 'use_adapter') and \
           self.feature_extractor.use_adapter and \
           not self.feature_extractor.router.is_initialized:
            with torch.no_grad():
                real_feats = self.feature_extractor(x_support, use_pure_clip=True)
                real_feats = F.normalize(real_feats, p=2, dim=1)
                n_samples = real_feats.shape[0]
                pool_size = self.feature_extractor.router.pool_size
                indices = torch.randperm(n_samples)[:pool_size] if n_samples >= pool_size else torch.randint(0, n_samples, (pool_size,))
                self.feature_extractor.router.prompt_key.data.copy_(real_feats[indices])
                self.feature_extractor.router.is_initialized = True
                print(f"[Init] Done! Keys initialized from {pool_size} support images.")

        n_way = self.n_way
        n_shot = self.n_shot
        n_query = x_query.shape[0]
        
        # 2. Router Decision
        x_support_view = x_support.view(n_way, n_shot, *x_support.shape[-3:])
        with torch.no_grad():
            flat_support = x_support.reshape(-1, *x_support.shape[-3:])
            raw_feats = self.feature_extractor(flat_support, use_pure_clip=True)
            class_embs = raw_feats.view(n_way, n_shot, -1).mean(dim=1)
        
        best_indices = torch.zeros(n_way, dtype=torch.long, device=self.device)
        router_main_loss = torch.tensor(0.0, device=self.device)
        ortho_loss = torch.tensor(0.0, device=self.device)
        load_balance_loss = torch.tensor(0.0, device=self.device)

        if hasattr(self.feature_extractor, 'use_adapter') and self.feature_extractor.use_adapter:
            _, raw_logits = self.feature_extractor.router.get_best_expert_idx(class_embs, training=True)
            
            if self.training:
                noise = torch.randn_like(raw_logits) * 0.5
                noisy_logits = raw_logits + noise
                best_indices = noisy_logits.argmax(dim=1)
            else:
                best_indices = raw_logits.argmax(dim=1)
            
            selected_scores = F.softmax(raw_logits, dim=1).gather(1, best_indices.unsqueeze(1))
            router_main_loss = 1.0 - selected_scores.mean()

            keys = F.normalize(self.feature_extractor.router.prompt_key, p=2, dim=1)
            sim_matrix = torch.matmul(keys, keys.t())
            eye = torch.eye(sim_matrix.shape[0], device=self.device)
            ortho_loss = torch.mean((sim_matrix - eye) ** 2)
            
            probs = F.softmax(raw_logits, dim=1) 
            avg_usage = probs.mean(dim=0)
            target_usage = torch.full_like(avg_usage, 1.0 / raw_logits.shape[1])
            load_balance_loss = torch.sum(avg_usage * torch.log(avg_usage / (target_usage + 1e-6) + 1e-6))
            
            if not hasattr(self, 'batch_print_count'): self.batch_print_count = 0
            self.batch_print_count += 1
            if self.batch_print_count % 50 == 0:
                 print(f"\n[Batch {self.batch_print_count}] Choices: {best_indices.tolist()}")

        # 3. MoE Extraction
        z_support_all = torch.zeros(n_way * n_shot, self.feat_dim, device=self.device)
        for class_id in range(n_way):
            exp_idx = best_indices[class_id].item()
            self._set_active_adapter(exp_idx)
            imgs = x_support_view[class_id]
            feats = self.feature_extractor(imgs)
            start = class_id * n_shot
            z_support_all[start : start+n_shot] = feats

        z_query_all = torch.zeros(n_query, self.feat_dim, device=self.device)
        query_class_ids = y_query.argmax(dim=1)
        unique_experts = torch.unique(best_indices)
        for exp_idx in unique_experts:
            self._set_active_adapter(exp_idx.item())
            query_mask = (best_indices[query_class_ids] == exp_idx)
            if query_mask.any():
                z_query_all[query_mask] = self.feature_extractor(x_query[query_mask])

        # Fidelity Loss
        sim_scores = F.cosine_similarity(z_support_all.view(-1, self.feat_dim), raw_feats, dim=-1)
        mean_sim = sim_scores.mean()
        self.fidelity_history.append(mean_sim.item())
        
        if self.n_shot == 1:
            fidelity_threshold = getattr(self, 'fidelity_threshold_1shot', 0.96)
        elif self.n_shot >= 5:
            fidelity_threshold = getattr(self, 'fidelity_threshold_5shot', 0.90)

        loss_fidelity = torch.relu(torch.tensor(fidelity_threshold, device=self.device) - mean_sim)

        # 5. BCR Calculation
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
        
        # [修复 2] 恢复温度系数 64.0 (弱化辅助任务)
        sscores = -self.euclidean_dist(x, proto) / 64.0
        loss_li = nn.CrossEntropyLoss()(sscores, torch.softmax(yy, dim=1))
        
        # 6. Total Loss
        w_main   = getattr(self, 'router_weight', 1.0)
        w_bal    = getattr(self, 'alpha_balance', 1.0)
        w_ortho  = getattr(self, 'ortho_weight', 0.0)
        w_beta   = getattr(self, 'beta', 2.0)
        
        
        total_loss = loss_cls + \
                     loss_cl * self.eta + \
                     loss_li * self.gamma + \
                     w_main * router_main_loss + \
                     w_bal * load_balance_loss + \
                     w_ortho * ortho_loss + \
                     w_beta * loss_fidelity

        self.print_counter += 1

        if self.print_counter % 50 == 0:
             recent_history = self.fidelity_history[-50:]
             avg_fid = sum(recent_history)/len(recent_history) if recent_history else 0.0
             min_fid = min(recent_history) if recent_history else 0.0
             if len(self.fidelity_history) > 2000: self.fidelity_history = self.fidelity_history[-50:]

             print(f"\n\033[92m[Debug] Total: {total_loss.item():.2f}| "
                   f"Main: {(w_main * router_main_loss).item():.3f} | "
                   f"KL: {(w_bal * load_balance_loss).item():.3f} | "
                   f"Fid(Avg/Min): {avg_fid:.4f}/{min_fid:.4f}\033[0m", end="")

        return total_loss