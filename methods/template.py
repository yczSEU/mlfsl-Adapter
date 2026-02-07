import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
import os
from collections import deque
from utils.metrics import evaluation
import torch.nn.functional as F

# ========================================================
# 1. 补回丢失的函数 (放在最前面)
# ========================================================
def batch_augment(images):
    import random
    with torch.no_grad():
        B, C, H, W = images.shape
        device = images.device
        augmented = images.clone()
        flip_mask = torch.rand(B, device=device) > 0.5
        augmented[flip_mask] = torch.flip(augmented[flip_mask], dims=[3])
        pad = 28 
        padded = F.pad(augmented, (pad, pad, pad, pad), mode='reflect')
        h_start = random.randint(0, 2*pad)
        w_start = random.randint(0, 2*pad)
        augmented = padded[:, :, h_start:h_start+H, w_start:w_start+W]
        return augmented

class MLLTemplate(nn.Module):
    def __init__(self, model_func, n_way=16, n_shot=5, n_query=8,
                 gradient_clip_value=5.0, device='cuda:0', verbose=False):
        super(MLLTemplate, self).__init__()
        self.feature_extractor = model_func()
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.feat_dim = self.feature_extractor.final_feat_dim
        self.gradient_clip_value, self.gradient_norm_queue = gradient_clip_value, deque([3.0], maxlen=5)
        self.device = device
        self.verbose = verbose

    @abstractmethod
    def set_forward_loss(self, x_support, y_support, x_query, y_query):
        pass

    @abstractmethod
    def set_forward(self, x_support, y_support, x_query):
        pass

    def train_loop(self, train_loader):
        self.train()
        
        # === [修正] 基于 Batch=65 的正确算术 ===
        # Support 是结构化的: 10 way * 5 shot = 50
        num_support = self.n_way * self.n_shot
        
        # Query 是稀疏的: 直接等于 n_query (5)
        # 不再乘以 n_way
        num_query = self.n_query 
        
        epoch_loss_stats = {} 
        num_batches = 0
        
        # Router 状态管理
        if hasattr(self.feature_extractor, 'router'):
            if not hasattr(self, 'epoch_counter'):
                self.epoch_counter = 0
            self.feature_extractor.router.current_epoch = self.epoch_counter
            if hasattr(self.feature_extractor.router, 'reset_stats'):
                self.feature_extractor.router.reset_stats()
            self.epoch_counter += 1

        for batch in train_loader:
            x = batch['image'].to(self.device)
            y = batch['labels'].float()

            # ----------------------------------------------------
            # 安全切片 (适配 Batch=65)
            # ----------------------------------------------------
            # x_support: [50, C, H, W]
            x_support = x[:num_support]
            y_support = y[:num_support]
            
            # x_query: [5, C, H, W]
            x_query = x[num_support : num_support + num_query]
            y_query = y[num_support : num_support + num_query]
            
            # 剩下的 (10) 是 Class Prototypes，用于采样 Sampled Idx
            y_class = y[num_support + num_query:]
            
            # 筛选活跃类
            sampled_idx = y_class.sum(0).bool()
            y_support = y_support[:, sampled_idx].to(self.device)
            y_query = y_query[:, sampled_idx].to(self.device)
            
            if self.n_shot > 1:
                x_support_aug = batch_augment(x_support)
            else:
                x_support_aug = x_support
            
            self.optimizer.zero_grad()
            
            # 调用 BCR
            output = self.set_forward_loss(x_support_aug, y_support, x_query, y_query)
            
            if isinstance(output, tuple):
                loss, log_dict = output
            else:
                loss = output
                log_dict = {"Total": loss.item()}
            
            loss.backward()
            self.clip_gradient()
            self.optimizer.step()
            
            for k, v in log_dict.items():
                if k not in epoch_loss_stats:
                    epoch_loss_stats[k] = 0.0
                epoch_loss_stats[k] += v
            num_batches += 1
            
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"[LR Info] Current LR decayed to: {current_lr:.6f}")

        # === Monitor (修改了这里的打印逻辑) ===
        print("\n[Adapter Pool Monitor]")
        try:
            backbone = self.feature_extractor
            if hasattr(backbone.model, 'visual'):
                first_wrapper = backbone.model.visual.transformer.resblocks[0].mlp
            else:
                first_wrapper = backbone.model.blocks[0].mlp
            router = backbone.router
            
            usage_counts = []
            if hasattr(router, 'expert_usage_counts'):
                usage_counts = router.expert_usage_counts.cpu().tolist()
            total_calls = sum(usage_counts) + 1e-6
            
            active_id = first_wrapper.active_idx
            
            if router.prompt_key.grad is not None:
                r_grad = router.prompt_key.grad.abs().mean().item()
                print(f"    Router Key Grad: {r_grad:.8f} (Learning!)")
            else:
                print("    Router Key Grad: None")

            keys = F.normalize(router.prompt_key.data, p=2, dim=1)
            sim_matrix = torch.matmul(keys, keys.t())
            avg_inter_sim = (sim_matrix.sum() - keys.shape[0]) / (keys.shape[0] * (keys.shape[0]-1))
            print(f"    Router Diversity: Avg Key Sim = {avg_inter_sim:.4f}")

            print(f"  > [Experts Stats in Layer 0] (Total: {len(first_wrapper.adapters)})")
            print(f"    {'ID':<3} | {'Usage (Epoch)':<15} | {'State':<8} | {'Scale':<10} | {'Weight Norm':<12} | {'Grad (Last Batch)'}")
            print("    " + "-" * 85)
            
            for i, adapter in enumerate(first_wrapper.adapters):
                state = "ACTIVE" if i == active_id else "Sleep"
                if usage_counts:
                    count = usage_counts[i]
                    percent = (count / total_calls) * 100.0
                    usage_str = f"{count:<5} ({percent:4.1f}%)"
                else:
                    usage_str = "N/A"

                # [修正开始] =============================================
                # 使用正确的公式：Scale = Min + (Max - Min) * Sigmoid
                logits = adapter.scale_logits.data
                probs = torch.sigmoid(logits)
                
                # 动态获取当前 Adapter 的配置，防止日志公式和实际训练不一致
                min_s = getattr(adapter, 'min_scale_val', 0.0) 
                max_s = getattr(adapter, 'max_scale_val', 0.20)
                
                real_scale = min_s + (max_s - min_s) * probs
                # [修正结束] =============================================

                w_norm = adapter.up_proj.weight.data.abs().mean().item()
                
                if adapter.up_proj.weight.grad is not None:
                    grad_norm = adapter.up_proj.weight.grad.abs().mean().item()
                    has_grad = f"YES ({grad_norm:.1e})"
                else:
                    has_grad = "no"
                
                prefix = ">> " if i == active_id else "   "
                print(f"  {prefix}{i:<3} | {usage_str:<15} | {state:<8} | {real_scale.item():.6f}   | {w_norm:.6f}     | {has_grad}")
                
        except AttributeError as e:
            print(f"  Warning: Monitor failed. Error: {e}")
        print("-" * 50)

        # Loss 打印
        print("  > [Loss Breakdown (Avg)]")
        loss_str = "    "
        for k, v in epoch_loss_stats.items():
            if num_batches > 0:
                avg_val = v / num_batches
                loss_str += f"{k}: {avg_val:.4f} | "
        print(loss_str)
        print("-" * 50)

        if num_batches > 0:
            return epoch_loss_stats.get("Total", 0.0) / num_batches
        else:
            return 0.0

    def test_loop(self, test_loader):
        # 保持你原来的 test_loop 代码不变
        # ... (略) ...
        # 为节省篇幅，这里假设你保留了原有的 test_loop 和其他辅助函数
        self.eval()
        num_support = self.n_way * self.n_shot
        iter_num = len(test_loader)
        results = {}
        results['mAP'] = []
        for batch in test_loader:
            x = batch['image'].to(self.device)
            y = batch['labels'].float()
            x_support = x[:num_support]
            y_support = y[:num_support]
            x_query = x[num_support:num_support + self.n_query] # test loop 中 n_query 也要注意
            y_query = y[num_support:num_support + self.n_query] # 但 test data 通常比较标准
            
            # 简单的 test loop 逻辑
            y_class = y[num_support + self.n_query:]
            sampled_idx = y_class.sum(0).bool()
            y_support = y_support[:, sampled_idx].to(self.device)
            y_query = y_query[:, sampled_idx]
            if y_query.sum() == 0: continue
            with torch.no_grad():
                y_pred = self.set_forward(x_support, y_support, x_query)
            if type(y_pred) == torch.tensor:
                y_pred = y_pred.detach().cpu().numpy()
            y_test = y_query.numpy()
            result = evaluation(y_test, y_pred)
            results['mAP'].append(result['mAP'])
        
        results['mAP-std'] = 1.96 * np.std(results['mAP']) / np.sqrt(iter_num) * 100
        results['mAP'] = np.mean(results['mAP']) * 100
        return results

    # 保持 save, load, clip_gradient, cosine_similarity, mahalanobis_dist, euclidean_dist 不变
    def save(self, path, epoch=None, save_optimizer=False):
        os.makedirs(path, exist_ok=True)
        if type(epoch) is str: save_path = os.path.join(path, '%s.tar' % epoch)
        elif epoch is None: save_path = os.path.join(path, 'model.tar')
        else: save_path = os.path.join(path, '%d.tar' % epoch)
        while True:
            try:
                if not save_optimizer: torch.save({'model': self.state_dict(), }, save_path)
                else: torch.save({'model': self.state_dict(), 'optimizer': self.optimizer.state_dict(), }, save_path)
                return
            except: pass

    def load(self, path, epoch=None, load_optimizer=False):
        if type(epoch) is str: load_path = os.path.join(path, '%s.tar' % epoch)
        else:
            if epoch is None:
                files = os.listdir(path)
                files = np.array(list(map(lambda x: int(x.replace('.tar', '')), files)))
                epoch = np.max(files)
            load_path = os.path.join(path, '%d.tar' % epoch)
        tmp = torch.load(load_path)
        self.load_state_dict(tmp['model'])
        if load_optimizer: self.optimizer.load_state_dict(tmp['optimizer'])

    def clip_gradient(self):
        if self.gradient_clip_value is not None:
            max_norm = max(self.gradient_norm_queue)
            if max_norm > 10.0: max_norm = 10.0
            total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm * self.gradient_clip_value)
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                print("Warning: NaN/Inf gradient detected! Skipping update.")
                self.optimizer.zero_grad()
                return
            self.gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))

    def cosine_similarity(self, x, y):
        assert x.size(1) == y.size(1)
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
        return x @ y.transpose(0, 1)

    def mahalanobis_dist(self, x, y):
        assert x.size(1) == y.size(1)
        cov = torch.cov(x)
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))
        y = y.unsqueeze(0).expand(x.shape)
        delta = x - y
        return torch.einsum('abc,abc->ab', torch.einsum('abc,ad->abc', delta, torch.inverse(cov)), delta)

    def euclidean_dist(self, x, y):
        assert x.size(1) == y.size(1)
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))
        y = y.unsqueeze(0).expand(x.shape)
        return torch.pow(x - y, 2).sum(2)