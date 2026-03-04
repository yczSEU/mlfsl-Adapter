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
        
        num_support = self.n_way * self.n_shot
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

            x_support = x[:num_support]
            y_support = y[:num_support]
            x_query = x[num_support : num_support + num_query]
            y_query = y[num_support : num_support + num_query]
            y_class = y[num_support + num_query:]
            
            sampled_idx = y_class.sum(0).bool()
            y_support = y_support[:, sampled_idx].to(self.device)
            y_query = y_query[:, sampled_idx].to(self.device)
            
            if self.n_shot > 1:
                x_support_aug = batch_augment(x_support)
            else:
                x_support_aug = x_support
            
            self.optimizer.zero_grad()
            
            output = self.set_forward_loss(x_support_aug, y_support, x_query, y_query)
            
            if isinstance(output, tuple):
                loss, log_dict = output
            else:
                loss = output
                log_dict = {"Total": loss.item()}
            
            loss.backward()
            # 👇 --- [开始粘贴：案发现场实时抓拍] --- 👇
            if num_batches % 50 == 0:
                print(f"\n🕵️ [Batch {num_batches} 实时抓拍] Backward 刚结束:")
                try:
                    r_ctx = self.feature_extractor.router.ctx
                    r_meta = self.feature_extractor.router.meta_net[0].weight
                    print(f"  ⚡ 真实 Ctx 梯度: {r_ctx.grad.abs().max().item() if r_ctx.grad is not None else 'None'}")
                    print(f"  ⚡ 真实 MetaNet 梯度: {r_meta.grad.abs().max().item() if r_meta.grad is not None else 'None'}")
                except Exception as e:
                    print(f"  ❌ 抓拍失败: {e}")
            # 👆 --- [粘贴结束] --- 👆
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

        # === [Monitor Update: CoCoOp & Top-1 Ready] ===
        print("\n[Adapter Pool Monitor (Top-1 & CoCoOp)]")
        try:
            backbone = self.feature_extractor
            # 1. 自动定位第一个 Wrapper (兼容不同架构)
            if hasattr(backbone.model, 'visual'):
                first_wrapper = backbone.model.visual.transformer.resblocks[0].mlp
            else:
                first_wrapper = backbone.model.blocks[0].mlp
            
            router = backbone.router
            
            # --- Part A: Router 监控 (区分 CoCoOp 和 Static) ---
            print("  > [Router Status]")
            
            # 情况 1: CoCoOp Router (动态)
            if hasattr(router, 'ctx'):
                # 检查 Context Vector 梯度 (✅ 改用 max 提取最大绝对梯度，防止被 mean 抹零)
                if router.ctx.grad is not None:
                    ctx_grad = router.ctx.grad.abs().max().item()
                    print(f"    Ctx Vector Grad: {ctx_grad:.4e} (✅ Learning)")
                else:
                    print(f"    Ctx Vector Grad: None (⚠️ Frozen?)")
                
                # 检查 MetaNet 梯度 (看第一层) (✅ 改用 max)
                if hasattr(router, 'meta_net') and len(router.meta_net) > 0:
                    first_layer = router.meta_net[0]
                    if hasattr(first_layer, 'weight') and first_layer.weight.grad is not None:
                        meta_grad = first_layer.weight.grad.abs().max().item()
                        print(f"    MetaNet Grad   : {meta_grad:.4e} (✅ Learning)")
                    else:
                        print(f"    MetaNet Grad   : None")
                        
            # 情况 2: Static Router (旧版)
            elif hasattr(router, 'prompt_key'):
                if router.prompt_key.grad is not None:
                    r_grad = router.prompt_key.grad.abs().max().item()
                    print(f"    Router Key Grad: {r_grad:.8f}")
                
                # 计算 Key 多样性
                keys = F.normalize(router.prompt_key.data, p=2, dim=1)
                sim_matrix = torch.matmul(keys, keys.t())
                avg_inter_sim = (sim_matrix.sum() - keys.shape[0]) / (keys.shape[0] * (keys.shape[0]-1))
                print(f"    Key Diversity  : Avg Sim = {avg_inter_sim:.4f}")

            # --- Part B: Expert 统计 ---
            usage_counts = []
            if hasattr(router, 'expert_usage_counts'):
                usage_counts = router.expert_usage_counts.cpu().tolist()
            total_calls = sum(usage_counts) + 1e-6

            print(f"  > [Experts Stats (Layer 0)]")
            print(f"    {'ID':<3} | {'Usage (Top-1)':<15} | {'State':<8} | {'Scale (Mean) [Grad]':<25} | {'Weight Grad'}")
            print("    " + "-" * 90)
            
            for i, adapter in enumerate(first_wrapper.adapters):
                # 1. 检查是否有梯度 (✅ 这里也改用了 max)
                has_grad_bool = False
                grad_norm_str = "None"
                if adapter.up_proj.weight.grad is not None:
                    has_grad_bool = True
                    g_val = adapter.up_proj.weight.grad.abs().max().item()
                    grad_norm_str = f"{g_val:.1e}"
                
                state = "ACTIVE" if has_grad_bool else "---"
                prefix = ">> " if has_grad_bool else "   "
                
                # 2. 使用率
                if usage_counts:
                    count = usage_counts[i]
                    percent = (count / total_calls) * 100.0
                    usage_str = f"{int(count):<5} ({percent:4.1f}%)"
                else:
                    usage_str = "N/A"

                # 3. Scale 状态
                scale_str = "N/A"
                if hasattr(adapter, 'channel_scale_logits'):
                    s_param = adapter.channel_scale_logits
                    
                    # 梯度
                    s_grad = "No"
                    if s_param.grad is not None:
                        s_grad = f"{s_param.grad.abs().max().item():.1e}"
                    
                    # 数值 (Mean)
                    with torch.no_grad():
                        probs = torch.sigmoid(s_param)
                        min_s, max_s = 0.01, 0.10 # 这里要和你定义的对应
                        scales = min_s + (max_s - min_s) * probs
                        s_mean = scales.mean().item()
                    
                    scale_str = f"{s_mean:.4f} [G:{s_grad}]"

                print(f"  {prefix}{i:<3} | {usage_str:<15} | {state:<8} | {scale_str:<25} | {grad_norm_str}")
                
        except Exception as e:
            print(f"  Warning: Monitor error: {e}")
        print("-" * 110)

        # Loss 打印 (保持不变)
        print("  > [Loss Breakdown (Avg)]")
        loss_str = "    "
        for k, v in epoch_loss_stats.items():
            if num_batches > 0:
                avg_val = v / num_batches
                loss_str += f"{k}: {avg_val:.4f} | "
        print(loss_str)
        print("-" * 110)

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