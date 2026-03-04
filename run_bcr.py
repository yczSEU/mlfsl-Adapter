import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE  # 必须导入这个做可视化
from utils.utils import set_seed, base_path, get_dataloader, Logger
from methods.bcr import BCR
from utils.backbone import model_dict
import torch.nn.functional as F
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='VG',
                    choices=['COCO', 'CUB', 'NUSWIDE', 'VG'])
parser.add_argument('--algorithm', type=str, default='bcr')
parser.add_argument('--model_name', type=str, default='Conv4')
parser.add_argument('--n_way', type=int, default=10)
parser.add_argument('--n_shot', type=int, default=1)
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--eta', type=float, default=0.5)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=0)

# =========================================================================
# [修复] 可视化分析函数: 适配 CoCoOp 动态路由
# =========================================================================
def visualize_analysis(model, test_loader, device, save_dir):
    print("\n🔍 [Diagnosis] Starting Deep Forensics (Expert & Feature Analysis)...")
    model.eval()
    
    all_raw_feats = []
    all_adapted_feats = []
    all_labels = []
    all_expert_indices = [] 
    
    # 跑 20 个 Batch 以收集足够的数据点
    max_batches = 20
    batch_count = 0
    
    # 1. 检查 Router 是否存在
    if hasattr(model.feature_extractor, 'router'):
        router = model.feature_extractor.router
        print(f"    👉 Router found: {type(router).__name__}")
    else:
        print("    ❌ Error: Could not find 'router' object in feature_extractor.")
        return

    print(f"    Scanning first {max_batches} batches...")
    
    with torch.no_grad():
        for batch in test_loader:
            if batch_count >= max_batches: break
            
            x = batch['image'].to(device)
            y = batch['labels']
            
            # 处理标签格式 (如果是 One-hot 转 Index)
            if y.dim() > 1:
                y_idx = y.argmax(dim=1).cpu().numpy()
            else:
                y_idx = y.cpu().numpy()
            
            # 1. 获取 Raw Feature (用于 Router 输入)
            # 注意：OpenCLIP 输出可能是 float16 或 float32，Router 内部会自动处理
            raw = model.feature_extractor(x, use_pure_clip=True)
            
            # 2. 获取 Adapted Feature (用于画图对比)
            adapted = model.feature_extractor(x, use_pure_clip=False)
            
            # 3. [核心修复] 直接调用 Router API 获取专家选择
            # 不再手动计算 Similarity，因为 CoCoOp 的 Key 是动态生成的
            # 返回: indices [B, K], weights [B, K]
            topk_indices, _, _ = router.get_best_expert_idx(raw, training=False)
            
            # 我们只关心 Top-1 专家是谁
            expert_idx = topk_indices[:, 0].cpu().numpy() # [B]
            
            # 存入列表 (转为 numpy float32 以便后续 TSNE)
            all_raw_feats.append(raw.float().cpu().numpy())
            all_adapted_feats.append(adapted.float().cpu().numpy())
            all_labels.append(y_idx)
            all_expert_indices.append(expert_idx)
            
            batch_count += 1

    # 数据合并
    if len(all_raw_feats) == 0:
        print("    ⚠️ No data collected. Check dataloader.")
        return

    raw_np = np.concatenate(all_raw_feats)
    adap_np = np.concatenate(all_adapted_feats)
    labels_np = np.concatenate(all_labels)
    experts_np = np.concatenate(all_expert_indices)
    
    # --- 统计分析：每个专家都吃了什么？ ---
    print("\n🕵️‍♀️ [Expert Forensics Report]")
    
    # 你的专家名字映射 (按 adapter_pool.py 里的顺序)
    expert_names = ["Animal", "Vehicle", "Furniture", "Food", "Sports"]
    
    unique_experts = np.unique(experts_np)
    for exp_id in unique_experts:
        # 找出选中这个专家的所有样本
        indices = np.where(experts_np == exp_id)[0]
        selected_labels = labels_np[indices]
        count = len(selected_labels)
        
        # 获取专家名字
        exp_name = expert_names[exp_id] if exp_id < len(expert_names) else f"Exp_{exp_id}"
        
        # 计算成分纯度 (这里显示的是 Dataset 中的 Class ID)
        label_counts = Counter(selected_labels)
        most_common = label_counts.most_common(3) 
        
        print(f"  🤖 Expert {exp_id} ({exp_name}) - Chosen {count} times:")
        print(f"     Top Classes (ID:Count): {most_common}")
        
        # 简单判断纯度
        if len(label_counts) > 10:
            print(f"     ⚠️  High Entropy! Contains {len(label_counts)} classes. (Possible Generalist)")
        else:
            print(f"     ✅ Clean Expert. Focused on specific classes.")
    
    # --- 绘图 ---
    print("\n🎨 Computing T-SNE...")
    # 只画 Adapter 后的特征，减少计算量，反正我们要看的是结果
    # 随机采样 2000 个点避免画图太慢
    max_points = 2000
    if len(adap_np) > max_points:
        indices = np.random.choice(len(adap_np), max_points, replace=False)
        adap_vis = adap_np[indices]
        labels_vis = labels_np[indices]
        experts_vis = experts_np[indices]
    else:
        adap_vis = adap_np
        labels_vis = labels_np
        experts_vis = experts_np

    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
    embed_2d = tsne.fit_transform(adap_vis)
    
    plt.figure(figsize=(20, 9))
    
    # Subplot 1: Ground Truth Labels
    plt.subplot(1, 2, 1)
    # 使用 tab20 颜色板以支持更多类别
    sns.scatterplot(x=embed_2d[:,0], y=embed_2d[:,1], hue=labels_vis, palette="tab20", s=60, alpha=0.7, legend=False)
    plt.title(f"Adapted Feature Space\n(Color = Ground Truth Class)")
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Expert ID
    plt.subplot(1, 2, 2)
    # 使用 Set1 区分 5 个专家
    sns.scatterplot(x=embed_2d[:,0], y=embed_2d[:,1], hue=experts_vis, palette="Set1", s=60, alpha=0.8, legend='full')
    plt.title(f"Expert Decision Map\n(Color = Which Expert was Used?)")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, 'analysis_expert_map.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Forensic Map to: {save_path}")
    print("    -> Check the image: If Subplot 2 has clear color clusters that match Subplot 1's semantic clusters,")
    print("       it means the Router has successfully learned semantic separation.")
    
def get_model():
    model = BCR(model_func=model_dict[model_name],
                device=device,
                n_way=n_way,
                n_shot=n_shot,
                n_query=n_query,
                hidden_dim=hidden_dim,
                eta=eta,
                gamma=gamma,
                verbose=True)
    return model


def _train():
    print('Start Training!')
    model = get_model()
    best_mAP = 0
    for epoch in range(max_epoch):
        avg_loss = model.train_loop(train_loader)
        print('epoch %d training done | Loss: %f' % (epoch, avg_loss))
        result = model.test_loop(val_loader)
        mAP = result['mAP']
        print('Epoch %d validation done | MAP: %.4f' % (epoch, mAP))
        if mAP > best_mAP:
            best_mAP = mAP
            model.save(train_dir, epoch='best_mAP')
        if epoch == max_epoch - 1:
            model.save(train_dir, epoch=epoch)
        if (epoch + 1) % 100 == 0:
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5


def _test():
    print('Start Testing!')
    model = get_model()
    model.load(train_dir, epoch='best_mAP')
    # === [核心修改] 在正式测试前，插入可视化分析 ===
    # 将图片保存到 train_dir (或者你可以指定 log_dir)
    visualize_analysis(model, test_loader, device, train_dir)
    # ============================================
    result = model.test_loop(test_loader)
    mAP, mAP_std = result['mAP'], result['mAP-std']
    print(f'{mAP}±{mAP_std}')
    model.print_test_stats()


if __name__ == '__main__':
    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    device = args.device
    seed = args.seed
    n_way = args.n_way
    algorithm = args.algorithm
    max_epoch = args.max_epoch
    n_shot = args.n_shot
    eta = args.eta
    gamma = args.gamma
    num_workers = args.num_workers
    hidden_dim = args.hidden_dim
    
    if 'ViT-H' in model_name:
        image_size = 378
    elif model_name == 'Conv4':
        image_size = 84
    else:
        image_size = 224
    set_seed(seed)
    if dataset_name == 'COCO':
        n_way = np.minimum(n_way, 16)
    else:
        n_way = np.minimum(n_way, 20)
    n_query = n_way // 2

    train_dir = os.path.join(base_path, 'save','text_encoder64_analyse_noKL2', dataset_name,
                             f'{algorithm}_{model_name}_{max_epoch}_{n_way}_{n_shot}_{n_query}_{seed}_{hidden_dim}_{eta}_{gamma}',
                             'train')
    log_dir = os.path.join(base_path, 'save','text_encoder64_analyse_noKL2', dataset_name,
                           f'{algorithm}_{model_name}_{max_epoch}_{n_way}_{n_shot}_{n_query}_{seed}_{hidden_dim}_{eta}_{gamma}',
                           'log')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_loader = get_dataloader(dataset_name=dataset_name, phase='train', n_way=n_way, n_shot=n_shot,
                                  n_query=n_query, transform=True, num_iter=200,
                                  num_workers=num_workers, image_size=image_size)
    val_loader = get_dataloader(dataset_name=dataset_name, phase='val', n_way=n_way, n_shot=n_shot,
                                n_query=n_query, transform=False, num_iter=100, num_workers=num_workers,
                                image_size=image_size)
    test_loader = get_dataloader(dataset_name=dataset_name, phase='test', n_way=n_way, n_shot=n_shot,
                                 n_query=n_query, transform=False, num_iter=1000,
                                 num_workers=num_workers, image_size=image_size)

    print = Logger(f'{log_dir}/log.txt').logger.warning
    print(
        f'{dataset_name}, {algorithm}, model_name: {model_name}, max_epoch: {max_epoch}, n_way: {n_way}, n_shot: {n_shot}, n_query: {n_query}, seed: {seed}, hidden_dim: {hidden_dim}, eta: {eta}, gamma: {gamma}')

    if not os.path.exists(os.path.join(train_dir, f'{max_epoch - 1}.tar')):
        _train()
    _test()

# HF_ENDPOINT=https://hf-mirror.com XDG_CACHE_HOME=/root/autodl-tmp/cache python run_bcr.py \
#   --dataset_name COCO\
#   --model_name ViT-B-Adapter \
#   --n_way 10 \
#   --n_shot 5 \
#   --max_epoch 5 \
#   --hidden_dim 512 \
#   --eta 0.5 \
#   --gamma 0.5 \
#   --device cuda:0
