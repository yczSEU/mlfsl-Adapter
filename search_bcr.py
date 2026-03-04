import optuna
import torch
import os
import argparse
import numpy as np
import csv

from utils.utils import set_seed, base_path, get_dataloader
from methods.bcr import BCR
from utils.backbone import model_dict
from utils.adapter_pool import APART_Adapter

DEFAULT_CONFIG = {
    'model_name': 'ViT-B-Adapter',
    'n_way': 10,
    'max_epoch': 8,  # 8 轮足够分辨出参数的好坏
    'hidden_dim': 512,
    'device': 'cuda:0',
    'num_workers': 4
}

def objective(trial, args):
    # =================================================================
    # 1. 定义精简搜索空间 (只搜这 3 个核心路由参数)
    # =================================================================
    
    suggest_alpha = trial.suggest_float("alpha_balance", 0.01, 0.50)
    suggest_temp = trial.suggest_float("temperature", 0.02, 0.10)
    suggest_max_scale = trial.suggest_float("max_scale", 0.10, 0.25)

    # 固定其余安全阈值
    fixed_lr_router = 2e-3
    fixed_lr_adapter = 5e-4
    if args.n_shot == 1:
        fixed_fid_thresh = 0.90
    else:
        fixed_fid_thresh = 0.85

    # =================================================================
    # 2. 初始化模型
    # =================================================================
    device = DEFAULT_CONFIG['device']
    
    model = BCR(model_func=model_dict[DEFAULT_CONFIG['model_name']],
                n_way=DEFAULT_CONFIG['n_way'],
                n_shot=args.n_shot,
                n_query=DEFAULT_CONFIG['n_way'] // 2,
                hidden_dim=DEFAULT_CONFIG['hidden_dim'],
                device=device,
                verbose=False)

    # =================================================================
    # 3. 动态注入 Optuna 参数
    # =================================================================
    
    # 注入 BCR 层的 Alpha 和固定的 Fid 阈值
    model.alpha_balance = suggest_alpha
    if args.n_shot == 1:
        model.fidelity_threshold_1shot = fixed_fid_thresh
    else:
        model.fidelity_threshold_5shot = fixed_fid_thresh

    # 注入 Router 温度
    if hasattr(model.feature_extractor, 'router'):
        model.feature_extractor.router.temperature = suggest_temp

    # 注入 Adapter 缩放上限
    for module in model.modules():
        if isinstance(module, APART_Adapter):
            module.max_scale = suggest_max_scale

    # 注入固定的学习率
    for group in model.optimizer.param_groups:
        is_router = any(['router' in name for name, p in model.named_parameters() if p in group['params']])
        if is_router:
            group['lr'] = fixed_lr_router
        else:
            group['lr'] = fixed_lr_adapter

    # =================================================================
    # 4. 数据加载与日志
    # =================================================================
    log_filename = f"search_log_{args.dataset_name}_{args.n_shot}shot_core.csv"
    if not os.path.exists(log_filename):
        with open(log_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Trial_ID', 'mAP', 'Alpha', 'Temp', 'Max_Scale'])

    train_loader = get_dataloader(args.dataset_name, 'train', 
                                  DEFAULT_CONFIG['n_way'], args.n_shot, 
                                  5, True, 200, DEFAULT_CONFIG['num_workers'], 224)
    val_loader = get_dataloader(args.dataset_name, 'val', 
                                DEFAULT_CONFIG['n_way'], args.n_shot, 
                                5, False, 100, DEFAULT_CONFIG['num_workers'], 224)

    best_mAP = 0.0
    
    print(f"\n🚀 Trial {trial.number} | Alpha={suggest_alpha:.3f} | Temp={suggest_temp:.3f} | Scale={suggest_max_scale:.2f}")

    # =================================================================
    # 5. 训练循环
    # =================================================================
    for epoch in range(DEFAULT_CONFIG['max_epoch']):
        try:
            model.train_loop(train_loader)
            result = model.test_loop(val_loader)
            mAP = result['mAP']
        except Exception as e:
            print(f"Error in trial {trial.number}: {e}")
            return 0.0 

        if mAP > best_mAP:
            best_mAP = mAP
        
        trial.report(mAP, epoch)
        if trial.should_prune():
            print(f"✂️ Trial {trial.number} pruned at epoch {epoch}.")
            raise optuna.TrialPruned()

    # 写入结果
    with open(log_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            trial.number, f"{best_mAP:.4f}", f"{suggest_alpha:.4f}", 
            f"{suggest_temp:.4f}", f"{suggest_max_scale:.3f}"
        ])
        print(f"💾 [Saved] Trial {trial.number} finished (mAP: {best_mAP:.2f})")

    return best_mAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='COCO', help='Name of the dataset')
    parser.add_argument('--n_shot', type=int, default=5, choices=[1, 5], help='Number of shots')
    parser.add_argument('--n_trials', type=int, default=40, help='Number of trials')
    args = parser.parse_args()

    db_name = f"bcr_search_{args.dataset_name}_{args.n_shot}shot_core"
    storage_name = f"sqlite:///{db_name}.db"
    
    print(f"🚀 Starting Core Search: Dataset={args.dataset_name}, Shot={args.n_shot}")
    
    study = optuna.create_study(
        study_name=db_name,
        direction="maximize",
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)