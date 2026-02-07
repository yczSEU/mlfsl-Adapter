import optuna
import torch
import os
import argparse
import numpy as np
import sys
import csv

from utils.utils import set_seed, base_path, get_dataloader
from methods.bcr import BCR
from utils.backbone import model_dict

# ----------------------------------------------------------------------
# 基础配置 (会被命令行参数覆盖)
# ----------------------------------------------------------------------
DEFAULT_CONFIG = {
    'model_name': 'ViT-B-Adapter',
    'n_way': 10,
    'max_epoch': 15,  # 搜索时跑 15 轮足够看趋势
    'hidden_dim': 512,
    'device': 'cuda:0',
    # 'seed': 0, # [修改] 去掉默认 seed，让并行进程随机
    'num_workers': 4
}

def objective(trial, args):
    # =================================================================
    # 1. 定义精简后的搜索空间 (Smart Search Space)
    # =================================================================
    
    # [A] 关键变量 (重点搜)
    # Scale: 根据分析，0.12~0.16 最好，但也保留 0.02 的可能性
    suggest_min_scale = trial.suggest_float("min_scale", 0.01, 0.10, step=0.01)
    
    # Alpha: 0.5 和 2.0 都有可能
    suggest_alpha = trial.suggest_categorical("alpha_balance", [0.5, 1.0, 1.5, 2.0])
    
    # LR: 集中在 5e-4 ~ 1.2e-3
    suggest_lr_adapter = trial.suggest_float("lr_adapter", 5e-4, 1.5e-3, log=True)
    
    # (可选) Head LR: 如果你还想稍微动一下，可以搜，或者直接跟 Adapter 保持一致
    # 这里建议还是给一点自由度，但范围很小
    suggest_lr_head = trial.suggest_float("lr_head", 5e-4, 1.5e-3, log=True)

    # [B] 固定变量 (直接写死，节省算力)
    fixed_beta = 4.5         # 根据 Top 20 分析得出
    fixed_ortho = 0.5        # 不敏感，取中间值
    fixed_router_w = 1.0     # 默认值
    fixed_aux = 0.8          # 根据 Top 20 分析得出
    
    if args.n_shot == 1:
        fixed_fid_thresh = 0.96
    else:
        fixed_fid_thresh = 0.89 # 根据 Top 20 分析得出

    # =================================================================
    # 2. 初始化环境和数据
    # =================================================================
    device = DEFAULT_CONFIG['device']
    # set_seed(0) # [修改] 不要在 objective 里设固定的 seed，否则并行会撞车
    
    # 建立模型
    model = BCR(model_func=model_dict[DEFAULT_CONFIG['model_name']],
                n_way=DEFAULT_CONFIG['n_way'],
                n_shot=args.n_shot,
                n_query=DEFAULT_CONFIG['n_way'] // 2,
                hidden_dim=DEFAULT_CONFIG['hidden_dim'],
                eta=fixed_aux,     # [固定]
                gamma=fixed_aux,   # [固定]
                device=device,
                verbose=False)

    # =================================================================
    # 3. 动态注入参数
    # =================================================================
    
    # A. Scale
    for module in model.modules():
        if hasattr(module, 'min_scale_val'):
            module.min_scale_val = suggest_min_scale
            module.max_scale_val = 0.20 # 固定上限

    # B. Loss 权重
    model.router_weight = fixed_router_w
    model.beta = fixed_beta          # [固定]
    model.alpha_balance = suggest_alpha # [搜索]
    model.ortho_weight = fixed_ortho # [固定]
    
    # C. Threshold
    if args.n_shot == 1:
        model.fidelity_threshold_1shot = fixed_fid_thresh
    else:
        model.fidelity_threshold_5shot = fixed_fid_thresh

    # D. LR 注入
    # 1. Head (倒数第1组)
    if len(model.optimizer.param_groups) >= 1:
        model.optimizer.param_groups[-1]['lr'] = suggest_lr_head
        
    # 2. Adapter Scale (倒数第2组)
    if len(model.optimizer.param_groups) >= 2:
        model.optimizer.param_groups[-2]['lr'] = suggest_lr_adapter
        
    # 3. Adapter Weights (倒数第3组)
    if len(model.optimizer.param_groups) >= 3:
        model.optimizer.param_groups[-3]['lr'] = suggest_lr_adapter

    # =================================================================
    # 4. CSV Logging (精简版)
    # =================================================================
    log_filename = f"search_log_{args.dataset_name}_{args.n_shot}shot_smart.csv"
    
    if not os.path.exists(log_filename):
        with open(log_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            # 只记录变化的参数，固定的就不记了
            header = ['Trial_ID', 'mAP', 'Min_Scale', 'Alpha', 'LR_Adap', 'LR_Head']
            writer.writerow(header)

    # =================================================================
    # 5. 训练循环
    # =================================================================
    train_loader = get_dataloader(args.dataset_name, 'train', 
                                  DEFAULT_CONFIG['n_way'], args.n_shot, 
                                  5, True, 200, DEFAULT_CONFIG['num_workers'], 224)
    val_loader = get_dataloader(args.dataset_name, 'val', 
                                DEFAULT_CONFIG['n_way'], args.n_shot, 
                                5, False, 100, DEFAULT_CONFIG['num_workers'], 224)

    best_mAP = 0.0
    
    print(f"\n🚀 Trial {trial.number}: Scale={suggest_min_scale:.2f}, Alpha={suggest_alpha}, "
          f"LR_Adp={suggest_lr_adapter:.1e}, LR_Head={suggest_lr_head:.1e}")

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
            raise optuna.TrialPruned()

    # 写入结果
    with open(log_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        row = [
            trial.number,
            f"{best_mAP:.4f}",
            f"{suggest_min_scale:.4f}",
            f"{suggest_alpha:.1f}",
            f"{suggest_lr_adapter:.6f}",
            f"{suggest_lr_head:.6f}"
        ]
        writer.writerow(row)
        print(f"💾 [Saved] Trial {trial.number} (mAP: {best_mAP:.2f})")

    return best_mAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='COCO', help='Name of the dataset')
    parser.add_argument('--n_shot', type=int, default=5, choices=[1, 5], help='Number of shots')
    parser.add_argument('--n_trials', type=int, default=30, help='Number of trials')
    args = parser.parse_args()

    # 使用新的数据库名，避免混淆
    db_name = f"bcr_search_{args.dataset_name}_{args.n_shot}shot_smart"
    storage_name = f"sqlite:///{db_name}.db"
    
    print(f"🚀 Starting Smart Search: Dataset={args.dataset_name}, Shot={args.n_shot}")
    
    study = optuna.create_study(
        study_name=db_name,
        direction="maximize",
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler()
    )
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)