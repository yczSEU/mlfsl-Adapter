import os
import torch
import numpy as np
import argparse

# 假设你的项目结构是标准的，可以直接导入这些模块
# 如果报错，请确保你在项目根目录下运行
from utils.utils import set_seed, get_dataloader, base_path
from utils.backbone import model_dict
from methods.bcr import BCR

# ================= 配置区域 =================
# 权重文件的绝对路径 (直接复制你提供的路径)
CHECKPOINT_PATH = "save/VG/bcr_ViT-H-CLIP_200_10_5_5_0_512_0.5_0.5/train/best_mAP.tar"

# 必须与训练时的参数保持完全一致
CONFIG = {
    'dataset_name': 'VG',
    'model_name': 'ViT-H-CLIP',
    'n_way': 10,
    'n_shot': 5,
    'n_query': 5,      # 根据你之前的log，query应该是5
    'hidden_dim': 512,
    'eta': 0.5,
    'gamma': 0.5,
    'seed': 0,
    'device': 'cuda:0',
    'num_workers': 8
}
# ===========================================

def run_test():
    print(f"🔄 正在初始化环境，使用设备: {CONFIG['device']}...")
    
    # 1. 设置随机种子
    set_seed(CONFIG['seed'])
    
    # 2. 确定图像尺寸 (与 run_bcr.py 逻辑一致)
    if 'ViT-H' in CONFIG['model_name']:
        image_size = 378
    elif CONFIG['model_name'] == 'Conv4':
        image_size = 84
    else:
        image_size = 224
    
    print(f"📏 图像尺寸设置为: {image_size}")

    # 3. 准备数据加载器
    print("📦 正在加载测试数据集...")
    # 注意：VG 数据集通常 n_way 在测试时可能不同，这里保持与训练一致的 10-way
    test_loader = get_dataloader(
        dataset_name=CONFIG['dataset_name'], 
        phase='test', 
        n_way=CONFIG['n_way'], 
        n_shot=CONFIG['n_shot'],
        n_query=CONFIG['n_query'], 
        transform=False, 
        num_iter=1000, # 测试通常跑 1000 个 episode
        num_workers=CONFIG['num_workers'], 
        image_size=image_size
    )

    # 4. 初始化模型
    print(f"🧠 正在构建模型 {CONFIG['model_name']}...")
    model = BCR(
        model_func=model_dict[CONFIG['model_name']],
        device=CONFIG['device'],
        n_way=CONFIG['n_way'],
        n_shot=CONFIG['n_shot'],
        n_query=CONFIG['n_query'],
        hidden_dim=CONFIG['hidden_dim'],
        eta=CONFIG['eta'],
        gamma=CONFIG['gamma'],
        verbose=True
    )

    # 5. 加载权重 (关键步骤)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"❌ 找不到权重文件: {CHECKPOINT_PATH}")

    # 解析路径：分离出 文件夹路径 和 文件名(不带后缀)
    # 例如: .../train/best_mAP.tar -> dir: .../train, epoch: best_mAP
    model_dir = os.path.dirname(CHECKPOINT_PATH)
    file_name = os.path.basename(CHECKPOINT_PATH)
    epoch_tag = file_name.replace('.tar', '')

    print(f"📥 正在加载权重: {file_name} 从 {model_dir}")
    # 调用 MLLTemplate 中的 load 方法
    model.load(model_dir, epoch=epoch_tag)
    
    # 6. 执行测试
    print("🚀 开始测试...")
    result = model.test_loop(test_loader)
    
    # 7. 打印结果
    mAP, mAP_std = result['mAP'], result['mAP-std']
    print("\n" + "="*30)
    print(f"🏆 最终测试结果 ({CONFIG['dataset_name']} {CONFIG['n_way']}-way {CONFIG['n_shot']}-shot)")
    print(f"📊 mAP: {mAP:.2f}% ± {mAP_std:.2f}%")
    print("="*30 + "\n")

if __name__ == '__main__':
    run_test()