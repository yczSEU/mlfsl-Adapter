import torch
import open_clip
from sklearn.cluster import KMeans
import numpy as np

def generate_cluster_prompts(labels, dataset_name="Dataset", num_clusters=5, top_k=5):
    print(f"🚀 开始为 {dataset_name} 生成数据驱动的 5 大 Prompt...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载我们使用的 CLIP 模型
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-16')

    # 2. 提取所有标签的文本特征
    text_tokens = tokenizer(labels).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    features_np = text_features.cpu().numpy()

    # 3. 执行 K-Means 聚类 (分为 5 个专家域)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(features_np)

    # 4. 提取每个聚类最核心的代表词汇
    print("="*60)
    for i in range(num_clusters):
        cluster_indices = np.where(cluster_ids == i)[0]
        center = kmeans.cluster_centers_[i]
        
        # 计算该簇内所有词到中心的距离
        distances = np.linalg.norm(features_np[cluster_indices] - center, axis=1)
        sorted_sub_indices = np.argsort(distances) # 按距离从小到大排序
        
        # 提取前 K 个最靠近中心的词
        top_words = [labels[cluster_indices[idx]] for idx in sorted_sub_indices[:top_k]]
        prompt = ", ".join(top_words)
        
        print(f"🤖 [专家 {i}] 负责的子领域包含: {[labels[idx] for idx in cluster_indices][:10]} 等 {len(cluster_indices)} 个类别...")
        print(f"🎯 建议提取的 Prompt: \033[92m\"{prompt}\"\033[0m\n")
    print("="*60)

if __name__ == "__main__":
    # 你可以直接替换这里的列表来生成 COCO 或 VG 的 Prompt
    coco_labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
    
    generate_cluster_prompts(coco_labels, dataset_name="COCO")