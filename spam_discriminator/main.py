import torch
import numpy as np
import sys
import os

# 导入垃圾文本判别器类
from spam_discriminator import SpamDiscriminator  
from utils import *  
from ContrastiveSpamDiscriminator import ContrastiveSpamDiscriminator

# 读取数据集
def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text_data = f.readlines()

    dataset = [s.strip().split('\t', 1) for s in text_data]
    dataset = [data for data in dataset if len(data) == 2 and data[1].strip()]

    tag = [data[0] for data in dataset]
    text = [data[1] for data in dataset]

    return tag, text


def main():
    # 设置数据路径
    dataset_path = './数据集/dataset.txt'
    hanzi_path = './数据集/hanzi.txt'
    
    # 检查文件路径是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集文件 {dataset_path} 不存在!")
        return
    
    if not os.path.exists(hanzi_path):
        print(f"错误: 汉字文件 {hanzi_path} 不存在!")
        return

    # 读取数据
    print("正在读取数据...")
    tags, texts = read_data(dataset_path)
    
    if not tags or not texts:
        print("错误: 没有读取到数据!")
        return
    
    print(f"读取了 {len(texts)} 条数据")
    
    # 统计汉字并加载汉字数据
    print("正在处理汉字数据")
    cache_dir = "./数据集/cache"
    os.makedirs(cache_dir, exist_ok=True)  # 创建缓存目录
    
    # 定义缓存文件路径
    hanzi_stats_cache = os.path.join(cache_dir, "hanzi_stats.pkl")
    sim_mat_cache = os.path.join(cache_dir, "sim_mat.npy")    
    # 尝试加载缓存的汉字统计
    if os.path.exists(hanzi_stats_cache):
        print("检测到汉字统计缓存，正在加载...")
        with open(hanzi_stats_cache, "rb") as f:
            (chinese_characters, 
             chinese_characters_count, 
             chinese_characters_code) = pickle.load(f)
    else:
        print("未找到汉字统计缓存，正在计算...")
        chinese_characters, chinese_characters_count, chinese_characters_code = count_chinese_characters(texts, hanzi_path)
        # 保存结果到缓存
        with open(hanzi_stats_cache, "wb") as f:
            pickle.dump((chinese_characters, 
                        chinese_characters_count, 
                        chinese_characters_code), f)
        print("汉字统计结果已缓存")
    
    print(f"加载了 {len(chinese_characters)} 个汉字")
    
    # 计算相似度矩阵
    print("正在计算汉字相似度矩阵...")
    if os.path.exists(sim_mat_cache):
        print("检测到相似度矩阵缓存，正在加载...")
        sim_mat = np.load(sim_mat_cache)
    else:
        print("未找到相似度矩阵缓存，正在计算...")
        sim_mat = compute_sim_mat(chinese_characters, chinese_characters_code)
        np.save(sim_mat_cache, sim_mat)
        print("相似度矩阵已缓存")
    
    # 转换标签
    labels = ["spam" if tag == "1" else "normal" for tag in tags]

    
    # 初始化判别器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    discriminator = SpamDiscriminator(
        embedding_dim=100,
        hidden_dim=128,
        temperature=0.1,
        contrastive_weight=0.2,  # 对比学习权重
        device=device
    )
    
    # 训练判别器
    print("开始训练判别器...")
    # 在主函数中修改这部分：
    history, train_texts, test_texts, train_labels, test_labels = discriminator.fit(
        texts=texts,  # 传入所有文本
        labels=labels,  # 传入所有标签
        chinese_characters=chinese_characters,
        chinese_characters_count=chinese_characters_count,
        sim_mat=sim_mat,
        test_size=0.2,  # 设置测试集比例
        random_state=42,
        batch_size=32,
        epochs=5,
    )
    
    # 评估模型性能
    print("\n最终模型评估:")
    test_loss, test_acc = discriminator.evaluate(test_texts, test_labels)
    print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")
    
    # 测试判别功能
    print("\n测试判别功能:")
    sample_texts = test_texts[:10]  # 取10个样本进行测试
    predictions, probabilities = discriminator.discriminate(sample_texts)
    
    for i, (text, pred, prob) in enumerate(zip(sample_texts, predictions, probabilities)):
        label = "垃圾文本" if pred == 1 else "正常文本"
        print(f"样本 {i+1}: '{text[:20]}...' - 预测: {label}, 概率: {prob:.4f}")
    
    # 保存模型
    save_path = "./spam_discriminator_model.pth"
    discriminator.save(save_path)
    print(f"\n模型已保存到 {save_path}")
    
    # # 测试模型加载
    # print("\n测试模型加载:")
    # loaded_discriminator = SpamDiscriminator.load(save_path, device)
    # test_loss, test_acc = loaded_discriminator.evaluate(test_texts, test_labels)
    # print(f"加载后的模型 - 测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

if __name__ == "__main__":
    main()