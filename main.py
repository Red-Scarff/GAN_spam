import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random
import os
from tqdm import tqdm
import pickle
from generator.generator import Generator, ReplacementPolicy
from discriminator.ssc_similarity import computeSSCSimilarity
from discriminator.utils import load_chinese_characters, count_chinese_characters, compute_sim_mat
from discriminator.discriminator import SpamDiscriminator


def read_data(filename):
    """读取数据集"""
    with open(filename, "r", encoding="utf-8") as f:
        text_data = f.readlines()

    dataset = [s.strip().split("\t", 1) for s in text_data]
    dataset = [data for data in dataset if len(data) == 2 and data[1].strip()]

    tags = [data[0] for data in dataset]
    texts = [data[1] for data in dataset]

    # 转换标签格式
    labels = ["spam" if tag == "1" else "normal" for tag in tags]

    return labels, texts


def prepare_similarity_function(chinese_characters_code):
    """准备相似度计算函数"""

    def similarity_func(a, b):
        return computeSSCSimilarity(chinese_characters_code[a], chinese_characters_code[b])

    return similarity_func


class SpamGAN:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 初始化组件
        self.discriminator = None
        self.generator = None
        self.similarity_func = None

    def setup_data(self, dataset_path, hanzi_path):
        """设置数据和字符（带缓存机制）"""
        print("正在读取数据...")
        self.labels, self.texts = read_data(dataset_path)
        print(f"读取了 {len(self.texts)} 条数据")

        # 划分数据集
        self.train_labels, self.test_labels, self.train_texts, self.test_texts = train_test_split(
            self.labels, self.texts, test_size=0.2, random_state=42, stratify=self.labels
        )

        # 分离正常和垃圾文本
        self.normal_train_texts = [
            text for text, label in zip(self.train_texts, self.train_labels) if label == "normal"
        ]
        self.spam_train_texts = [text for text, label in zip(self.train_texts, self.train_labels) if label == "spam"]

        print(
            f"训练集: {len(self.train_texts)} (正常: {len(self.normal_train_texts)}, 垃圾: {len(self.spam_train_texts)})"
        )
        print(f"测试集: {len(self.test_texts)}")

        # 创建缓存目录
        cache_dir = "data/cache"
        os.makedirs(cache_dir, exist_ok=True)

        # 定义缓存文件路径
        hanzi_stats_cache = os.path.join(cache_dir, "hanzi_stats.pkl")
        sim_mat_cache = os.path.join(cache_dir, "discriminator_sim_mat.npy")

        # 计算字符统计（带缓存）
        if os.path.exists(hanzi_stats_cache):
            print("检测到汉字统计缓存，正在加载...")
            with open(hanzi_stats_cache, "rb") as f:
                (self.chinese_characters, self.chinese_characters_count, self.chinese_characters_code) = pickle.load(f)
        else:
            print("未找到汉字统计缓存，正在计算...")
            self.chinese_characters, self.chinese_characters_count, self.chinese_characters_code = (
                count_chinese_characters(self.texts, hanzi_path)
            )
            with open(hanzi_stats_cache, "wb") as f:
                pickle.dump((self.chinese_characters, self.chinese_characters_count, self.chinese_characters_code), f)
            print("汉字统计结果已缓存")

        print(f"加载了 {len(self.chinese_characters)} 个汉字")

        self.similarity_func = prepare_similarity_function(self.chinese_characters_code)
        # 计算相似度矩阵（带缓存）
        print("正在处理汉字相似度矩阵...")
        if os.path.exists(sim_mat_cache):
            print("检测到相似度矩阵缓存，正在加载...")
            self.sim_mat = np.load(sim_mat_cache)
        else:
            print("未找到相似度矩阵缓存，正在计算...")
            self.sim_mat = compute_sim_mat(self.chinese_characters, self.chinese_characters_code)
            np.save(sim_mat_cache, self.sim_mat)
            print("相似度矩阵已缓存")

    def init_discriminator(self):
        """初始化判别器"""
        print("初始化判别器...")
        self.discriminator = SpamDiscriminator(
            embedding_dim=self.config["discriminator"]["embedding_dim"],
            hidden_dim=self.config["discriminator"]["hidden_dim"],
            dynamic_threshold=self.config["discriminator"]["dynamic_threshold"],
            device=self.device,
            temperature=self.config["discriminator"]["temperature"],
            contrastive_weight=self.config["discriminator"]["contrastive_weight"],
        )

    def init_generator(self):
        """初始化生成器"""
        print("初始化生成器...")
        self.generator = Generator(
            hidden_dim=self.config["generator"]["hidden_dim"],
            computeSSCSimilarity=self.similarity_func,
            lambda_sim=self.config["generator"]["lambda_sim"],
            base_model_name=self.config["generator"]["base_model_name"],
        ).to(self.device)

    def train_discriminator(self, batch_texts, batch_labels):
        """训练判别器"""

        dis_result = self.discriminator.train_batch(
            batch_texts=batch_texts,
            batch_labels=batch_labels,
            optimizer=self.d_optimizer,
        )

        return dis_result

    def train_generator(self, batch_texts):
        """训练生成器"""

        gen_result = self.generator.train_step(
            batch_texts=batch_texts,  # 使用垃圾文本训练生成器
            discriminator=self.discriminator,
            optimizer=self.g_optimizer,
            device=self.device,
        )
        return gen_result

    def adversarial_training(self):
        """对抗训练"""
        print("=" * 50)
        print("开始对抗训练...")
        print("=" * 50)

        # 设置优化器
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.config["generator"]["learning_rate"])
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.config["discriminator"]["learning_rate"]
        )
        print("文本预处理")
        cleaned_texts = self.discriminator._clean_texts(self.texts)
        tokenized_texts = self.discriminator._tokenize_and_remove_stopwords(cleaned_texts)

        print("生成词向量和字符向量")
        self.discriminator.w2v_vectors = self.discriminator._generate_w2v_vectors(tokenized_texts)
        self.discriminator.char_vectors = self.discriminator._generate_char_vectors(
            self.chinese_characters,
            self.discriminator.w2v_vectors,
            self.sim_mat,
            tokenized_texts,
            self.chinese_characters_count,
        )
        spam_batch_size = self.config["gan"]["spam_batch_size"]
        normal_batch_size = self.config["gan"]["normal_batch_size"]
        steps_per_epoch = self.config["gan"].get("steps_per_epoch", len(self.spam_train_texts) // spam_batch_size)

        for epoch in range(self.config["gan"]["epochs"]):
            print(f"\nEpoch {epoch + 1}/{self.config['gan']['epochs']}")
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            # 打乱训练数据
            random.shuffle(self.spam_train_texts)
            random.shuffle(self.normal_train_texts)
            # 创建进度条
            progress_bar = tqdm(range(steps_per_epoch), desc=f"Training")

            for step in progress_bar:
                # === 准备训练数据 ===
                # 顺序抽取垃圾文本作为生成器输入
                batch_spam_texts = self.spam_train_texts[step * spam_batch_size : (step + 1) * spam_batch_size]

                # 顺序抽取正常文本
                batch_normal_texts = self.normal_train_texts[
                    step
                    * normal_batch_size
                    % len(self.normal_train_texts) : (step + 1)
                    * normal_batch_size
                    % len(self.normal_train_texts)
                ]

                # === 训练生成器 ===
                g_result = self.train_generator(batch_spam_texts)
                epoch_g_loss += g_result["gen_loss"]

                # 获取生成的文本
                generated_spam_texts = []
                for text in batch_spam_texts:
                    generated_text = self.generator.generate(text, self.device)
                    generated_spam_texts.append(generated_text)

                # === 训练判别器 ===
                # 构建判别器训练数据：正常文本(label=0) + 真实垃圾文本(label=1) + 生成的文本(label=1)
                d_train_texts = batch_normal_texts + batch_spam_texts + generated_spam_texts
                d_train_labels = (
                    ["normal"] * len(batch_normal_texts)
                    + ["spam"] * len(batch_spam_texts)
                    + ["spam"] * len(generated_spam_texts)
                )

                d_result = self.train_discriminator(d_train_texts, d_train_labels)
                epoch_d_loss += d_result["dis_loss"]

                # 更新进度条
                progress_bar.set_postfix(
                    {
                        "G_Loss": f"{g_result['gen_loss']:.4f}",
                        "D_Loss": f"{d_result['dis_loss']:.4f}",
                    }
                )

            # 输出epoch统计
            avg_g_loss = epoch_g_loss / steps_per_epoch
            avg_d_loss = epoch_d_loss / steps_per_epoch
            print(f"Epoch {epoch + 1} 完成:")
            print(f"  平均生成器损失: {avg_g_loss:.4f}")
            print(f"  平均判别器损失: {avg_d_loss:.4f}")

            # 每个epoch结束后进行一次性能评估
            print(f"\n--- Epoch {epoch + 1} 性能评估 ---")
            self.evaluate_epoch_performance()

    def evaluate_epoch_performance(self):
        """生成器和判别器的性能评估"""

        # 生成一些测试样本
        test_spam_sample = random.sample(self.spam_train_texts, 20)
        generated_samples = []

        for text in test_spam_sample:
            gen_text = self.generator.generate(text, self.device)
            generated_samples.append(gen_text)

        # 用判别器评估生成的文本
        predictions, probabilities = self.discriminator.discriminate(generated_samples)

        # 计算统计指标
        spam_count = sum(1 for pred in predictions if pred == 1)

        print(
            f"生成文本被判别为垃圾邮件的比例: {spam_count}/{len(predictions)} ({spam_count/len(predictions)*100:.1f}%)"
        )

        test_loss, test_accuracy = self.discriminator.evaluate(
            self.test_texts,
            self.test_labels,
        )

        print(f"判别器测试集损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}")

        # 显示一个生成示例
        print("\n生成示例:")
        orig_text = test_spam_sample[0]
        gen_text = generated_samples[0]
        print(f"  原文: {orig_text}")
        print(f"  生成: {gen_text}")
        print(f"  判别: {predictions[0]} (概率: {probabilities[0]:.3f})")
        print()

    def save_models(self, save_dir="models"):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存判别器
        discriminator_path = os.path.join(save_dir, "discriminator.pth")
        self.discriminator.save(discriminator_path)

        # 保存生成器
        generator_path = os.path.join(save_dir, "generator.pth")
        torch.save(self.generator.state_dict(), generator_path)

        print(f"模型已保存到 {save_dir}")

    def load_models(self, save_dir="models"):
        """加载模型"""
        # 加载判别器
        discriminator_path = os.path.join(save_dir, "discriminator.pth")
        if os.path.exists(discriminator_path):
            self.discriminator = SpamDiscriminator.load(discriminator_path, self.device)
            print("判别器模型已加载")

        # 加载生成器
        generator_path = os.path.join(save_dir, "generator.pth")
        if os.path.exists(generator_path):
            self.init_generator()  # 先初始化结构
            self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
            print("生成器模型已加载")


def main():
    # 配置参数
    config = {
        "discriminator": {
            "embedding_dim": 100,
            "hidden_dim": 128,
            "dynamic_threshold": True,
            "temperature": 0.1,
            "contrastive_weight": 0.3,
            "learning_rate": 0.001,
        },
        "generator": {
            "hidden_dim": 768,
            "lambda_sim": 0.3,
            "base_model_name": "/root/nfs/GAN_spam/models/bert-base-chinese",
            "learning_rate": 5e-5,
        },
        "gan": {
            "epochs": 5,
            "spam_batch_size": 32,
            "normal_batch_size": 32,
        },
    }

    # 数据路径
    dataset_path = "data/dataset.txt"
    hanzi_path = "data/hanzi.txt"

    # 初始化GAN
    gan = SpamGAN(config)

    # 设置数据
    gan.setup_data(dataset_path, hanzi_path)

    # 检查是否有预训练模型
    if os.path.exists("models/discriminator.pth") and os.path.exists("models/generator.pth"):
        print("发现预训练模型，是否加载？(y/n)")
        choice = input().strip().lower()
        if choice == "y":
            gan.load_models()
        else:
            # 从头训练
            gan.init_discriminator()
            gan.init_generator()
    else:
        # 从头训练
        gan.init_discriminator()
        gan.init_generator()

    # 对抗训练
    gan.adversarial_training()

    # 最终评估
    print("\n" + "=" * 50)
    print("最终评估")
    print("=" * 50)
    gan.evaluate_epoch_performance()

    # 保存模型
    gan.save_models()

    print("\n训练完成！")


if __name__ == "__main__":
    main()
