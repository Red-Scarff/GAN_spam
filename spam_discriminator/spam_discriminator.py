import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import re
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix, classification_report
from utils import *

class SpamDiscriminator(nn.Module):
    def __init__(self, embedding_dim=100, hidden_dim=128, dynamic_threshold=True, 
                 threshold_init=0.6, device=None):
        super(SpamDiscriminator, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dynamic_threshold = dynamic_threshold
        self.threshold = nn.Parameter(torch.tensor(threshold_init), requires_grad=dynamic_threshold)
        self.char_similarity_projection = nn.Linear(embedding_dim, hidden_dim)
        self.query_proj = nn.Linear(embedding_dim, hidden_dim)
        self.key_proj = nn.Linear(embedding_dim, hidden_dim)
        self.value_proj = nn.Linear(embedding_dim, hidden_dim)
        self.attention_scale = hidden_dim ** -0.5
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.char_vectors = {}
        self.w2v_vectors = {}

        
        # 移动所有参数到指定设备
        self.to(self.device)
        
        # # 调试：检查参数设备
        # for name, param in self.named_parameters():
        #     print(f"参数 {name} 在设备: {param.device}")

    def _generate_w2v_vectors(self, tokenized_texts, d=100):
        """
        生成词向量，存储为张量
        """
        if not tokenized_texts or not any(tokenized_texts):
            print("警告: tokenized_texts 为空，返回空词向量字典")
            return {}
        # 将字符串拆分为字符列表
        tokenized_texts = [[char for char in text] for text in tokenized_texts]
        model = Word2Vec(sentences=tokenized_texts, vector_size=d, window=5, min_count=1, sg=0)
        word_vectors = model.wv
        w2v_vectors = {}
        for tokens in tokenized_texts:
            for word in tokens:
                if word not in w2v_vectors and word in word_vectors:
                    # 直接存储张量
                    w2v_vectors[word] = torch.tensor(word_vectors[word], dtype=torch.float32, device=self.device)
        return w2v_vectors

    def _generate_char_vectors(self, chinese_characters, w2v_vectors, sim_mat, texts, 
                              chinese_characters_count, threshold=None):
        """
        生成字符向量
        """
        # 转换为 NumPy 数组
        sim_mat = np.array(sim_mat, dtype=np.float32)
        
        # 检查 sim_mat 形状
        if sim_mat.shape != (len(chinese_characters), len(chinese_characters)):
            raise ValueError(f"sim_mat 形状 {sim_mat.shape} 不匹配 chinese_characters 长度 {len(chinese_characters)}")
        
        # 使用动态阈值或固定阈值
        threshold_value = threshold if threshold is not None else self.threshold.item()
        
        # 转换为 PyTorch 张量
        sim_mat_tensor = torch.tensor(sim_mat, dtype=torch.float32, device=self.device)
        
        # 获取向量维度
        if not w2v_vectors:
            return {}
        vec_dim = self.embedding_dim
        
        char_vectors = {}
        for i in range(len(chinese_characters)):
            character = chinese_characters[i]
            similar_indices = torch.where(sim_mat_tensor[i] >= threshold_value)[0].cpu().numpy()
            similar_group = [chinese_characters[j] for j in similar_indices]
            sum_count = 0
            emb = torch.zeros(vec_dim, dtype=torch.float32, device=self.device)
            for c in similar_group:
                if c not in w2v_vectors:
                    self._update_w2v_vectors(w2v_vectors, texts, c, vec_dim)
                if c in w2v_vectors:
                    # 直接使用存储的张量
                    c_vec = w2v_vectors[c]
                    count = chinese_characters_count.get(c, 0)
                    emb += count * c_vec
                    sum_count += count
            if sum_count > 0:
                emb /= sum_count
            char_vectors[character] = emb
        return char_vectors

    def _generate_sentence_vectors(self, texts):
        """
        根据字符相似性网络生成句子向量表示
        """
        sentence_vectors = []
        for text in texts:
            if not text or not isinstance(text, str):
                sentence_vectors.append(torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.device))
                continue
            char_tensors = []
            for char in text:
                if char in self.char_vectors:
                    char_vec = self.char_vectors[char]
                else:
                    char_vec = torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.device)
                char_tensors.append(char_vec)
            if not char_tensors:
                sentence_vectors.append(torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.device))
                continue
            char_matrix = torch.stack(char_tensors)
            alpha = torch.matmul(char_matrix, char_matrix.t()) / (self.embedding_dim ** 0.5)
            alpha_exp = torch.exp(alpha)
            alpha_sum = alpha_exp.sum(dim=1, keepdim=True)
            alpha_hat = alpha_exp / (alpha_sum + 1e-8)
            m = torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.device)
            for i in range(len(text)):
                mi = torch.matmul(alpha_hat[i].unsqueeze(0), char_matrix).squeeze(0)
                m += mi
            m = m / self.embedding_dim
            sentence_vectors.append(m)
        return sentence_vectors

    def _apply_self_attention(self, sentence_tensors):
        """
        应用自注意力机制处理句子向量
        
        Args:
            sentence_tensors: 句子向量张量 [batch_size, embedding_dim]
            
        Returns:
            attention_output: 经过自注意力处理后的向量 [batch_size, hidden_dim]
        """
        if sentence_tensors.dim() != 2:
            raise ValueError(f"预期输入张量为 2D，实际形状为 {sentence_tensors.shape}")
        # 调试：检查张量和参数设备
        #print(f"sentence_tensors 设备: {sentence_tensors.device}")
        #print(f"query_proj.weight 设备: {self.query_proj.weight.device}")
        
        queries = self.query_proj(sentence_tensors)  # [batch_size, hidden_dim]
        keys = self.key_proj(sentence_tensors)      # [batch_size, hidden_dim]
        values = self.value_proj(sentence_tensors)  # [batch_size, hidden_dim]
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.attention_scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)  # [batch_size, hidden_dim]
        
        return attention_output  # 直接返回 [batch_size, hidden_dim]

    def fit(self, texts, labels, chinese_characters, chinese_characters_count, 
            sim_mat, test_size=0.5, random_state=42,
            batch_size=32, epochs=5, learning_rate=0.001):
        """
        训练判别器模型
        """
        # 调试信息
        # print(f"texts 长度: {len(texts)}, 示例: {texts[:2]}")
        # print(f"labels 长度: {len(labels)}, 示例: {labels[:2]}")
        # print(f"chinese_characters 长度: {len(chinese_characters)}")
        # print(f"sim_mat 类型: {type(sim_mat)}, 形状: {np.array(sim_mat).shape}")
        
        # 检查标签格式
        if not all(label in ["spam", "normal"] for label in labels):
            raise ValueError("标签必须为 'spam' 或 'normal'")
        
        # 确保模型参数在正确设备上
        self.to(self.device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        print("文本预处理")
        cleaned_texts = self._clean_texts(texts)
        tokenized_texts = self._tokenize_and_remove_stopwords(cleaned_texts)
        print("生成词向量和字符向量")
        self.w2v_vectors = self._generate_w2v_vectors(tokenized_texts)
        self.char_vectors = self._generate_char_vectors(
            chinese_characters, self.w2v_vectors, sim_mat, 
            tokenized_texts, chinese_characters_count
        )
        #print(f"char_vectors 示例: {list(self.char_vectors.items())[:2]}")
        
        print("划分训练集和测试集...")
        from sklearn.model_selection import train_test_split
        train_indices, test_indices, train_labels_split, test_labels_split = train_test_split(
            list(range(len(texts))), labels, test_size=test_size, random_state=random_state
        )
        train_texts = [texts[i] for i in train_indices]
        test_texts = [texts[i] for i in test_indices]
        train_labels_tensor = torch.tensor([1.0 if label == "spam" else 0.0 
                                        for label in train_labels_split], 
                                        dtype=torch.float32, device=self.device)
        history = {
            "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []
        }
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}:")
            self.train()
            total_loss = 0
            correct = 0
            train_shuffle_indices = list(range(len(train_texts)))
            np.random.shuffle(train_shuffle_indices)
            for i in range(0, len(train_shuffle_indices), batch_size):
                batch_indices = train_shuffle_indices[i:i+batch_size]
                batch_texts = [train_texts[idx] for idx in batch_indices]
                batch_labels = train_labels_tensor[batch_indices]
                batch_sentence_vectors = self._generate_sentence_vectors(batch_texts)
                batch_sentence_tensors = torch.stack(batch_sentence_vectors)
                #print(f"批次大小: {len(batch_texts)}, 张量形状: {batch_sentence_tensors.shape}, 设备: {batch_sentence_tensors.device}")
                optimizer.zero_grad()
                attention_output = self._apply_self_attention(batch_sentence_tensors)  # [batch_size, hidden_dim]
                predictions = self.classifier(attention_output).squeeze(-1)  # [batch_size]
                #print(f"predictions 形状: {predictions.shape}, batch_labels 形状: {batch_labels.shape}")
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                predictions_binary = (predictions >= 0.5).float()
                correct += (predictions_binary == batch_labels).sum().item()
            train_loss = total_loss / (len(train_shuffle_indices) / batch_size)
            train_acc = correct / len(train_texts)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            test_loss, test_acc = self.evaluate(test_texts, test_labels_split)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)
            print(f"Epoch {epoch+1}/{epochs}: 训练损失={train_loss:.4f}, 训练准确率={train_acc:.4f}, "
                f"测试损失={test_loss:.4f}, 测试准确率={test_acc:.4f}")

        return history, train_texts, test_texts, train_labels_split, test_labels_split

    def forward(self, texts):
        """
        前向传播函数
        
        Args:
            texts: 输入文本列表
            
        Returns:
            predictions: 垃圾文本预测概率 [batch_size]
        """

        print("开始前向传播...")
        sentence_vectors = self._generate_sentence_vectors(texts)
        sentence_tensors = torch.stack(sentence_vectors)
        attention_output = self._apply_self_attention(sentence_tensors)  # [batch_size, hidden_dim]
        predictions = self.classifier(attention_output).squeeze(-1)  # [batch_size]
        return predictions

    

    def evaluate(self, texts, labels):
        """
        评估模型性能
        
        Args:
            texts: 测试文本列表
            labels: 测试标签列表
            
        Returns:
            loss: 测试损失
            accuracy: 测试准确率
        """
        self.eval()
        criterion = nn.BCELoss()
        labels_tensor = torch.tensor([1.0 if label == "spam" else 0.0 
                                    for label in labels], 
                                    dtype=torch.float32, device=self.device)
        with torch.no_grad():
            predictions = self(texts)
            print(f"evaluate predictions 形状: {predictions.shape}, labels_tensor 形状: {labels_tensor.shape}")
            loss = criterion(predictions, labels_tensor)
            predictions_binary = (predictions >= 0.5).float()
            accuracy = (predictions_binary == labels_tensor).sum().item() / len(texts)
        return loss.item(), accuracy

    def discriminate(self, texts):
        self.eval()
        with torch.no_grad():
            probabilities = self(texts).cpu().numpy()
            predictions = (probabilities >= 0.5).astype(int)
        return predictions, probabilities

    def gan_loss(self, real_normal_texts, real_spam_texts, generated_texts):
        """
        计算GAN判别器损失
        
        Args:
            real_normal_texts: 真实正常文本列表
            real_spam_texts: 真实垃圾文本列表
            generated_texts: 生成器生成的文本列表
            
        Returns:
            loss: 判别器损失
            real_normal_acc: 真实正常文本判别准确率
            real_spam_acc: 真实垃圾文本判别准确率
            generated_acc: 生成文本判别准确率
        """
        self.train()
        real_normal_preds = self(real_normal_texts)
        real_normal_labels = torch.zeros(len(real_normal_texts), dtype=torch.float32, device=self.device)
        loss_real_normal = F.binary_cross_entropy(real_normal_preds, real_normal_labels)
        real_spam_preds = self(real_spam_texts)
        real_spam_labels = torch.ones(len(real_spam_texts), dtype=torch.float32, device=self.device)
        loss_real_spam = F.binary_cross_entropy(real_spam_preds, real_spam_labels)
        generated_preds = self(generated_texts)
        generated_labels = torch.ones(len(generated_texts), dtype=torch.float32, device=self.device)
        loss_generated = F.binary_cross_entropy(generated_preds, generated_labels)
        total_loss = loss_real_normal + loss_real_spam + loss_generated
        real_normal_acc = ((real_normal_preds < 0.5).float().mean()).item()
        real_spam_acc = ((real_spam_preds >= 0.5).float().mean()).item()
        generated_acc = ((generated_preds >= 0.5).float().mean()).item()
        return total_loss, real_normal_acc, real_spam_acc, generated_acc


    
    def get_reward_for_generator(self, generated_texts, original_texts=None, lambda_param=0.7):
        """
        为生成器提供奖励信号
        
        reward = λ * (1-D(x̂)) + (1-λ) * similarity(x, x̂)
        
        Args:
            generated_texts: 生成器生成的文本列表
            original_texts: 原始文本列表(用于计算相似度)
            lambda_param: 平衡因子
            
        Returns:
            rewards: 生成文本的奖励值
        """
        self.eval()
        
        with torch.no_grad():
            # 判别器打分(欺骗成功)
            discriminator_scores = self(generated_texts).cpu().numpy()
            deception_reward = 1 - discriminator_scores  # 欺骗判别器的奖励
            
            # 如果提供了原始文本，则计算相似度奖励
            if original_texts:
                similarities = self._compute_text_similarities(original_texts, generated_texts)
                rewards = lambda_param * deception_reward + (1 - lambda_param) * similarities
            else:
                rewards = deception_reward
                
        return rewards
    
    def _compute_text_similarities(self, original_texts, generated_texts):
        """
        计算原始文本与生成文本之间的相似度

        Args:
            original_texts: 原始文本列表
            generated_texts: 生成文本列表
            
        Returns:
            similarities: 相似度列表(0-1之间)
        """
        assert len(original_texts) == len(generated_texts), "文本列表长度不匹配"
        
        similarities = []
        for orig, gen in zip(original_texts, generated_texts):
            # 基于字符重叠率的简单相似度计算
            orig_chars = set(orig)
            gen_chars = set(gen)
            
            if not orig_chars or not gen_chars:
                similarities.append(0.0)
                continue
                
            overlap = len(orig_chars.intersection(gen_chars))
            similarity = overlap / max(len(orig_chars), len(gen_chars))
            similarities.append(similarity)
            
        return np.array(similarities)
    
    def _clean_texts(self, texts):
        """
        清洗文本
        
        Args:
            texts: 文本列表
            
        Returns:
            cleaned_texts: 清洗后的文本列表
        """
        cleaned_texts = []
        for text in texts:
            clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
            cleaned_texts.append(clean.strip())
        return cleaned_texts
    
    def _tokenize_and_remove_stopwords(self, texts, stopwords_file=None):
        """
        分词并移除停用词
        
        Args:
            texts: 文本列表
            stopwords_file: 停用词文件路径
            
        Returns:
            tokenized_texts: 处理后的文本列表
        """
        # 获取停用词
        stopwords = set()
        if stopwords_file:
            try:
                with open(stopwords_file, 'r', encoding='utf-8') as file:
                    stopwords = {line.strip() for line in file}
            except FileNotFoundError:
                print(f"警告: 停用词文件 {stopwords_file} 未找到。不使用停用词过滤。")
        
        # 分词并移除停用词
        tokenized_texts = []
        for text in texts:
            cleaned_text = ''.join([char for char in text 
                                   if char not in stopwords and re.search("[\u4e00-\u9fa5]", char)])
            tokenized_texts.append(cleaned_text)
            
        return tokenized_texts
    
    
    
    def _update_w2v_vectors(self, w2v_vectors, texts, character, d=100):
        """
        更新词向量
        
        Args:
            w2v_vectors: 词向量字典
            texts: 文本列表
            character: 字符
            d: 向量维度
            
        Returns:
            w2v_vectors: 更新后的词向量字典
        """
        model = Word2Vec(sentences=texts, vector_size=d, window=5, min_count=1, sg=0)
        word_vectors = model.wv
        if character in word_vectors:
            w2v_vectors[character] = word_vectors[character]
            
        return w2v_vectors
    

    
    def save(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 文件路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'dynamic_threshold': self.dynamic_threshold,
            'char_vectors': self.char_vectors,
            'w2v_vectors': self.w2v_vectors
        }, filepath)
        print(f"模型已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath, device=None):
        """
        加载模型
        
        Args:
            filepath: 文件路径
            device: 计算设备
            
        Returns:
            model: 加载的模型
        """
        device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            dynamic_threshold=checkpoint['dynamic_threshold'],
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.char_vectors = checkpoint['char_vectors']
        model.w2v_vectors = checkpoint['w2v_vectors']
        model.to(device)
        
        return model


"""
#示例用法
class GANSpamDetector:
    
    def __init__(self, discriminator=None, generator=None, device=None):

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化判别器和生成器
        self.discriminator = discriminator if discriminator else SpamDiscriminator(device=self.device)
        self.generator = generator
        
    def train_discriminator(self, real_normal_texts, real_spam_texts, generated_texts=None, 
                          optimizer=None, epochs=1):

        if optimizer is None:
            optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
            
        metrics = {
            'loss': [],
            'real_normal_acc': [],
            'real_spam_acc': [],
            'generated_acc': []
        }
        
        # 如果没有生成文本，使用空列表
        if generated_texts is None:
            generated_texts = []
            
        for epoch in range(epochs):
            # 计算GAN损失
            loss, real_normal_acc, real_spam_acc, generated_acc = self.discriminator.gan_loss(
                real_normal_texts, real_spam_texts, generated_texts
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录指标
            metrics['loss'].append(loss.item())
            metrics['real_normal_acc'].append(real_normal_acc)
            metrics['real_spam_acc'].append(real_spam_acc)
            metrics['generated_acc'].append(generated_acc)
            
            print(f"Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, "
                  f"Normal Acc={real_normal_acc:.4f}, Spam Acc={real_spam_acc:.4f}, "
                  f"Generated Acc={generated_acc:.4f}")
            
        return metrics

"""