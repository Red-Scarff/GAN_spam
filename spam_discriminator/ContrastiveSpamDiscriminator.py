import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import re
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix, classification_report
import random
from collections import defaultdict
import math

class ContrastiveSpamDiscriminator(nn.Module):
    def __init__(self, embedding_dim=100, hidden_dim=128, dynamic_threshold=True, 
                 threshold_init=0.6, device=None, temperature=0.1, contrastive_weight=0.3,
                 projection_dim=128):
        super(ContrastiveSpamDiscriminator, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dynamic_threshold = dynamic_threshold
        self.threshold = nn.Parameter(torch.tensor(threshold_init), requires_grad=dynamic_threshold)
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.projection_dim = projection_dim
        
        # 改进的自注意力机制：多头注意力
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        assert self.head_dim * self.num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=self.num_heads, 
            dropout=0.1,
            batch_first=True
        )
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 对比学习投影头
        self.contrastive_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
            L2Norm(dim=-1)  # L2归一化
        )
        
        # 改进的分类器：真正的残差连接
        self.classifier_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier_norm1 = nn.LayerNorm(hidden_dim)
        self.classifier_layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.classifier_norm2 = nn.LayerNorm(hidden_dim // 2)
        self.classifier_output = nn.Linear(hidden_dim // 2, 1)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        
        # 缓存机制
        self.char_vectors = {}
        self.w2v_vectors = {}
        self._sentence_cache = {}
        
        # 移动所有参数到指定设备
        self.to(self.device)

    def _classify_with_residual(self, x):
        """
        带残差连接的分类器
        
        Args:
            x: 输入特征 [batch_size, hidden_dim]
            
        Returns:
            predictions: 分类预测 [batch_size]
        """
        # 第一个残差块
        residual = x
        x = self.classifier_layer1(x)
        x = self.classifier_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = x + residual  # 残差连接
        
        # 第二个块（维度变化，无法直接残差连接）
        x = self.classifier_layer2(x)
        x = self.classifier_norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # 输出层
        x = self.classifier_output(x)
        x = torch.sigmoid(x)
        
        return x.squeeze(-1)

    def _generate_w2v_vectors(self, tokenized_texts, d=100):
        """
        优化的词向量生成，增加缓存机制
        """
        if not tokenized_texts or not any(tokenized_texts):
            print("警告: tokenized_texts 为空，返回空词向量字典")
            return {}
        
        # 将字符串拆分为字符列表
        tokenized_texts = [[char for char in text if char.strip()] for text in tokenized_texts]
        tokenized_texts = [text for text in tokenized_texts if text]  # 过滤空列表
        
        if not tokenized_texts:
            return {}
            
        # 使用更好的Word2Vec参数
        model = Word2Vec(
            sentences=tokenized_texts, 
            vector_size=d, 
            window=5, 
            min_count=1, 
            sg=1,  # 使用Skip-gram
            workers=4,
            epochs=10
        )
        word_vectors = model.wv
        w2v_vectors = {}
        
        for tokens in tokenized_texts:
            for word in tokens:
                if word not in w2v_vectors and word in word_vectors:
                    w2v_vectors[word] = torch.tensor(
                        word_vectors[word], 
                        dtype=torch.float32, 
                        device=self.device
                    )
        return w2v_vectors

    def _generate_char_vectors(self, chinese_characters, w2v_vectors, sim_mat, texts, 
                              chinese_characters_count, threshold=None):
        """
        优化的字符向量生成
        """
        if not chinese_characters or not w2v_vectors:
            return {}
            
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
                    c_vec = w2v_vectors[c]
                    count = chinese_characters_count.get(c, 1)  # 避免除零
                    emb += count * c_vec
                    sum_count += count
                    
            if sum_count > 0:
                emb /= sum_count
            else:
                # 如果没有相似字符，使用随机初始化
                emb = torch.randn(vec_dim, dtype=torch.float32, device=self.device) * 0.01
                
            char_vectors[character] = emb
        return char_vectors

    def _apply_multihead_attention_to_chars(self, char_tensors):
        """
        应用多头自注意力机制和位置编码
        
        Args:
            char_tensors: 字符向量张量 [seq_len, embedding_dim]
            
        Returns:
            sentence_vector: 经过处理后的句子向量 [hidden_dim]
        """
        if char_tensors.dim() != 2:
            raise ValueError(f"预期输入张量为 2D，实际形状为 {char_tensors.shape}")
        
        seq_len, embedding_dim = char_tensors.shape
        
        # 添加位置编码
        char_tensors = self.positional_encoding(char_tensors.unsqueeze(0))  # [1, seq_len, embedding_dim]
        
        # 应用多头注意力
        attention_output, attention_weights = self.multihead_attention(
            char_tensors, char_tensors, char_tensors
        )  # [1, seq_len, embedding_dim]
        
        # 特征投影
        projected_features = self.feature_projection(attention_output.squeeze(0))  # [seq_len, hidden_dim]
        
        # 使用加权平均而不是简单平均
        weights = F.softmax(torch.sum(projected_features, dim=-1), dim=0)  # [seq_len]
        sentence_vector = torch.sum(projected_features * weights.unsqueeze(-1), dim=0)  # [hidden_dim]
        
        return sentence_vector

    def _generate_sentence_vectors_with_attention(self, texts):
        """
        优化的句子向量生成，包含缓存机制
        """
        sentence_vectors = []
        for text in texts:
            # 使用缓存
            if text in self._sentence_cache:
                cached_vec = self._sentence_cache[text]
                sentence_vectors.append(cached_vec.detach().clone() if cached_vec.requires_grad else cached_vec)
                continue

            # 对于空文本，返回零向量
            if not text or not isinstance(text, str):
                zero_vec = torch.zeros(self.hidden_dim, dtype=torch.float32, device=self.device)
                sentence_vectors.append(zero_vec)
                self._sentence_cache[text] = zero_vec
                continue
                
            char_tensors = []
            for char in text:
                if char in self.char_vectors:
                    char_vec = self.char_vectors[char]
                else:
                    char_vec = torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.device)
                char_tensors.append(char_vec)
                
            if not char_tensors:
                zero_vec = torch.zeros(self.hidden_dim, dtype=torch.float32, device=self.device)
                sentence_vectors.append(zero_vec)
                self._sentence_cache[text] = zero_vec
                continue
                
            # 堆叠字符向量
            char_matrix = torch.stack(char_tensors)  # [seq_len, embedding_dim]
            
            # 应用改进的注意力机制
            sentence_vector = self._apply_multihead_attention_to_chars(char_matrix)  # [hidden_dim]
            sentence_vectors.append(sentence_vector)
            self._sentence_cache[text] = sentence_vector.detach().clone()
            
        return sentence_vectors

    def contrastive_loss(self, embeddings, labels, temperature=None):
        """
        计算对比学习损失（InfoNCE Loss）
        
        Args:
            embeddings: 句子嵌入 [batch_size, hidden_dim]
            labels: 标签 [batch_size]
            temperature: 温度参数
            
        Returns:
            loss: 对比学习损失
        """
        if temperature is None:
            temperature = self.temperature
            
        batch_size = embeddings.shape[0]
        
        # 获取对比学习投影
        projections = self.contrastive_projection(embeddings)  # [batch_size, projection_dim]
        
        # 计算相似度矩阵（对比学习相似度矩阵，与字符相似度是两个概念）
        similarity_matrix = torch.matmul(projections, projections.T) / temperature  # [batch_size, batch_size]
        
        # 创建标签掩码
        labels = labels.unsqueeze(1)  # [batch_size, 1]
        label_mask = torch.eq(labels, labels.T).float()  # [batch_size, batch_size]
        
        # 移除对角线（自己与自己的相似度）
        identity_mask = torch.eye(batch_size, device=self.device)
        label_mask = label_mask * (1 - identity_mask)
        
        # 正样本掩码（同类别）
        positive_mask = label_mask
        # 负样本掩码（不同类别）
        negative_mask = (1 - label_mask) * (1 - identity_mask)
        
        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        
        # 对于每个样本，计算其正样本和负样本的损失
        loss = 0.0
        num_positive_pairs = 0
        
        for i in range(batch_size):
            # 当前样本的正样本
            positive_indices = torch.where(positive_mask[i] > 0)[0]
            
            if len(positive_indices) == 0:
                continue
                
            # 分母：当前样本与所有其他样本的相似度（除了自己）
            denominator = torch.sum(exp_sim[i] * (1 - identity_mask[i]))
            
            # 对每个正样本计算损失
            for pos_idx in positive_indices:
                numerator = exp_sim[i, pos_idx]
                loss += -torch.log(numerator / (denominator + 1e-8))
                num_positive_pairs += 1
        
        if num_positive_pairs > 0:
            loss = loss / num_positive_pairs
        
        return loss

    def create_contrastive_pairs(self, texts, labels, batch_size):
        """
        创建对比学习的样本对
        
        Args:
            texts: 文本列表
            labels: 标签列表
            batch_size: 批次大小
            
        Returns:
            paired_texts: 配对的文本
            paired_labels: 配对的标签
        """
        # 按标签分组
        spam_texts = [text for text, label in zip(texts, labels) if label == "spam"]
        normal_texts = [text for text, label in zip(texts, labels) if label == "normal"]
        
        paired_texts = []
        paired_labels = []
        
        # 确保每个批次都有正负样本
        for _ in range(batch_size // 2):
            if spam_texts and normal_texts:
                # 添加垃圾文本
                paired_texts.append(random.choice(spam_texts))
                paired_labels.append("spam")
                
                # 添加正常文本
                paired_texts.append(random.choice(normal_texts))
                paired_labels.append("normal")
        
        # 如果批次大小是奇数，随机添加一个
        if len(paired_texts) < batch_size:
            all_texts = spam_texts + normal_texts
            all_labels = ["spam"] * len(spam_texts) + ["normal"] * len(normal_texts)
            idx = random.randint(0, len(all_texts) - 1)
            paired_texts.append(all_texts[idx])
            paired_labels.append(all_labels[idx])
        
        return paired_texts[:batch_size], paired_labels[:batch_size]

    def fit(self, texts, labels, chinese_characters, chinese_characters_count, 
            sim_mat, test_size=0.2, random_state=42,
            batch_size=32, epochs=5, learning_rate=0.001, warmup_epochs=1):
        """
        训练判别器模型，包含对比学习
        """
        # 检查标签格式
        if not all(label in ["spam", "normal"] for label in labels):
            raise ValueError("标签必须为 'spam' 或 'normal'")
        
        # 确保模型参数在正确设备上
        self.to(self.device)
        
        # 使用学习率调度器
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        criterion = nn.BCELoss()
        
        print("文本预处理...")
        cleaned_texts = self._clean_texts(texts)
        tokenized_texts = self._tokenize_and_remove_stopwords(cleaned_texts)
        
        print("生成词向量和字符向量...")
        self.w2v_vectors = self._generate_w2v_vectors(tokenized_texts)
        self.char_vectors = self._generate_char_vectors(
            chinese_characters, self.w2v_vectors, sim_mat, 
            tokenized_texts, chinese_characters_count
        )
        
        print("划分训练集和测试集...")
        from sklearn.model_selection import train_test_split
        train_indices, test_indices, train_labels_split, test_labels_split = train_test_split(
            list(range(len(texts))), labels, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
        train_texts = [texts[i] for i in train_indices]
        test_texts = [texts[i] for i in test_indices]
        
        history = {
            "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
            "contrastive_loss": [], "classification_loss": []
        }
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}:")
            self.train()
            
            total_loss = 0
            total_contrastive_loss = 0
            total_classification_loss = 0
            correct = 0
            
            # 创建训练批次
            num_batches = len(train_texts) // batch_size
            
            for batch_idx in range(num_batches):
                # 创建对比学习样本对
                batch_texts, batch_labels_str = self.create_contrastive_pairs(
                    train_texts, train_labels_split, batch_size
                )
                
                batch_labels = torch.tensor([1.0 if label == "spam" else 0.0 
                                           for label in batch_labels_str], 
                                           dtype=torch.float32, device=self.device)
                
                # 生成句子向量
                batch_sentence_vectors = self._generate_sentence_vectors_with_attention(batch_texts)
                batch_sentence_tensors = torch.stack(batch_sentence_vectors)  # [batch_size, hidden_dim]
                
                optimizer.zero_grad()
                
                # 分类损失
                predictions = self._classify_with_residual(batch_sentence_tensors).squeeze(-1)  # [batch_size]
                classification_loss = criterion(predictions, batch_labels)
                
                # 对比学习损失
                contrastive_loss = self.contrastive_loss(batch_sentence_tensors, batch_labels)
                
                # 总损失
                if epoch < warmup_epochs:
                    # 预热阶段：只使用分类损失
                    total_batch_loss = classification_loss
                else:
                    # 正常训练：结合两种损失
                    total_batch_loss = (1 - self.contrastive_weight) * classification_loss + \
                                     self.contrastive_weight * contrastive_loss
                
                total_batch_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += total_batch_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_classification_loss += classification_loss.item()
                
                predictions_binary = (predictions >= 0.5).float()
                correct += (predictions_binary == batch_labels).sum().item()
            
            scheduler.step()
            
            # 计算平均损失和准确率
            avg_loss = total_loss / num_batches
            avg_contrastive_loss = total_contrastive_loss / num_batches
            avg_classification_loss = total_classification_loss / num_batches
            train_acc = correct / (num_batches * batch_size)
            
            history["train_loss"].append(avg_loss)
            history["contrastive_loss"].append(avg_contrastive_loss)
            history["classification_loss"].append(avg_classification_loss)
            history["train_acc"].append(train_acc)
            
            # 评估
            test_loss, test_acc = self.evaluate(test_texts, test_labels_split)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)
            
            print(f"训练损失={avg_loss:.4f}, 对比损失={avg_contrastive_loss:.4f}, "
                  f"分类损失={avg_classification_loss:.4f}, 训练准确率={train_acc:.4f}, "
                  f"测试损失={test_loss:.4f}, 测试准确率={test_acc:.4f}")

        return history, train_texts, test_texts, train_labels_split, test_labels_split

    def forward(self, texts):
        """前向传播函数"""
        sentence_vectors = self._generate_sentence_vectors_with_attention(texts)
        sentence_tensors = torch.stack(sentence_vectors)  # [batch_size, hidden_dim]
        predictions = self._classify_with_residual(sentence_tensors).squeeze(-1)  # [batch_size]
        return predictions

    def evaluate(self, texts, labels):
        """评估模型性能"""
        self.eval()
        criterion = nn.BCELoss()
        labels_tensor = torch.tensor([1.0 if label == "spam" else 0.0 
                                    for label in labels], 
                                    dtype=torch.float32, device=self.device)
        with torch.no_grad():
            predictions = self(texts)
            loss = criterion(predictions, labels_tensor)
            predictions_binary = (predictions >= 0.5).float()
            accuracy = (predictions_binary == labels_tensor).sum().item() / len(texts)
        return loss.item(), accuracy

    def discriminate(self, texts):
        """判别文本"""
        self.eval()
        with torch.no_grad():
            probabilities = self(texts).cpu().numpy()
            predictions = (probabilities >= 0.5).astype(int)
        print(f"判别结果: {predictions}")
        print(f"判别概率: {probabilities}")
        return predictions, probabilities

    def get_embeddings(self, texts):
        """获取文本嵌入表示"""
        self.eval()
        with torch.no_grad():
            sentence_vectors = self._generate_sentence_vectors_with_attention(texts)
            embeddings = torch.stack(sentence_vectors)
            return embeddings.cpu().numpy()

    # 保留原有的其他方法...
    def _clean_texts(self, texts):
        """清洗文本"""
        cleaned_texts = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
            cleaned_texts.append(clean.strip())
        return cleaned_texts
    
    def _tokenize_and_remove_stopwords(self, texts, stopwords_file=None):
        """分词并移除停用词"""
        stopwords = set()
        if stopwords_file:
            try:
                with open(stopwords_file, 'r', encoding='utf-8') as file:
                    stopwords = {line.strip() for line in file}
            except FileNotFoundError:
                print(f"警告: 停用词文件 {stopwords_file} 未找到。不使用停用词过滤。")
        
        tokenized_texts = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            cleaned_text = ''.join([char for char in text 
                                   if char not in stopwords and re.search("[\u4e00-\u9fa5a-zA-Z0-9]", char)])
            tokenized_texts.append(cleaned_text)
            
        return tokenized_texts
    
    def _update_w2v_vectors(self, w2v_vectors, texts, character, d=100):
        """更新词向量"""
        if not texts:
            return w2v_vectors
            
        try:
            tokenized_texts = [[char for char in text if char.strip()] for text in texts]
            tokenized_texts = [text for text in tokenized_texts if text]
            
            if not tokenized_texts:
                return w2v_vectors
                
            model = Word2Vec(sentences=tokenized_texts, vector_size=d, window=5, min_count=1, sg=1)
            word_vectors = model.wv
            if character in word_vectors:
                w2v_vectors[character] = torch.tensor(
                    word_vectors[character], 
                    dtype=torch.float32, 
                    device=self.device
                )
        except Exception as e:
            print(f"更新词向量时出错: {e}")
            
        return w2v_vectors
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'dynamic_threshold': self.dynamic_threshold,
            'temperature': self.temperature,
            'contrastive_weight': self.contrastive_weight,
            'projection_dim': self.projection_dim,
            'char_vectors': self.char_vectors,
            'w2v_vectors': self.w2v_vectors
        }, filepath)
        print(f"模型已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath, device=None):
        """加载模型"""
        device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            dynamic_threshold=checkpoint['dynamic_threshold'],
            temperature=checkpoint.get('temperature', 0.07),
            contrastive_weight=checkpoint.get('contrastive_weight', 0.5),
            projection_dim=checkpoint.get('projection_dim', 64),
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.char_vectors = checkpoint['char_vectors']
        model.w2v_vectors = checkpoint['w2v_vectors']
        model.to(device)
        
        return model


class PositionalEncoding(nn.Module):
    """动态生成位置编码，支持任意长度序列"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # 计算频率调节项并注册为缓冲区（自动设备同步）
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            (-math.log(10000.0) / d_model)
        )
        self.register_buffer('div_term', div_term)  # 关键修复
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 动态生成位置编码（全部在输入x的设备上）
        position = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * self.div_term)  # div_term已在正确设备
        pe[:, 1::2] = torch.cos(position * self.div_term)
        
        return x + pe.unsqueeze(0)


class L2Norm(nn.Module):
    """L2归一化层"""
    def __init__(self, dim=-1):
        super(L2Norm, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
