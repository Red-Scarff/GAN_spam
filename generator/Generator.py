import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import random
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import build_sim_matrix

class Generator(nn.Module):
    def __init__(self, 
                 hidden_dim: int, 
                 computeSSCSimilarity,
                 lambda_sim: float = 1.0,
                 bert_model_name: str = 'bert-base-chinese'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.vocab_size = self.bert.config.vocab_size
        self.hidden_dim = hidden_dim    
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.computeSSCSimilarity = computeSSCSimilarity
        self.lambda_sim = lambda_sim

        self.mask_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.replace_head = nn.Linear(self.bert.config.hidden_size, self.vocab_size)

        # 构建 index 到 token 的映射表
        self.id2token = {i: tok for tok, i in self.tokenizer.vocab.items()}
        self.token2id = self.tokenizer.vocab
        
        print("Computing Generator Similarity_matrix...")
        self.sim_matrix = self._build_similarity_matrix()
        print("Similarity_matrix Computing Finished")
        
    def _build_similarity_matrix(self):
        sim_matrix = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float32)
        for i in range(self.vocab_size):
            tok_i = self.id2token[i]
            if len(tok_i) != 1 or not self._is_chinese_char(tok_i):
                continue
            for j in range(i, self.vocab_size):
                tok_j = self.id2token[j]
                if len(tok_j) != 1 or not self._is_chinese_char(tok_j):
                    continue
                sim_matrix[i][j] = sim_matrix[j][i] = self.computeSSCSimilarity(tok_i, tok_j)
        return torch.tensor(sim_matrix, dtype=torch.float32)


    def forward(self, input_ids, attention_mask, discriminator):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state  # [B, T, H]

        mask_logits = self.mask_head(seq_out).squeeze(-1)  # [B, T]
        mask_probs = torch.sigmoid(mask_logits)           # [B, T]

        rep_logits = self.replace_head(seq_out)           # [B, T, V]
        rep_probs = F.gumbel_softmax(rep_logits, tau=0.5, hard=True)  # [B, T, V]

        # === 计算相似度损失 ===
        print("=== Generator 计算相似度损失 ===")
        B, T = input_ids.shape
        sim_loss = 0.0
        for b in range(B):
            for t in range(T):
                orig_id = input_ids[b, t].item()
                orig_token = self.id2token.get(orig_id, '[UNK]')
                if len(orig_token) != 1 or not self._is_chinese_char(orig_token):
                    continue 
                sim_vec = self.sim_matrix[orig_id].to(rep_probs.device)  # [V]
                sim_loss += ((1.0 - sim_vec) * rep_probs[b, t]).sum()
                
        sim_loss = sim_loss / (B * T)

        one_hot_orig = F.one_hot(input_ids, num_classes=self.vocab_size).float()
        mask_probs_exp = mask_probs.unsqueeze(-1)
        token_probs = (1 - mask_probs_exp) * one_hot_orig + mask_probs_exp * rep_probs
        gen_ids = token_probs.argmax(dim=-1)
        
        generated_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        
        print("Generator 生成数据交由判别器评估")
        
        _, prob_spam = discriminator.discriminate(generated_texts)  # returns (preds, probs)
        
        prob_normal = 1 - prob_spam
    
        prob_normal = torch.tensor(prob_normal, dtype=torch.float32, device=input_ids.device)  # [B]
        
        dis_loss = -torch.log(torch.clamp(prob_normal, min=1e-8)).mean()
        total_loss = dis_loss + self.lambda_sim * sim_loss
        
        print(f"Generator total loss:{total_loss} dis_loss: {dis_loss} sim_loss: {sim_loss}")

        return {
            "loss": total_loss,
            "dis_loss": dis_loss,
            "sim_loss": sim_loss,
            "gen_ids": gen_ids
        }

    def train_step(self, batch_texts, discriminator, optimizer, device):
        encodings = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        print("Generator train_step")
        self.train()
        optimizer.zero_grad()
        result = self.forward(input_ids, attention_mask, discriminator)
        result['loss'].backward()
        optimizer.step()
        return result

    def generate(self, text, device='cuda' if torch.cuda.is_available() else 'cpu') -> str:
        """
        input:
            正常文本
        
        output:
            替换后文本
        """
        self.eval()
        with torch.no_grad():
            encoding = self.tokenizer(text, return_tensors='pt')
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            seq_out = outputs.last_hidden_state
            mask_logits = self.mask_head(seq_out).squeeze(-1)
            mask_probs = torch.sigmoid(mask_logits)
            rep_logits = self.replace_head(seq_out)
            # rep_probs = F.softmax(rep_logits, dim=-1)
            rep_probs = F.gumbel_softmax(rep_logits, tau=0.5, hard=False) 

            gen_ids = []
            for t in range(input_ids.size(1)):
                orig_id = input_ids[0, t].item()
                orig_token = self.id2token.get(orig_id, '[UNK]')
                if len(orig_token) != 1 or not self._is_chinese_char(orig_token):
                    gen_ids.append(orig_id)
                    continue
                p_mask = mask_probs[0, t].item()
                if random.random() < p_mask:
                    sim_vec = self.sim_matrix[orig_id].to(rep_probs.device)  # [V]
                    scores = rep_probs[0, t] * sim_vec
                    best_id = torch.argmax(scores).item()
                    gen_ids.append(best_id)
                
                else:
                    gen_ids.append(orig_id)
            return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def _is_chinese_char(self, token):
        return len(token) == 1 and '\u4e00' <= token <= '\u9fff'

    def train_model(self, train_texts, discriminator, num_epochs=1, batch_size=10, lr=5e-5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        训练过程
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        from torch.utils.data import DataLoader
        dataloader = DataLoader(train_texts, batch_size=batch_size, shuffle=True)
        
        print(f"Generator 开始训练， 使用设备: {device}")

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0.0
            total_dis_loss = 0.0
            total_sim_loss = 0.0
            for batch_texts in tqdm(dataloader, desc=f"Generator Epoch {epoch+1}/{num_epochs}"):
                result = self.train_step(batch_texts, discriminator, optimizer, device)
                total_loss += result['loss'].item()
                total_dis_loss += result['dis_loss'].item()
                total_sim_loss += result['sim_loss'].item()

            print(f"[Epoch {epoch+1}] Total Loss: {total_loss:.4f} | Dis Loss: {total_dis_loss:.4f} | Sim Loss: {total_sim_loss:.4f}")


class ReplacementPolicy(nn.Module):
    def __init__(self, 
                 hidden_dim: int, 
                 computeSSCSimilarity,
                 bert_model_name: str = 'bert-base-chinese',
                 sim_matrix_path: str = 'models/sim_matrix.pt',
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 topk: int = 5,
                 temperature: float = 1.0):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.vocab_size = self.bert.config.vocab_size
        self.hidden_dim = hidden_dim
        self.computeSSCSimilarity = computeSSCSimilarity
        self.topk = topk
        self.temperature = temperature
        self.device = device

        # 冻结 BERT 所有参数
        for param in self.bert.parameters():
            param.requires_grad = False

        # 只训练这个线性层，判断是否替换
        self.mask_head = nn.Linear(self.bert.config.hidden_size, 1)

        # vocab 映射
        self.id2token = {i: tok for tok, i in self.tokenizer.vocab.items()}
        self.token2id = self.tokenizer.vocab

        # 加载或构建相似度矩阵
        if sim_matrix_path:
            print(f"Loading similarity matrix from {sim_matrix_path} ...")
            self.sim_matrix = torch.load(sim_matrix_path, map_location='cpu').to(self.device)
            print("Loaded similarity matrix.")
        else:
            print("Computing Generator Similarity_matrix...")
            self.sim_matrix = build_sim_matrix(model_name = bert_model_name).to(self.device)
            print("Similarity_matrix Computing Finished")

        self._precompute_topk_indices()

    def _precompute_topk_indices(self):
        """预计算每个token的top-k相似度索引"""
        print("Precomputing top-k similarity indices...")
        with tqdm(total=1, desc="Computing top-k indices") as pbar:
            self.topk_indices = torch.topk(self.sim_matrix, k=self.topk, dim=1).indices
            pbar.update(1)
        print("Top-k indices precomputed.")

    def _sample_from_topk(self, similarities, temperature=None):
        """从top-k相似度中按概率采样"""
        if temperature is None:
            temperature = self.temperature
            
        # 应用温度缩放
        scaled_sims = similarities / temperature
        # 使用softmax计算概率
        probs = torch.softmax(scaled_sims, dim=-1)
        # 按概率采样
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def forward(self, input_ids, attention_mask, discriminator):
        with torch.no_grad():  # BERT 不训练
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            seq_out = outputs.last_hidden_state  # [B, T, H]

        mask_logits = self.mask_head(seq_out).squeeze(-1)  # [B, T]
        mask_probs = torch.sigmoid(mask_logits)  # [B, T]

        B, T = input_ids.shape
        gen_ids = input_ids.clone()

        for b in range(B):
            for t in range(T):
                orig_id = input_ids[b, t].item()
                orig_token = self.id2token.get(orig_id, '[UNK]')
                if len(orig_token) != 1 or not self._is_chinese_char(orig_token):
                    continue

                p_mask = mask_probs[b, t].item()
                if p_mask >= 0.5:
                    sim_vec = self.sim_matrix[orig_id].to(input_ids.device)
                    topk_id = torch.argmax(sim_vec).item()
                    gen_ids[b, t] = topk_id

        # 使用 discriminator 判别是否是“正常文本”
        generated_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        _, prob_spam = discriminator.discriminate(generated_texts)
        
        prob_normal = 1 - prob_spam

        prob_normal = torch.tensor(prob_normal, dtype=torch.float32, device=input_ids.device)
        dis_loss = -torch.log(torch.clamp(prob_normal, min=1e-8)).mean()

        return {
            "loss": dis_loss,
            "dis_loss": dis_loss,
            "gen_ids": gen_ids
        }
        
    def train_from_texts(self, 
                     train_texts,
                     discriminator,
                     num_epochs=5, 
                     batch_size=16, 
                     lr=1e-4, 
                     device='cuda'):
        """
        使用原始文本训练 mask_head，决定哪些 token 被替换。
        """
        tokenizer = self.tokenizer
        self.to(device)
        self.train()
        discriminator.eval()

        # 编码文本为 input_ids
        encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        dataset = TensorDataset(input_ids, attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 只训练 mask_head
        optimizer = torch.optim.Adam(self.mask_head.parameters(), lr=lr)

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids_batch, attention_mask_batch = [x.to(device) for x in batch]

                output = self(input_ids_batch, attention_mask_batch, discriminator)
                loss = output['loss']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"[Epoch {epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")
            
    def generate(self, texts, threshold=0.5, device='cuda'):
        """
        根据训练好的 mask_head 和 sim_matrix，将正常文本转换为伪垃圾文本。
        - texts: str 或 List[str]
        - threshold: 替换概率阈值，大于该值的 token 才会被替换
        - return: str 或 List[str]
        """
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True

        self.eval()
        self.to(device)
        tokenizer = self.tokenizer
        encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [B, T, H]
            mask_scores = self.mask_head(hidden_states).squeeze(-1)  # [B, T]
            replace_probs = torch.sigmoid(mask_scores)  # [B, T], 值越大越倾向替换

        result_texts = []
        for b in range(input_ids.size(0)):
            orig_ids = input_ids[b]
            rep_probs = replace_probs[b]
            new_ids = orig_ids.clone()

            for t in range(orig_ids.size(0)):
                if attention_mask[b, t] == 0:
                    continue  # pad token 跳过

                token_id = orig_ids[t].item()
                token_str = self.id2token[token_id]
                if len(token_str) != 1 or not self._is_chinese_char(token_str):
                    continue

                if rep_probs[t].item() > threshold:
                    sim_vector = self.sim_matrix[token_id]  # [V]
                    top_sim_id = torch.argmax(sim_vector).item()
                    if top_sim_id != token_id:
                        new_ids[t] = top_sim_id

            new_text = tokenizer.decode(new_ids, skip_special_tokens=True).replace(' ', '')
            result_texts.append(new_text)

        return result_texts[0] if single_input else result_texts
