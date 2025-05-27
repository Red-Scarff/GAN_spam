import torch
import numpy as np
from transformers import BertTokenizer
from spam_discriminator.ssc_similarity import computeSSCSimilarity
from tqdm import tqdm

def is_chinese_char(tok: str):
    return len(tok) == 1 and '\u4e00' <= tok <= '\u9fff'

def build_and_save(model_name='bert-base-chinese',
                   output_path='../models/sim_matrix.pt'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    vocab = tokenizer.vocab
    id2token = {i: t for t, i in vocab.items()}
    V = len(vocab)

    sim = np.zeros((V, V), dtype=np.float32)
    for i in tqdm(range(V), desc="Building similarity matrix"):
        ti = id2token[i]
        if not is_chinese_char(ti):
            continue
        for j in range(i, V):
            tj = id2token[j]
            if not is_chinese_char(tj):
                continue
            s = computeSSCSimilarity(ti, tj)
            sim[i, j] = sim[j, i] = s

    # 转成 torch tensor 并保存
    sim_tensor = torch.from_numpy(sim)
    torch.save(sim_tensor, output_path)
    print(f"Saved similarity matrix to {output_path}, shape = {sim_tensor.shape}")

if __name__ == '__main__':
    build_and_save()
