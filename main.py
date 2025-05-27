import torch
from generator.Generator import Generator, ReplacementPolicy
from spam_discriminator.ssc_similarity import computeSSCSimilarity
from spam_discriminator.utils import load_chinese_characters, count_chinese_characters, compute_sim_mat
from spam_discriminator.spam_discriminator import SpamDiscriminator
from sklearn.model_selection import train_test_split

def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text_data = f.readlines()

    dataset = [s.strip().split('\t', 1) for s in text_data]
    dataset = [data for data in dataset if len(data) == 2 and data[1].strip()]

    tag = [data[0] for data in dataset]
    text = [data[1] for data in dataset]


    return tag, text

def read_characters_all(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text_data = f.readlines()
    text = [line.strip().split()[0] for line in text_data[6:]]
    print(len(text))
    return [text]

def Similarity(a, b):
    return computeSSCSimilarity(chinese_characters_code[a], chinese_characters_code[b])
        
dataset_path = './data/dataset.txt'
print("正在读取数据...")
tags, texts= read_data(dataset_path)
print(f"读取了 {len(texts)} 条数据")

# 划分90%数据
train_tags, test_tags, train_texts, test_texts = train_test_split(
        tags, texts, test_size=0.1, random_state=42
    )

# 取出正常数据
pos_train_texts = [train_texts[i] for i in range(len(train_texts)) if train_tags[i] == '0']
neg_train_texts = [train_texts[i] for i in range(len(train_texts)) if train_tags[i] == '1']


print("正在处理汉字数据...")
_, __, chinese_characters_code = load_chinese_characters("./data/hanzi.txt")
#chinese_characters, chinese_characters_count, chinese_characters_code = count_chinese_characters(read_characters_all("./data/chinese_unicode_table.txt"), "./data/hanzi.txt")

# if 训练判别器
# chinese_characters, chinese_characters_count, ____ = count_chinese_characters(texts, "./data/hanzi_dataset.txt")
# print(f"加载了 {len(chinese_characters)} 个汉字")

# print("正在计算汉字相似度矩阵...")
# sim_mat = compute_sim_mat(chinese_characters, chinese_characters_code)




def main():
    
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化判别器
    # discriminator = SpamDiscriminator(embedding_dim=100, hidden_dim=128, device=device)
    discriminator = SpamDiscriminator.load("./spam_discriminator/spam_discriminator_model.pth", device)
    
    # 初始化 Generator
    G = Generator(
        hidden_dim=768,
        computeSSCSimilarity=Similarity,
        lambda_sim=1.0
    ).to(device)
    
    # 读取Generator
    # generator.load_state_dict(torch.load('./generator/generator_model.pt'))
    # generator.to(device)
    
    # 训练判别器
    # print("开始训练判别器...")
    # history, train_texts, test_texts, train_labels, test_labels = discriminator.fit(
    #     texts=texts,  # 传入所有文本
    #     labels=tags,  # 传入所有标签
    #     chinese_characters=chinese_characters,
    #     chinese_characters_count=chinese_characters_count,
    #     sim_mat=sim_mat,
    #     test_size=0.2,  # 设置测试集比例
    #     random_state=42,
    #     batch_size=32,
    #     epochs=1
    # )
    
    # test_loss, test_acc = discriminator.evaluate(texts, tags)
    # print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")
    # # 评估判别器性能
    # print("\n判别器评估:")
    # test_loss, test_acc = discriminator.evaluate(test_texts, test_tags)
    # print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")
    
    
    # sen = "我是清华代培的学生。刚进校的时候我也对清华抱有很高的期望，当时确实觉得很多事情让我失望。然而在这里两年多了，我觉得清华的所有人没有把我们当外人，我在各种活动、学业中找到了自己，并且还在坚持自己的梦想。我和我的同学们都感激清华。积极适应吧，我觉得清华还是给人提供了很大的空间的，关键是自己要学会利用。 赞一下 如果是真的,可见原作者是一个有理想有想法的人"
    
    gen = []
    for t in range(0, 100):
        now = G.generate(pos_train_texts[t])
        gen.append(now)
        
    print(gen[:10], discriminator.discriminate(gen))
    
    G.train_model(neg_train_texts, discriminator)
    
    gen = []
    for t in range(0, 100):
        now = G.generate(pos_train_texts[t])
        gen.append(now)
    print(gen[:10], discriminator.discriminate(gen))
    
    # gen = []
    # tag_gen = []

    # for t in range(100, len(pos_train_texts), 100):
    #     now = G.generate(pos_train_texts[t - 100:t])
    #     a, b = discriminator.discriminate(now)
    #     for i in range(len(a)):
    #         if a[i] == 0:
    #             gen.append(now[i])
    #             tag_gen.append("1")
        
    # print(len(gen), len(tag_gen))
    
    
    # print(generator.generate(sen, device).replace(" ", ""))
    
    # # 训练生成器
    # generator.train_model(pos_train_texts, discriminator)
    
    # print(generator.generate(sen).replace(" ", ""))
    
    # # 保存 Generator
    # torch.save(generator.state_dict(), './generator/generator_model_2.pt')
    
if __name__ == "__main__":
    main()