from pypinyin import pinyin, Style
from .four_corner_method import FourCornerMethod
from spam_discriminator.ssc_similarity import *
from tqdm import tqdm
import pickle
import numpy as np

class ChineseCharacterCoder:
    def __init__(self):
        # 初始化字典
        self.structure_dict = {}
        self.strokes_dict = {'1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9', '10':'A',
                                '11':'B', '12':'C', '13':'D', '14':'E', '15':'F', '16':'G', '17':'H', '18':'I', '19':'J', '20':'K',
                                '21':'L', '22':'M', '23':'N', '24':'O', '25':'P', '26':'Q', '27':'R', '28':'S', '29':'T', '30':'U',
                                '31':'V', '32':'W', '33':'X', '34':'Y', '35':'Z', '36':'a', '37':'b', '38':'c', '39':'d', '40':'e', 
                                '41':'f', '42':'g', '43':'h', '44':'i', '45':'j', '46':'k', '47':'l', '48':'m', '49':'n', '50':'o',
                                '51':'p'}

        # 加载汉字结构对照文件
        with open('./data/hanzijiegou_2w.txt', 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    structure, chinese_character = parts
                    self.structure_dict[chinese_character] = structure

        # 加载汉字笔画对照文件，参考同级目录下的 chinese_unicode_table.txt 文件格式
        self.chinese_char_map = {}
        with open('./data/chinese_unicode_table.txt', 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines[6:]:  # 前6行是表头，去掉
                line_info = line.strip().split()
                # 处理后的数组第一个是文字，第7个是笔画数量
                self.chinese_char_map[line_info[0]] = self.strokes_dict[line_info[6]]

    def split_pinyin(self, chinese_character):
        # 将汉字转换为拼音（带声调）
        pinyin_result = pinyin(chinese_character, style=Style.TONE3, heteronym=True)

        # 多音字的话，选择第一个拼音
        if pinyin_result:
            py = pinyin_result[0][0]

            initials = ""  # 声母
            finals = ""    # 韵母
            codas = ""     # 补码
            tone = ""      # 声调

            # 声母列表
            initials_list = ["b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w"]

            # 韵母列表
            finals_list = ["a", "o", "e", "i", "u", "v", "ai", "ei", "ui", "ao", "ou", "iu", "ie", "ve", "er", "an", "en", "in", "un", "vn", "ang", "eng", "ing", "ong"]

            # 获取声调
            if py[-1].isdigit():
                tone = py[-1]
                py = py[:-1]

            # 获取声母
            for initial in initials_list:
                if py.startswith(initial):
                    initials = initial
                    py = py[len(initial):]
                    break

            # 获取韵母
            for final in finals_list:
                if py.endswith(final):
                    finals = final
                    py = py[:-len(final)]
                    break

            # 获取补码
            codas = py

            return initials, finals, codas, tone        

        return None

    def generate_pronunciation_code(self, hanzi):
        initial, final, coda, tone = self.split_pinyin(hanzi)

        # 轻声字，例如'了'
        if tone == '':
            tone = '0'

        # 声母映射
        initials_mapping = {'b': '1', 'p': '2', 'm': '3', 'f': '4', 'd': '5', 't': '6', 'n': '7', 'l': '8',
                            'g': '9', 'k': 'a', 'h': 'b', 'j': 'c', 'q': 'd', 'x': 'e', 'zh': 'f', 'ch': 'g',
                            'sh': 'h', 'r': 'i', 'z': 'j', 'c': 'k', 's': 'l', 'y': 'm', 'w': 'n'}

        # 韵母映射
        finals_mapping = {'a': '1', 'o': '2', 'e': '3', 'i': '4', 'u': '5', 'v': '6', 'ai': '7', 'ei': '8',
                            'ui': '9', 'ao': 'a', 'ou': 'b', 'iu': 'c', 'ie': 'd', 've': 'e', 'er': 'f',
                            'an': 'g', 'en': 'h', 'in': 'i', 'un': 'j', 'vn': 'k', 'ang': 'l', 'eng': 'm',
                            'ing': 'n', 'ong': 'o'}

        # 补码映射
        coda_mapping = {'': '0', 'u':'1', 'i':'1'}

        # 获取映射值
        initial_code = initials_mapping.get(initial, '0')
        final_code = finals_mapping.get(final, '0')
        coda_code = coda_mapping.get(coda, '0')

        # 组合生成四位数的字音编码
        pronunciation_code = initial_code + final_code + coda_code + tone

        return pronunciation_code

    def generate_glyph_code(self, hanzi):
        if hanzi not in self.structure_dict: 
            structure_code = '0'
        else:# 获取汉字的结构
            structure_code = self.structure_dict[hanzi]

        # 获取汉字的四角编码
        fcc = FourCornerMethod().query(hanzi)

        # 获取汉字的笔画数
        stroke = self.chinese_char_map[hanzi]

        # 组合生成的字形编码
        glyph_code = structure_code + fcc + stroke

        return glyph_code
    
    def generate_character_code(self, hanzi):
        return self.generate_pronunciation_code(hanzi) + self.generate_glyph_code(hanzi)

# 统计data中的所有汉字以及对应的出现次数，并对其进行编码
def count_chinese_characters(content, output_file_path):
    chinese_characters = []
    chinese_characters_count = {}
    chinese_characters_code = {}

    for line in tqdm(content, desc='Counting characters', unit='line'):
        for char in line:
            if '\u4e00' <= char <= '\u9fff':  # 判断是否为汉字
                chinese_characters_count[char] = chinese_characters_count.get(char, 0) + 1

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for char, count in tqdm(chinese_characters_count.items(), desc='Computing Character Code', unit='char'):
            character_code = ChineseCharacterCoder().generate_character_code(char)
            chinese_characters_code[char] = character_code
            output_file.write(f'{char}\t{character_code}\t{count}\n')
            chinese_characters.append(char)

    print(f'Results saved to {output_file_path}')

    return chinese_characters, chinese_characters_count, chinese_characters_code

# 加载已有的汉字库
def load_chinese_characters(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        line = f.readlines()
    chinese_characters = []
    chinese_characters_count = {}
    chinese_characters_code = {}
    
    for row in line:
        char, code, count = row.strip().split('\t')
        chinese_characters.append(char)
        chinese_characters_code[char] = code
        chinese_characters_count[char] = count

    return chinese_characters, chinese_characters_count, chinese_characters_code

# 构建字符相似性网络（用矩阵形式表示）
def compute_sim_mat(chinese_characters, chinese_characters_code):
    sim_mat = [[0] * len(chinese_characters) for _ in range(len(chinese_characters))]
    for i in tqdm(range(len(chinese_characters)), desc='Constructing Similarity Matrix', unit='i'):
        for j in range(i, len(chinese_characters)):
            similarity = computeSSCSimilarity(chinese_characters_code[chinese_characters[i]], chinese_characters_code[chinese_characters[j]])
            sim_mat[i][j] = similarity
            sim_mat[j][i] = similarity

    # 将结果保存到pkl文件
    output_file = './data/similarity_matrix.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(sim_mat, f)

    return sim_mat

# 从pkl文件中加载相似性矩阵
def load_sim_mat(filename):
    with open(filename, 'rb') as f:
        sim_mat = pickle.load(f)
    
    return sim_mat

# 更新相似性矩阵
def update_sim_mat(new_characters, chinese_characters_code, sim_mat):
    for char in new_characters:
        # 计算新汉字与现有汉字之间的相似性
        new_code = chinese_characters_code[char]
        similarities = [computeSSCSimilarity(new_code, code) for code in chinese_characters_code.values()]
        
        # 更新相似性矩阵
        new_row = np.array(similarities)
        sim_mat = np.vstack([sim_mat, new_row])
        sim_mat = np.hstack([sim_mat, new_row.reshape(-1, 1)])
    
    return sim_mat

# 构建字符相似性网络（用矩阵形式表示）
#def compute_sim_mat(chinese_characters, chinese_characters_code):
    sim_mat = [[0] * len(chinese_characters) for _ in range(len(chinese_characters))]
    for i in tqdm(range(len(chinese_characters)), desc='Constructing Similarity Matrix', unit='i'):
        for j in range(i, len(chinese_characters)):
            similarity = computeSSCSimilarity(chinese_characters_code[chinese_characters[i]], chinese_characters_code[chinese_characters[j]])
            sim_mat[i][j] = similarity
            sim_mat[j][i] = similarity

    # 将结果写入文件
    output_file = '第四章/高阶示例/data/similarity_matrix.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in sim_mat:
            f.write('\t'.join(map(str, row)) + '\n')

    return sim_mat

# 示例使用
#coder = ChineseCharacterCoder()
#pronunciation_code = coder.generate_pronunciation_code('晶')
#print(pronunciation_code)
#glyph_code = coder.generate_glyph_code('晶')
#print(glyph_code)
#print(coder.generate_character_code('金'))