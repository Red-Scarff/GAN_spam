import torch
import json
import os
from datetime import datetime
import requests
import time
from typing import List, Tuple, Dict
import random
from openai import OpenAI
import time


class DataAugmentation:
    """错误预测分析和数据生成工具"""
    
    def __init__(self, model, api_config=None):
        """
        初始化工具
        
        Args:
            model: 训练好的ContrastiveSpamDiscriminator模型
            api_config: API配置字典，包含API密钥、URL等信息
        """
        self.model = model
        self.api_config = api_config or {}
        self.error_samples = []
        
    def analyze_predictions(self, test_texts: List[str], test_labels: List[str], 
                          output_file: str = None) -> Tuple[List[str], List[str], List[float]]:
        """
        分析测试集中的预测错误
        
        Args:
            test_texts: 测试文本列表
            test_labels: 测试标签列表 
            output_file: 输出文件路径
            
        Returns:
            错误文本、错误标签、预测概率的元组
        """
        print("开始分析预测错误...")

        ## discriminate需新增返回文本
        predictions, probabilities = self.model.discriminate(test_texts)
        
        # 转换标签格式
        test_labels_binary = [1 if label == "spam" else 0 for label in test_labels]
        
        # 找出预测错误的样本
        error_texts = []
        error_true_labels = []
        error_pred_labels = []
        error_probabilities = []
        
        for i, (text, true_label, pred_label, prob) in enumerate(
            zip(test_texts, test_labels_binary, predictions, probabilities)
        ):

            if true_label != pred_label:
                error_texts.append(text)
                error_true_labels.append("spam" if true_label == 1 else "normal")
                error_pred_labels.append("spam" if pred_label == 1 else "normal")
                error_probabilities.append(prob)
        
        print(f"找到 {len(error_texts)} 个预测错误的样本")
        
        # 保存错误样本信息
        self.error_samples = [{
            'text': text,
            'true_label': true_label,
            'pred_label': pred_label,
            'probability': prob
        } for text, true_label, pred_label, prob in zip(
            error_texts, error_true_labels, error_pred_labels, error_probabilities
        )]
        
        # 保存到文件
        if output_file:
            self.save_error_samples(output_file)
        
        return error_texts, error_true_labels, error_probabilities
    
    def save_error_samples(self, filepath: str):
        """保存错误样本到文件"""

        # 保存为txt格式（可读性好）
        txt_file = filepath.replace('.txt', f'_errors.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"模型预测错误分析报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总错误样本数: {len(self.error_samples)}\n")
            f.write("="*80 + "\n\n")
            
            for i, sample in enumerate(self.error_samples, 1):
                f.write(f"错误样本 {i}:\n")
                f.write(f"文本: {sample['text']}\n")
                f.write(f"真实标签: {sample['true_label']}\n")
                f.write(f"预测标签: {sample['pred_label']}\n")
                f.write(f"预测概率: {sample['probability']:.4f}\n")
                f.write("-" * 50 + "\n")
        
    
    def setup_api_config(self, api_key: str, base_url: str = None, model :str = None):
        """
        设置API配置
        
        Args:
            api_key: API密钥
            base_url: API基础URL（可选）
        """
        self.api_config = {
            'api_key': api_key,
            'base_url': base_url,
            'model': model
        }
        
    def generate_similar_data_with_api(self, error_samples: List[Dict] = None, 
                                     num_generate: int = 10, num_to_input: int = 50) -> List[Dict]:
        """
        使用大模型API生成类似的数据
        
        Args:
            error_samples: 错误样本列表，如果为None则使用已分析的错误样本
            num_generate: 要生成的样本数量
            model_name: 模型名称
            
        Returns:
            生成的数据列表
        """
        if not self.api_config:
            raise ValueError("请先设置API配置")
        
        if error_samples is None:
            error_samples = self.error_samples
        
        if not error_samples:
            raise ValueError("没有可用的错误样本")
        
        print(f"开始生成 {num_generate} 个类似数据...")
        
        # 按标签分组错误样本
        spam_errors = [s for s in error_samples if s['true_label'] == 'spam']
        normal_errors = [s for s in error_samples if s['true_label'] == 'normal']
        
        generated_data = []
        
        for i in range(2): 
            # 生成垃圾文本样本
            if spam_errors:
                # 随机选择spam_errors中的一半元素
                # 确保选择的数量不超过spam_errors的实际数量
                num_spam_to_select = max(1, min(len(spam_errors) // 2, num_to_input))
                selected_spam_samples = random.sample(spam_errors, num_spam_to_select)
                
                spam_samples = self._generate_samples_for_category(
                    selected_spam_samples, 'spam', num_generate // 2, num_spam_to_select
                )
                generated_data.extend(spam_samples)
            
            # 生成正常文本样本
            if normal_errors:
                # 随机选择normal_errors中的一半元素
                # 确保选择的数量不超过normal_errors的实际数量
                num_normal_to_select = max(1, min(len(normal_errors) // 2, num_to_input))
                selected_normal_samples = random.sample(normal_errors, num_normal_to_select)

                normal_samples = self._generate_samples_for_category(
                    selected_normal_samples, 'normal', num_generate//2, num_normal_to_select
                )
                generated_data.extend(normal_samples)
        
        return generated_data
    
    def _generate_samples_for_category(self, category_errors: List[Dict], 
                                     category: str, num_to_gen: int, num_to_input: int
                                     ) -> List[Dict]:
        """为特定类别生成样本"""
        if num_to_gen <= 0:
            return []
        
        #选择num_to_input个错误样本
        example_texts = [sample['text'] for sample in 
                        random.sample(category_errors, min(num_to_input, len(category_errors)))]
        
        with open(f"./data_gen/error_samples_{category}.txt", "w",encoding='utf-8') as f:
            f.write(f"判别器分类错误样本 {category}:\n")
            for i, text in enumerate(example_texts, 1):
                f.write(f"{i}. {text}\n")

        # 构建提示词
        prompt = self._build_generation_prompt(example_texts, category, num_to_gen)
        
        # 调用API生成数据
        try:
            generated_texts = self._call_llm_api(prompt,category)
            

            
            return generated_texts[:num_to_gen]  # 确保不超过请求数量
            
        except Exception as e:
            print(f"生成{category}类别数据时出错: {e}")
            return []
    
    def _build_generation_prompt(self, example_texts: List[str], 
                               category: str, num_to_gen: int) -> str:
        """构建用于生成数据的提示词"""
        category_desc = "垃圾信息" if category == "spam" else "正常信息"
        
        prompt = f"""请根据以下{category_desc}示例，生成{num_to_gen}条类似的中文文本。

        示例{category_desc}:"""
        for i, text in enumerate(example_texts, 1):
            prompt += f"{i}. {text}\n"
        
        prompt += f"""
        要求:
        1. 生成的文本应该与示例具有相似的特征和风格
        2. 生成的文本与示例可以语义上相近，但是不可以直接摘抄、复制原本的示例
        3. 你必须要结合汉字的字音、字形的相似性，替换文本中的大部分词。
        下面是几个典型示例，你需要结合具体文本做出灵活多样的替换：“微信”可以替换成“威信”、“违心”、“胃星”，也可以
        简化成“佳V”、“加魏”等
        “qq”可以替换成“扣扣”、“蔻蔻"、“企鹅”等
        4. 文本长度与示例类似
        5.在生成垃圾文本时，为了伪装垃圾信息，即使文本不自然也可以。但是生成正常文本时，语言要自然，可以包含错别字。
        6. 生成的文本应该是{category_desc}类别，如果你觉得输入示例中有些文本不属于这个类别，可以忽略他。
        7. 请直接输出{num_to_gen}条文本，每条文本只能使用一行，不要分行输出，文本前不要添加序号，不要使用其他格式
        生成的{num_to_gen}条{category_desc}:"""
        
        return prompt
    
    def _call_llm_api(self, prompt: str, category: str) -> List[Dict]:
        """调用大模型API生成文本并返回结构化数据"""
        try:
            # 初始化OpenAI客户端
            client = OpenAI(
                api_key=self.api_config['api_key'],
                base_url=self.api_config['base_url']
            )

            # 调用DeepSeek API
            response = client.chat.completions.create(
                model=self.api_config['model'],
                messages=[
                    {"role": "system", "content": "你是一个生成中文文本的助手，确保生成的内容符合要求。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=1.5  # 设置温度
            )

            # 获取生成的内容
            generated_text = response.choices[0].message.content.strip()
            
            # 清理特殊字符
            generated_text = ''.join(c for c in generated_text if ord(c) < 0x10000)  # 移除高位Unicode字符

            # 清除think部分
            # 查找 </think> 标识符的起始索引
            end_think_index = generated_text.find("</think>")

            if end_think_index != -1: # 如果找到了
                # 从 </think> 后面开始截取，加上 len("</think>") 跳过标识符本身
                start_content_index = end_think_index + len("</think>")
                generated_text = generated_text[start_content_index:].strip()
            else:
                # 如果没有找到 </think>
                generated_text = generated_text.strip()


            # 按行分割生成的内容，并为每条文本分配标签
            generated_lines = generated_text.split('\n')
            generated_lines = [line.strip() for line in generated_lines if line.strip()]

            # 假设生成的内容为指定类别的文本，标签从提示词的category推导
            
            generated_data = [{'text': text, 'label': category} for text in generated_lines]

            return generated_data

        except Exception as e:
            print(f"调用API时出错: {e}")
            return []

    
    
    def save_generated_data(self, generated_data: List[Dict], filepath: str):
        """保存生成的数据"""
        
        # 保存为txt格式
        txt_file = filepath.replace('.txt', f'_generated.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            
            for i, sample in enumerate(generated_data, 1):
                if(sample['label']=='spam'):
                    binary_label = '1'
                else:
                    binary_label = '0'
               
                f.write(f"{binary_label} {sample['text']}\n")
        
        # # 保存为JSON格式
        # json_file = filepath.replace('.txt', f'_generated.json')
        # with open(json_file, 'w', encoding='utf-8') as f:
        #     json.dump({
        #         'total_generated': len(generated_data),
        #         'generated_data': generated_data
        #     }, f, ensure_ascii=False, indent=2)
        
        # print(f"生成数据已保存到:")
        # print(f"  TXT格式: {txt_file}")
        # print(f"  JSON格式: {json_file}")
    
    def run_complete_pipeline(self, test_texts: List[str], test_labels: List[str],
                            output_file: str = "./data_gen/analysis_results.txt",
                            num_generate: int = 20, num_to_input: int = 50):
        """运行完整的分析和生成流程"""
        print("="*60)
        print("开始错误分析和数据生成流程")
        print("="*60)
        
        # 添加时间戳，保留数据
        timestamp = int(time.time())  # 取整
        temp_out_file = output_file
        output_file = output_file.replace('.txt', f'_{timestamp}.txt')

        # 1. 分析预测错误
        error_texts, error_labels, error_probs = self.analyze_predictions(
            test_texts, test_labels, output_file
        )
        
        if not error_texts:
            print("没有预测错误的样本，流程结束。")
            return
        
        # 2. 生成类似数据
        if self.api_config:
            try:
                generated_data = self.generate_similar_data_with_api(
                    num_generate=num_generate,
                    num_to_input= num_to_input
                )
                
                if generated_data:
                    self.save_generated_data(generated_data, output_file)
                    print(f"成功生成 {len(generated_data)} 条新数据")
                    self.save_generated_data(generated_data, temp_out_file)
                else:
                    print("未能生成新数据")
            except Exception as e:
                print(f"生成数据时出错: {e}")
        else:
            print("未配置API，跳过数据生成步骤")
            
        print("="*60)
        print("流程完成")
        print("="*60)



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



# if __name__ == "__main__":

#     model = SpamDiscriminator.load('spam_discriminator_model.pth')
#     test_labels, test_texts = read_data('./数据集/dataset.txt')
    
#     # 创建分析工具
#     analyzer = DataAugmentation(model)
    
#     # 运行完整流程
#     analyzer.run_complete_pipeline(
#         test_texts=test_texts,
#         test_labels=test_labels,
#         output_file="spam_analysis_results.txt",
#         num_generate=30, #  生成样本数量
#         num_to_input=50  # 输入给大模型的样本数量
#     )