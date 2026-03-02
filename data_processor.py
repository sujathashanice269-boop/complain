"""
数据处理器 - 完整改进版
✅ 改进: 彻底删除括号、系统字段,保留专业词(4G/5G/VoLTE)
✅ 适用: 预训练、微调、大数据集、小数据集
✅ 修复: 添加了load_data, prepare_datasets, ComplaintDataset, custom_collate_fn
"""

import pandas as pd
import numpy as np
import jieba
import re
import torch
import pickle
import os
from typing import List, Dict, Tuple
from collections import Counter
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random

class ComplaintDataProcessor:
    """投诉数据处理器 - 带智能清洗功能"""

    def __init__(self, config, user_dict_file='new_user_dict.txt', stopword_file='new_stopword_dict.txt'):
        """
        初始化数据处理器

        Args:
            config: 配置对象
            user_dict_file: 用户词典文件路径（仅用于文本）
            stopword_file: 停用词文件路径（仅用于文本）
        """
        self.config = config

        # ========== 1️⃣ 首先初始化tokenizer（最重要！）==========
        self.tokenizer = BertTokenizer.from_pretrained(config.model.bert_model_name)

        # ========== 2️⃣ 加载用户词典（仅用于文本）==========
        self.user_dict_whitelist = set()

        if user_dict_file and os.path.exists(user_dict_file):
            # 加载到jieba
            jieba.load_userdict(user_dict_file)

            # 同时保存到白名单
            with open(user_dict_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        word = parts[0]
                        self.user_dict_whitelist.add(word)

            print(f"✅ 已加载用户词典: {user_dict_file} (仅用于文本)")
            print(f"✅ 用户词典白名单: {len(self.user_dict_whitelist)}个词 (用于智能清洗)")

        # ========== 3️⃣ 加载停用词（仅用于文本）==========
        self.stopwords = set()
        if stopword_file and os.path.exists(stopword_file):
            with open(stopword_file, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f if line.strip())
            print(f"✅ 已加载停用词: {stopword_file} ({len(self.stopwords)}个, 仅用于文本)")

        # ========== 4️⃣ 文本清洗缓存 ==========
        self.text_clean_cache = {}

        # ========== 5️⃣ 标签相关属性 ==========
        self.node_to_id = {}
        self.id_to_node = {}
        self.global_edges = []
        self.global_node_levels = {}

        # ========== 6️⃣ 结构化特征标准化器 ==========
        self.scaler = StandardScaler()

    def clean_text_smart(self, text: str) -> str:
        """
        智能清洗文本 - 改进版

        改进点:
        1. ⭐ 先删除系统噪音(括号、系统字段等)
        2. ✅ 保留所有中文字符
        3. ✅ 保留用户词典中的词(4G、5G、VoLTE等)
        4. ✅ 保留正常标点符号
        5. ❌ 删除其他所有内容

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        # ✅ 新增: 检查缓存，避免重复清洗
        if text in self.text_clean_cache:
            return self.text_clean_cache[text]

        if pd.isna(text) or not text:
            return ""

        text = str(text).strip()

        if not text:
            return ""

        # ========== 第一阶段: 删除系统噪音 ==========

        # 1. 删除#标记#
        text = re.sub(r'#[^#]*#', '', text)

        # 2. 删除[数字]标记和[begin][end]
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[/\d+\]', '', text)
        text = re.sub(r'\[begin\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[end\]', '', text, flags=re.IGNORECASE)

        # 3. ⭐ 核心改进: 删除所有带数字的括号
        text = re.sub(r'\(\d+\)【[^】]*】', '', text)  # (1)【自动】
        text = re.sub(r'【[^】]*】\(\d+\)', '', text)  # 【人工】(1)
        text = re.sub(r'\(\d+\)', '', text)            # (0) (1) (123)
        text = re.sub(r'【\d+】', '', text)            # 【1】【2】
        text = re.sub(r'【[^】]*】', '', text)          # 【人工】【自动】

        # 4. 删除剩余的空括号
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'【\s*】', '', text)

        # 5. ⭐ 删除系统字段前缀
        system_fields = [
            'te_system_log:', 'con_status:', 'pending',
            'te_', 'con_', 'ho_', 'sys_',
            '流程轨迹', '处理内容', '投诉内容', '活动信息',
            '受理号码', '取消原因', '联系号码', '详情描述',
            '联系要求', '上网账号', '联系电话', '申告号码',
            '宽带地址', '宽带账号', '工单号', '客户要求'
        ]
        for field in system_fields:
            text = re.sub(f'{field}[:：；]?', '', text, flags=re.IGNORECASE)

        # 6. 删除隐私脱敏符号
        text = re.sub(r'\*{3,}', '', text)  # ***
        text = re.sub(r'#{3,}', '', text)   # ###
        text = re.sub(r'WO\d+', '', text)   # 工单号

        # ========== 第二阶段: 保留有用内容 ==========

        # 定义保留的标点符号
        keep_punctuation = set('，。！？、；：""''（）《》')

        # 使用jieba分词(会自动识别用户词典中的词)
        words = jieba.lcut(text)

        cleaned_words = []

        for word in words:
            # 规则1: 用户词典中的词 → 直接保留(4G、5G、VoLTE等)
            if word in self.user_dict_whitelist:
                cleaned_words.append(word)
                continue

            # 规则2: 纯中文词 → 保留
            if all('\u4e00' <= c <= '\u9fff' for c in word):
                cleaned_words.append(word)
                continue

            # 规则3: 纯标点符号 → 保留
            if all(c in keep_punctuation for c in word):
                cleaned_words.append(word)
                continue

            # 规则4: 混合词 → 拆分后保留中文和标点
            cleaned_chars = []
            for char in word:
                if '\u4e00' <= char <= '\u9fff':  # 中文
                    cleaned_chars.append(char)
                elif char in keep_punctuation:  # 标点
                    cleaned_chars.append(char)
                # 其他字符(英文、数字、特殊符号) → 忽略

            if cleaned_chars:
                cleaned_words.append(''.join(cleaned_chars))

        # 合并清洗后的词
        cleaned_text = ''.join(cleaned_words)

        # 去除多余的连续标点
        cleaned_text = re.sub(r'[，。、；：]+', '，', cleaned_text)

        # 去除开头和结尾的标点
        cleaned_text = re.sub(r'^[，。；：]+', '', cleaned_text)
        cleaned_text = re.sub(r'[，。；：]+$', '', cleaned_text)

        # 去除所有空格
        cleaned_text = re.sub(r'\s+', '', cleaned_text)

        # ✅ 新增: 保存到缓存
        cleaned_text = cleaned_text.strip()

        # ⭐ 修改3-改进版: 只过滤极端长度,不检测重复度
        # 过滤太短的文本(< 5字符,没有实际内容)
        if len(cleaned_text) < 5:
            self.text_clean_cache[text] = ""
            return ""

        # 过滤太长的文本(> 500字符,截断以避免极端情况)
        if len(cleaned_text) > 500:
            cleaned_text = cleaned_text[:500]

        self.text_clean_cache[text] = cleaned_text
        return cleaned_text

    def process_text(self, texts: List[str], max_length: int = None) -> Dict[str, torch.Tensor]:
        """
        处理文本（带智能清洗和停用词过滤）

        处理流程：
        1. 智能清洗（去除噪音）
        2. 停用词过滤
        3. BERT tokenization

        Args:
            texts: 文本列表
            max_length: 最大长度

        Returns:
            包含input_ids和attention_mask的字典
        """
        if max_length is None:
            max_length = self.config.data.max_text_length

        cleaned_texts = []

        for text in texts:
            # 第一步：智能清洗
            cleaned = self.clean_text_smart(text)

            # 第二步：停用词过滤
            if self.stopwords and cleaned:
                words = jieba.lcut(cleaned)
                words = [w for w in words if w not in self.stopwords and len(w) > 0]
                cleaned = ''.join(words)

            # 如果清洗后为空，使用占位符
            if not cleaned or len(cleaned) < 2:
                cleaned = "无有效内容"

            cleaned_texts.append(cleaned)

        # 第三步：BERT tokenization
        encodings = self.tokenizer(
            cleaned_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }

    # ==================== 标签处理相关方法 ====================

    def build_global_label_graph(self, label_paths: List[str]) -> Dict:
        """
        从标签路径构建全局标签图
        """
        print("\n📊 构建全局标签图...")

        # 收集所有节点
        all_nodes = set()
        all_edges = []
        node_levels = {}

        for path_str in tqdm(label_paths, desc="扫描标签路径"):
            if pd.isna(path_str) or not path_str:
                continue

            # ✅ 统一处理：去除所有空格，统一箭头符号
            path_str = str(path_str).strip()
            # 替换所有可能的箭头为标准箭头
            path_str = path_str.replace('-', '→')  # 连字符
            path_str = path_str.replace('—', '→')  # 破折号
            path_str = path_str.replace('–', '→')  # en dash

            # 分割路径
            parts = path_str.split('→')
            # 去除每个部分的前后空格
            parts = [p.strip() for p in parts if p.strip()]

            # 构建累积路径
            for i in range(len(parts)):
                # 累积路径：移动业务, 移动业务→网络问题, 移动业务→网络问题→信号覆盖
                cumulative_path = '→'.join(parts[:i+1])
                all_nodes.add(cumulative_path)
                node_levels[cumulative_path] = i

                # 添加父子边
                if i > 0:
                    parent_path = '→'.join(parts[:i])
                    all_edges.append((parent_path, cumulative_path))
            # ✅ 在这里添加一行去重!!!
            all_edges = list(set(all_edges))  # ← 添加这一行!!!
        # 构建词汇表
        self.node_to_id = {'[PAD]': 0, '[UNK]': 1}
        for node in sorted(all_nodes):
            if node not in self.node_to_id:
                self.node_to_id[node] = len(self.node_to_id)

        self.id_to_node = {v: k for k, v in self.node_to_id.items()}

        # 转换边为ID
        self.global_edges = []
        for parent, child in all_edges:
            if parent in self.node_to_id and child in self.node_to_id:
                parent_id = self.node_to_id[parent]
                child_id = self.node_to_id[child]
                self.global_edges.append([parent_id, child_id])
                self.global_edges.append([child_id, parent_id])  # 双向边

        # 添加自环
        for node_id in range(len(self.node_to_id)):
            self.global_edges.append([node_id, node_id])

        # 保存节点层级
        self.global_node_levels = {}
        for node, level in node_levels.items():
            if node in self.node_to_id:
                self.global_node_levels[self.node_to_id[node]] = level

        print(f"✅ 全局标签图构建完成:")
        print(f"   节点数: {len(self.node_to_id)}")
        print(f"   边数: {len(self.global_edges)}")

        # 统计层级分布
        level_dist = Counter(node_levels.values())
        print(f"   层级分布: {dict(sorted(level_dist.items()))}")
        # ✅ 新增: 保存edge_index和node_levels为tensor属性
        # 这样pretrain_label_global_graph函数就能访问到了
        self.edge_index = torch.tensor(self.global_edges, dtype=torch.long).t().contiguous()
        self.node_levels = torch.zeros(len(self.node_to_id), dtype=torch.long)
        for node_id, level in self.global_node_levels.items():
            self.node_levels[node_id] = level

        return {
            'vocab_size': len(self.node_to_id),
            'num_edges': len(self.global_edges),
            'level_distribution': dict(level_dist)
        }

    def build_global_ontology_tree(self, label_paths: List[str]) -> Dict:
        """
        构建全局本体树（别名方法，调用build_global_label_graph）

        Args:
            label_paths: 标签路径列表

        Returns:
            包含节点词汇表和边信息的字典
        """
        return self.build_global_label_graph(label_paths)

    def build_subgraph_labels(self, df: pd.DataFrame, repeat_column='Repeat complaint',
                              label_column='Complaint label', min_samples=5) -> Dict[str, int]:
        """
        统计每个标签路径的重复投诉率，用于Subgraph Classification预训练

        Args:
            df: 包含标签和重复投诉标注的DataFrame（24万数据）
            repeat_column: 重复投诉列名
            label_column: 标签路径列名
            min_samples: 最小样本数阈值（过滤样本过少的路径）

        Returns:
            subgraph_labels: {标签路径: 0/1} 的字典
                - 0: 低风险（重复率 ≤ 50%）
                - 1: 高风险（重复率 > 50%）
        """
        from collections import defaultdict

        print("\n📊 统计标签路径的重复投诉率...")

        # 统计每个路径的重复情况
        path_stats = defaultdict(lambda: {'repeat': 0, 'total': 0})

        for idx, row in df.iterrows():
            if pd.isna(row[label_column]) or pd.isna(row[repeat_column]):
                continue

            # 标准化标签路径（与build_global_label_graph一致）
            path_str = str(row[label_column]).strip()
            path_str = path_str.replace('-', '→')
            path_str = path_str.replace('–', '→')
            path_str = path_str.replace('—', '→')

            is_repeat = int(row[repeat_column])

            path_stats[path_str]['total'] += 1
            if is_repeat == 1:
                path_stats[path_str]['repeat'] += 1

        # 计算重复率并二值化
        subgraph_labels = {}
        high_risk_count = 0
        low_risk_count = 0
        filtered_count = 0

        for path, stats in path_stats.items():
            # 过滤样本过少的路径
            if stats['total'] < min_samples:
                filtered_count += 1
                continue

            repeat_rate = stats['repeat'] / stats['total']

            # 重复率 > 50% 认为是高风险
            if repeat_rate > 0.5:
                subgraph_labels[path] = 1
                high_risk_count += 1
            else:
                subgraph_labels[path] = 0
                low_risk_count += 1

        print(f"✅ 子图标签统计完成:")
        print(f"   总路径数: {len(path_stats)}")
        print(f"   过滤路径数: {filtered_count} (样本 < {min_samples})")
        print(f"   有效路径数: {len(subgraph_labels)}")
        print(f"   高风险路径: {high_risk_count} ({high_risk_count / len(subgraph_labels) * 100:.1f}%)")
        print(f"   低风险路径: {low_risk_count} ({low_risk_count / len(subgraph_labels) * 100:.1f}%)")

        # 保存为实例属性
        self.subgraph_labels = subgraph_labels

        return subgraph_labels

    def compute_path_risk_scores(self, df, min_samples=10):
        """
        统计每条标签路径的重复投诉风险分数

        用途：Label预训练 - 路径风险回归任务

        从大规模数据（24万）中计算每条路径的历史重复率：
        - 路径A出现100次，其中35次重复 → 风险分数0.35
        - 路径B出现50次，其中6次重复 → 风险分数0.12

        这些风险分数将作为回归目标，训练GAT学会识别高风险路径模式

        Args:
            df: DataFrame，包含'Complaint label'和'Repeat complaint'列
            min_samples: 最小样本数阈值，样本数少于此值的路径不统计

        Returns:
            dict: {路径字符串: 风险分数(0-1)}
        """
        from collections import defaultdict
        import numpy as np

        print("\n" + "=" * 70)
        print("📊 统计标签路径的重复投诉风险分数")
        print("=" * 70)

        path_stats = defaultdict(lambda: {'repeat': 0, 'total': 0})

        # 遍历每条数据，统计每条路径的重复情况
        for idx, row in df.iterrows():
            if pd.isna(row['Complaint label']):
                continue

            path = str(row['Complaint label']).strip()

            # 检查是否重复投诉
            try:
                is_repeat = int(row['Repeat complaint'])
            except (ValueError, KeyError):
                continue

            path_stats[path]['total'] += 1
            if is_repeat == 1:
                path_stats[path]['repeat'] += 1

        # 计算风险分数（只保留样本数充足的路径）
        path_risk_scores = {}
        filtered_count = 0

        for path, stats in path_stats.items():
            if stats['total'] >= min_samples:
                # 计算该路径的历史重复率
                risk_score = stats['repeat'] / stats['total']
                path_risk_scores[path] = risk_score
            else:
                filtered_count += 1

        # 统计信息
        print(f"\n✓ 统计完成:")
        print(f"  - 总路径数: {len(path_stats)}")
        print(f"  - 有效路径数 (样本≥{min_samples}): {len(path_risk_scores)}")
        print(f"  - 过滤路径数 (样本<{min_samples}): {filtered_count}")

        if path_risk_scores:
            scores = list(path_risk_scores.values())
            print(f"\n📈 风险分数分布:")
            print(f"  - 最小值: {min(scores):.4f}")
            print(f"  - 最大值: {max(scores):.4f}")
            print(f"  - 平均值: {np.mean(scores):.4f}")
            print(f"  - 中位数: {np.median(scores):.4f}")

            # 统计风险等级分布
            low_risk = sum(1 for s in scores if s < 0.1)
            mid_risk = sum(1 for s in scores if 0.1 <= s < 0.3)
            high_risk = sum(1 for s in scores if s >= 0.3)

            print(f"\n🎯 风险等级分布:")
            print(f"  - 低风险 (<10%):  {low_risk} 条路径 ({low_risk / len(scores) * 100:.1f}%)")
            print(f"  - 中风险 (10-30%): {mid_risk} 条路径 ({mid_risk / len(scores) * 100:.1f}%)")
            print(f"  - 高风险 (≥30%):  {high_risk} 条路径 ({high_risk / len(scores) * 100:.1f}%)")

        print("=" * 70 + "\n")

        return path_risk_scores
    def encode_label_path_as_graph(self, label: str) -> Tuple[List[int], List[List[int]], List[int]]:
        """
        将标签路径编码为子图 - 修复版

        Args:
            label: 标签路径字符串

        Returns:
            (节点ID列表, 边列表, 节点层级列表)
        """
        # ========== 源头预防: 数据验证和默认值 ==========
        if pd.isna(label) or not self.node_to_id:
            # 返回一个有效的最小图（根节点）
            return [0], [], [0]

        # 检查标签是否为空字符串
        label_str = str(label).strip()
        if label_str == '' or label_str.lower() in ['nan', 'none', '未知', '无']:
            return [0], [], [0]

        # ✅ 统一处理：去除所有空格，统一箭头符号
        label = str(label).strip()
        # 替换所有可能的箭头为标准箭头
        label = label.replace('-', '→')  # 连字符
        label = label.replace('—', '→')  # 破折号
        label = label.replace('–', '→')  # en dash

        # 分割路径
        parts = label.split('→')
        # 去除每个部分的前后空格
        parts = [p.strip() for p in parts if p.strip()]

        node_ids = []
        node_levels = []

        # 构建路径节点
        for i in range(len(parts)):
            cumulative_path = '→'.join(parts[:i + 1])
            if cumulative_path in self.node_to_id:
                node_ids.append(self.node_to_id[cumulative_path])
                node_levels.append(i)
            else:
                # ✅ 找不到节点时的处理
                # 使用 [UNK] 代替缺失的节点
                node_ids.append(1)  # [UNK]的ID是1
                node_levels.append(i)
                # 可选：打印警告（仅在第一次遇到时打印）
                if not hasattr(self, '_warned_missing_nodes'):
                    self._warned_missing_nodes = set()
                if cumulative_path not in self._warned_missing_nodes:
                    print(f"⚠️  节点不在词汇表中: {cumulative_path}")
                    self._warned_missing_nodes.add(cumulative_path)

        # 如果没有找到任何节点（所有节点都是[UNK]）
        if not node_ids or all(nid == 1 for nid in node_ids):
            return [1], [], [0]  # 返回单个[UNK]

        # 构建路径边（父→子，子→父）
        edges = []
        for i in range(len(node_ids) - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])

        # 添加自环
        for i in range(len(node_ids)):
            edges.append([i, i])

        return node_ids, edges, node_levels

    def save_vocabulary(self, save_path: str):
        """
        保存词汇表到文件

        Args:
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        vocab_data = {
            'node_to_id': self.node_to_id,
            'id_to_node': self.id_to_node,
            'global_edges': self.global_edges,
            'global_node_levels': self.global_node_levels
        }

        with open(save_path, 'wb') as f:
            pickle.dump(vocab_data, f)

        print(f"✅ 词汇表已保存: {save_path}")
        print(f"   节点数: {len(self.node_to_id)}")

    def save_global_vocab(self, save_path: str):
        """
        保存全局词汇表（别名方法）

        Args:
            save_path: 保存路径
        """
        self.save_vocabulary(save_path)

    def load_vocabulary(self, load_path: str) -> bool:
        """
        从文件加载词汇表

        Args:
            load_path: 加载路径

        Returns:
            是否加载成功
        """
        if not os.path.exists(load_path):
            print(f"⚠️ 词汇表文件不存在: {load_path}")
            return False

        try:
            with open(load_path, 'rb') as f:
                vocab_data = pickle.load(f)

            self.node_to_id = vocab_data['node_to_id']
            self.id_to_node = vocab_data['id_to_node']
            self.global_edges = vocab_data['global_edges']
            self.global_node_levels = vocab_data['global_node_levels']

            print(f"✓ 从processor.pkl加载词汇表: {load_path}")
            print(f"   节点数: {len(self.node_to_id)}")

            # ✅ 新增: 同时构建edge_index和node_levels tensor
            if self.global_edges:
                self.edge_index = torch.tensor(self.global_edges, dtype=torch.long).t().contiguous()
            else:
                self.edge_index = torch.empty((2, 0), dtype=torch.long)

            if self.global_node_levels:
                self.node_levels = torch.zeros(len(self.node_to_id), dtype=torch.long)
                for node_id, level in self.global_node_levels.items():
                    self.node_levels[node_id] = level
            else:
                self.node_levels = torch.zeros(len(self.node_to_id), dtype=torch.long)

            return True

        except Exception as e:
            print(f"❌ 加载词汇表失败: {str(e)}")
            return False

    def load_global_vocab(self, load_path: str) -> bool:
        """
        加载全局词汇表（别名方法）

        Args:
            load_path: 加载路径

        Returns:
            是否加载成功
        """
        return self.load_vocabulary(load_path)

    # ==================== 结构化特征处理 ====================

    def process_structured_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        处理结构化特征（53个特征）- 修复版本

        数据格式:
        - A列(索引0): new_code - ID列，【排除】
        - B列(索引1): biz_cntt - 投诉文本
        - C列(索引2): Complaint label - 投诉标签
        - D列(索引3)~BD列(索引55): 结构化特征 (53列) 【使用这些】
        - BE列(索引56): Repeat complaint - 目标变量
        """
        print("\n📊 处理结构化特征...")

        col_names = df.columns.tolist()

        # 【核心修复】显式定义要排除的列
        exclude_cols = {'new_code', 'biz_cntt', 'Complaint label', 'Repeat complaint'}

        # 方法1: 通过关键列名定位
        if 'Complaint label' in col_names and 'Repeat complaint' in col_names:
            label_idx = col_names.index('Complaint label')
            target_idx = col_names.index('Repeat complaint')

            start_idx = label_idx + 1
            end_idx = target_idx

            feature_cols = col_names[start_idx:end_idx]

            print(f"✓ 方法1: 从 '{col_names[start_idx]}' 到 '{col_names[end_idx - 1]}'")
            print(f"  起始索引: {start_idx}, 结束索引: {end_idx}")

        # 方法2: 使用固定列索引
        elif len(col_names) >= 57:
            start_idx = 3
            end_idx = 56
            feature_cols = col_names[start_idx:end_idx]
            print(f"✓ 方法2: 使用固定索引 [{start_idx}:{end_idx}]")
        else:
            # 非标准数据集（台湾/Consumer等）: 没有足够列，使用全部非排除列
            feature_cols = [c for c in col_names if c not in exclude_cols]
            print(f"✓ 方法3: 使用所有非排除列，共{len(feature_cols)}列")

        # 二次过滤
        feature_cols = [col for col in feature_cols if col not in exclude_cols]

        print(f"  实际特征列数: {len(feature_cols)}")

        expected_dim = getattr(self.config.model, 'struct_feat_dim', 53)
        if expected_dim == 0 or len(feature_cols) == 0:
            print(f"  struct_feat_dim=0或无特征列，返回零矩阵")
            return torch.zeros((len(df), max(expected_dim, 1)), dtype=torch.float32)
        if len(feature_cols) != expected_dim:
            print(f"⚠️ 警告: 预期{expected_dim}列，实际{len(feature_cols)}列")
            if len(feature_cols) > expected_dim:
                feature_cols = feature_cols[:expected_dim]
                print(f"  已截断为{expected_dim}列")
            elif len(feature_cols) < expected_dim:
                print(f"  列数不足，将用零填充到{expected_dim}列")

        print(f"  前5列: {feature_cols[:5]}")
        print(f"  后5列: {feature_cols[-5:]}")

        if 'new_code' in feature_cols:
            print("❌ 错误: new_code 列被错误包含，正在移除...")
            feature_cols = [col for col in feature_cols if col != 'new_code']
        features = df[feature_cols].values
        # 如果列数不足expected_dim，用零填充
        expected_dim = getattr(self.config.model, 'struct_feat_dim', 53)
        if features.shape[1] < expected_dim:
            padding = np.zeros((features.shape[0], expected_dim - features.shape[1]))
            features = np.hstack([features, padding])
            print(f"✓ 提取特征矩阵: {features.shape} (含零填充)")
        else:
            print(f"✓ 提取特征矩阵: {features.shape}")



        features = np.nan_to_num(features, nan=0.0)

        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            features = self.scaler.fit_transform(features)
            print("✓ 首次标准化（fit_transform）")
        else:
            features = self.scaler.transform(features)
            print("✓ 标准化（transform）")

        print(f"✅ 结构化特征处理完成: {features.shape}")

        return torch.FloatTensor(features)

    # ==================== 完整数据加载 ====================

    def load_data(self, data_file: str, for_pretrain: bool = False):
        """
        加载数据文件(返回DataFrame)
        ⭐ main.py需要的方法

        Args:
            data_file: 数据文件路径
            for_pretrain: 是否用于预训练

        Returns:
            pd.DataFrame: 数据框
        """
        print(f"\n📂 加载数据文件: {data_file}")

        # 读取数据
        if data_file.endswith('.xlsx'):
            df = pd.read_excel(data_file)
        elif data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            raise ValueError(f"不支持的文件格式: {data_file}")

        print(f"✓ 数据集大小: {len(df)}")

        if 'Repeat complaint' in df.columns:
            print(f"✓ 重复投诉比例: {df['Repeat complaint'].mean()*100:.2f}%")

        return df

    def prepare_datasets(self, train_file: str = None, pretrain_file: str = None,
                        for_pretrain: bool = False):
        """
        准备训练/预训练数据集
        ⭐ main.py需要的方法

        Args:
            train_file: 训练数据文件路径
            pretrain_file: 预训练数据文件路径
            for_pretrain: 是否用于预训练

        Returns:
            处理后的数据字典
        """
        # 确定使用哪个文件
        data_file = pretrain_file if for_pretrain else train_file

        if data_file is None:
            raise ValueError("必须指定数据文件路径")

        print(f"\n{'='*60}")
        print(f"📊 准备{'预训练' if for_pretrain else '训练'}数据集")
        print(f"{'='*60}")
        print(f"数据文件: {data_file}")

        # 加载数据
        df = self.load_data(data_file, for_pretrain)

        # 提取文本和标签
        texts = df['biz_cntt'].fillna('').astype(str).tolist()
        labels = df['Complaint label'].fillna('').astype(str).tolist()

        print(f"\n处理文本和标签...")
        print(f"  文本数: {len(texts)}")
        print(f"  标签数: {len(labels)}")

        # 如果没有词汇表,构建全局标签图
        if len(self.node_to_id) == 0:
            print("\n⚠️ 词汇表为空,构建新的全局标签图...")
            self.build_global_label_graph(labels)
        else:
            print(f"\n✓ 使用已加载的词汇表: {len(self.node_to_id)}个节点")

        # 编码文本 - 使用批量处理提高效率
        print("\n编码文本(批量处理)...")

        # 步骤1: 批量清洗文本
        cleaned_texts = []
        for text in tqdm(texts, desc="清洗文本"):
            cleaned_texts.append(self.clean_text_smart(text))

        # 步骤2: 分批编码（避免内存溢出）
        batch_size = 1000
        all_input_ids = []
        all_attention_masks = []

        for i in tqdm(range(0, len(cleaned_texts), batch_size), desc="批量编码"):
            batch_texts = cleaned_texts[i:i + batch_size]

            # 批量BERT编码
            batch_encoding = self.tokenizer(
                batch_texts,
                max_length=self.config.model.bert_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            all_input_ids.append(batch_encoding['input_ids'])
            all_attention_masks.append(batch_encoding['attention_mask'])

        # 步骤3: 合并所有批次
        text_data = {
            'input_ids': torch.cat(all_input_ids, dim=0),
            'attention_mask': torch.cat(all_attention_masks, dim=0)
        }

        print(f"✓ 文本编码完成: {text_data['input_ids'].shape}")

        # 编码标签为图
        print("\n编码标签为图结构...")
        node_ids_list = []
        edges_list = []
        node_levels_list = []

        for label in tqdm(labels, desc="编码标签图"):
            node_ids, edges, levels = self.encode_label_path_as_graph(label)
            node_ids_list.append(node_ids)
            edges_list.append(edges)
            node_levels_list.append(levels)

        # 如果不是预训练,处理结构化特征和目标变量
        if not for_pretrain:
            print("\n处理结构化特征和目标变量...")
            struct_features = self.process_structured_features(df)

            if 'Repeat complaint' in df.columns:
                targets = torch.LongTensor(df['Repeat complaint'].values)
            else:
                print("⚠️ 未找到目标变量列'Repeat complaint',使用全0")
                targets = torch.zeros(len(df), dtype=torch.long)
        else:
            # 预训练时创建占位符
            _sdim = getattr(self.config.model, 'struct_feat_dim', 53)
            struct_features = torch.zeros((len(df), max(_sdim, 1)), dtype=torch.float32)
            targets = torch.zeros(len(df), dtype=torch.long)

        print(f"\n✅ 数据准备完成:")
        print(f"   文本数据: {text_data['input_ids'].shape}")
        print(f"   标签数据: {len(node_ids_list)}个图")
        print(f"   结构化特征: {struct_features.shape}")
        print(f"   目标变量: {targets.shape}")
        print(f"   词汇表大小: {len(self.node_to_id)}")

        return {
            'text_data': text_data,
            'node_ids_list': node_ids_list,
            'edges_list': edges_list,
            'node_levels_list': node_levels_list,
            'struct_features': struct_features,
            'targets': targets,
            'vocab_size': len(self.node_to_id)
        }

    def load_and_process_data(self, data_file: str, text_col='biz_cntt',
                              label_col='Complaint label', target_col='Repeat complaint'):
        """
        加载并处理完整数据集（文本会被智能清洗）

        Args:
            data_file: 数据文件路径
            text_col: 文本列名
            label_col: 标签列名
            target_col: 目标变量列名

        Returns:
            处理后的数据字典
        """
        print(f"\n📂 加载数据: {data_file}")

        # 读取数据
        if data_file.endswith('.xlsx'):
            df = pd.read_excel(data_file)
        elif data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            raise ValueError(f"不支持的文件格式: {data_file}")

        print(f"数据集大小: {len(df)}")
        print(f"重复投诉比例: {df[target_col].mean()*100:.2f}%")

        # 1. 处理文本（会自动智能清洗）
        print("\n处理文本和标签...")
        texts = df[text_col].fillna('').astype(str).tolist()

        # 2. 处理标签
        labels = df[label_col].fillna('').astype(str).tolist()

        # 如果没有词汇表，构建全局标签图
        if len(self.node_to_id) == 0:
            print("\n构建新的全局标签图...")
            self.build_global_label_graph(labels)
        else:
            print(f"\n使用已加载的词汇表: {len(self.node_to_id)}个节点")

        # 3. 处理结构化特征
        struct_features = self.process_structured_features(df)

        # 4. 编码标签为图
        print("\n编码标签为图结构...")
        encoded_labels = []
        for label in tqdm(labels, desc="编码标签图"):
            node_ids, edges, levels = self.encode_label_path_as_graph(label)
            encoded_labels.append({
                'node_ids': node_ids,
                'edges': edges,
                'node_levels': levels
            })

        # 5. 处理目标变量
        targets = df[target_col].values

        print(f"\n✅ 数据处理完成:")
        print(f"   文本数: {len(texts)}")
        print(f"   标签数: {len(labels)}")
        print(f"   结构化特征: {struct_features.shape}")
        print(f"   目标变量分布: {Counter(targets)}")

        return {
            'texts': texts,
            'labels': labels,
            'encoded_labels': encoded_labels,
            'struct_features': struct_features,
            'targets': targets,
            'dataframe': df
        }

    def get_vocab_size(self) -> int:
        """返回标签词汇表大小"""
        return len(self.node_to_id) if self.node_to_id else 0

    def save(self, save_path: str):
        """
        保存处理器状态

        Args:
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        state = {
            'node_to_id': self.node_to_id,
            'id_to_node': self.id_to_node,
            'global_edges': self.global_edges,
            'global_node_levels': self.global_node_levels,
            'scaler_mean': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
        }

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

        print(f"✅ 处理器状态已保存: {save_path}")
        print(f"   词汇表大小: {len(self.node_to_id)}")

    def load(self, load_path: str) -> bool:
        """
        加载处理器状态

        Args:
            load_path: 加载路径

        Returns:
            是否加载成功
        """
        if not os.path.exists(load_path):
            print(f"⚠️ 处理器状态文件不存在: {load_path}")
            return False

        try:
            with open(load_path, 'rb') as f:
                state = pickle.load(f)

            self.node_to_id = state['node_to_id']
            self.id_to_node = state['id_to_node']
            self.global_edges = state['global_edges']
            self.global_node_levels = state['global_node_levels']

            if state['scaler_mean'] is not None:
                self.scaler.mean_ = state['scaler_mean']
                self.scaler.scale_ = state['scaler_scale']

            print(f"✅ 处理器状态已加载: {load_path}")
            print(f"   词汇表大小: {len(self.node_to_id)}")

            return True
        except Exception as e:
            print(f"❌ 加载处理器状态失败: {str(e)}")
            return False


# ==================== ComplaintDataset类 ====================
# ⭐ 必须存在! main.py, train.py, ablation_study.py都需要导入这个类

class ComplaintDataset(Dataset):
    """
    投诉数据集类

    这个类封装数据供DataLoader使用,是PyTorch数据加载的标准方式
    """

    def __init__(self, text_data, node_ids_list, edges_list, node_levels_list,
                 struct_features, targets=None):
        """
        初始化数据集

        Args:
            text_data: 文本数据字典,包含input_ids和attention_mask
            node_ids_list: 节点ID列表(每个样本是一个列表)
            edges_list: 边列表(每个样本是一个列表)
            node_levels_list: 节点层级列表(每个样本是一个列表)
            struct_features: 结构化特征张量 [N, 53]
            targets: 目标标签(可选)
        """
        self.input_ids = text_data['input_ids']
        self.attention_mask = text_data['attention_mask']
        self.node_ids_list = node_ids_list
        self.edges_list = edges_list
        self.node_levels_list = node_levels_list
        self.struct_features = struct_features
        self.targets = targets

    def __len__(self):
        """返回数据集大小"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            包含所有模态数据的字典
        """
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'node_ids': self.node_ids_list[idx],
            'edges': self.edges_list[idx],
            'node_levels': self.node_levels_list[idx],
            'struct_features': self.struct_features[idx]
        }

        if self.targets is not None:
            item['target'] = torch.tensor(self.targets[idx], dtype=torch.long)

        return item


# ==================== custom_collate_fn函数 ====================
# ⭐ 必须存在! 处理变长图数据batching的关键函数

def custom_collate_fn(batch):
    """
    自定义批次整理函数

    为什么需要这个函数?
    因为每个样本的标签路径长度不同(有的3层,有的5层),
    标准的DataLoader无法自动处理变长的图数据,
    所以需要这个自定义函数来正确组batch

    Args:
        batch: 样本列表,每个样本是__getitem__返回的字典

    Returns:
        整理好的batch字典
    """
    # 提取batch中的所有数据
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    node_ids_list = [item['node_ids'] for item in batch]
    edges_list = [item['edges'] for item in batch]
    node_levels_list = [item['node_levels'] for item in batch]
    struct_features = torch.stack([item['struct_features'] for item in batch])

    # 处理target(如果存在)
    if 'target' in batch[0]:
        targets = torch.stack([item['target'] for item in batch])
    else:
        targets = None

    # 对文本进行padding(使batch内所有序列长度相同)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # 组装返回的batch
    batched_data = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'node_ids': node_ids_list,  # 保持为list,因为每个样本长度不同
        'edges': edges_list,  # 保持为list
        'node_levels': node_levels_list,  # 保持为list
        'struct_features': struct_features
    }

    if targets is not None:
        batched_data['target'] = targets

    return batched_data


# ============================================================
# 新增: 平衡批次采样器
# 作用: 确保每个batch中正负样本比例合理
# 用于: Text预训练阶段2 (对比学习)
# 位置: 在 if __name__ == "__main__": 之前插入
# ============================================================

class BalancedBatchSampler:
    """
    平衡批次采样器

    目的:
        解决对比学习中的样本不平衡问题

    原始数据分布:
        - 重复投诉: 7.31% (17,000条)
        - 非重复: 92.69% (223,000条)

    使用BalancedBatchSampler:
        - 强制: 每batch 30%重复 + 70%非重复
        - 实际: 每batch 19个重复 + 45个非重复
        - 正样本对: C(19,2)=171对 (充足!)
    """

    def __init__(self, labels, batch_size=64, pos_ratio=0.3, shuffle=True):
        """
        Args:
            labels: 所有样本的标签列表 (0或1)
            batch_size: 批次大小
            pos_ratio: 正样本(重复投诉)占比，推荐0.3
            shuffle: 是否每个epoch打乱
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.shuffle = shuffle

        # 分离正负样本的索引
        self.pos_indices = np.where(self.labels == 1)[0]  # 重复投诉索引
        self.neg_indices = np.where(self.labels == 0)[0]  # 非重复索引

        self.num_pos = len(self.pos_indices)
        self.num_neg = len(self.neg_indices)

        # 计算每个batch需要的正负样本数
        self.num_pos_per_batch = int(self.batch_size * self.pos_ratio)
        self.num_neg_per_batch = self.batch_size - self.num_pos_per_batch

        # 打印统计信息
        print(f"✅ BalancedBatchSampler 初始化:")
        print(f"  - 总样本: {len(labels)}")
        print(f"  - 正样本(重复): {self.num_pos} ({self.num_pos / len(labels) * 100:.2f}%)")
        print(f"  - 负样本(非重复): {self.num_neg} ({self.num_neg / len(labels) * 100:.2f}%)")
        print(f"  - Batch size: {batch_size}")
        print(f"  - 每batch正样本: {self.num_pos_per_batch} ({pos_ratio * 100:.0f}%)")
        print(f"  - 每batch负样本: {self.num_neg_per_batch} ({(1 - pos_ratio) * 100:.0f}%)")

        # 计算正样本对数量
        pos_pairs = self.num_pos_per_batch * (self.num_pos_per_batch - 1) // 2
        neg_pairs = self.num_neg_per_batch * (self.num_neg_per_batch - 1) // 2
        cross_pairs = self.num_pos_per_batch * self.num_neg_per_batch

        print(f"  - 每batch样本对:")
        print(f"    * 正类内对: {pos_pairs}对")
        print(f"    * 负类内对: {neg_pairs}对")
        print(f"    * 跨类对: {cross_pairs}对")

    def __iter__(self):
        """生成balanced batch的迭代器"""

        # 每个epoch开始时打乱索引
        if self.shuffle:
            np.random.shuffle(self.pos_indices)
            np.random.shuffle(self.neg_indices)

        pos_ptr = 0  # 正样本指针
        neg_ptr = 0  # 负样本指针

        # 持续生成batch直到某一类样本用完
        while (pos_ptr + self.num_pos_per_batch <= self.num_pos and
               neg_ptr + self.num_neg_per_batch <= self.num_neg):
            batch_indices = []

            # 采样正样本
            pos_batch = self.pos_indices[pos_ptr:pos_ptr + self.num_pos_per_batch]
            batch_indices.extend(pos_batch.tolist())
            pos_ptr += self.num_pos_per_batch

            # 采样负样本
            neg_batch = self.neg_indices[neg_ptr:neg_ptr + self.num_neg_per_batch]
            batch_indices.extend(neg_batch.tolist())
            neg_ptr += self.num_neg_per_batch

            # 打乱batch内的顺序（让正负样本混合）
            np.random.shuffle(batch_indices)

            yield batch_indices

    def __len__(self):
        """估算batch数量"""
        # 取决于哪个样本类型先用完
        num_batches_pos = self.num_pos // self.num_pos_per_batch
        num_batches_neg = self.num_neg // self.num_neg_per_batch
        return min(num_batches_pos, num_batches_neg)


# ============================================================
# 新增: 对比学习文本数据集
# 作用: 封装文本数据用于对比学习
# 用于: Text预训练阶段2
# ============================================================

class ContrastiveTextDataset(Dataset):
    """
    用于对比学习的文本数据集

    与普通分类Dataset的区别:
        普通: 返回 (text, label) 用于计算CE loss
        对比: 返回 (text, label) 用于构造正负样本对

    label的作用:
        - 不是用于计算分类loss
        - 而是用于判断哪些样本是正样本对(同标签)
        - SupConLoss内部会用label构造mask
    """

    def __init__(self, texts, labels, tokenizer, max_length=256):
        """
        Args:
            texts: 文本列表
            labels: 标签列表 (0或1)
            tokenizer: BERT tokenizer
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"✅ ContrastiveTextDataset 初始化:")
        print(f"  - 样本数: {len(texts)}")
        print(f"  - 最大长度: {max_length}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        获取单个样本

        Returns:
            dict:
                'input_ids': [seq_len] - token ids
                'attention_mask': [seq_len] - attention mask
                'label': scalar - 类别标签 (0或1)
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # [seq_len]
            'attention_mask': encoding['attention_mask'].squeeze(0),  # [seq_len]
            'label': torch.tensor(label, dtype=torch.long)  # scalar
        }


# ============================================================
# 新增: 平衡批次采样器
# ============================================================




class ContrastiveTextDataset(Dataset):
    """用于对比学习的文本数据集"""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
# ==================== 测试代码 ====================

if __name__ == "__main__":
    """测试数据处理器"""
    print("测试数据处理器...")

    # 创建简单的配置对象
    class SimpleConfig:
        class Model:
            bert_model_name = 'bert-base-chinese'
        class Data:
            max_text_length = 256
        model = Model()
        data = Data()

    config = SimpleConfig()

    # 创建处理器
    processor = ComplaintDataProcessor(
        config,
        user_dict_file='new_user_dict.txt',
        stopword_file='new_stopword_dict.txt'
    )

    # 测试智能清洗
    test_texts = [
        "#同步要素#客户反映[101]移动4G信号差[/101]多次投诉未解决#同步要素#",
        "流程轨迹：(1)【自动】是否有在途工单：否(0) 客户要求上门处理",
        "【人工】是否需要派单：是(1) 工单号：WO2024001234",
        "客户属于疑似户线问题关怀，请智慧家庭工程师尽快上门处理。",
    ]

    print("\n智能清洗测试:")
    print("="*70)
    for i, text in enumerate(test_texts, 1):
        print(f"\n原始文本 {i}:")
        print(f"  {text}")
        cleaned = processor.clean_text_smart(text)
        print(f"清洗后:")
        print(f"  {cleaned}")

        # 检查关键点
        has_brackets = '(' in cleaned or '【' in cleaned or ')' in cleaned or '】' in cleaned
        print(f"检查: 括号={'❌残留' if has_brackets else '✅清除'}", end="")
        if '4G' in text:
            print(f" | 4G={'✅保留' if '4G' in cleaned else '❌丢失'}", end="")
        print()

    print("\n✅ 数据处理器测试完成!")