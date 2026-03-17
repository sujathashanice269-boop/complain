"""
Taiwan Restaurant Dataset - 完全独立版
=================================================
完全独立运行，不依赖任何主文件（config.py, model.py, data_processor.py, ablation_study.py）
所有依赖已内联到本文件中。

数据集特点：三模态（文本+标签+结构化特征4维），类别不平衡需加权
BERT模型：bert-base-chinese
结构化特征：is_weekend, is_peak, season_encoded, meal_period_encoded

Usage:
    python run_taiwan_restaurant_standalone.py --mode all
    python run_taiwan_restaurant_standalone.py --mode baseline
    python run_taiwan_restaurant_standalone.py --mode ablation
"""

import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
import json
import gc
import re
import pickle
import random
from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from transformers import BertModel, BertTokenizer
from torch.nn.utils.rnn import pad_sequence

import jieba

try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("⚠️ torch_geometric not available, GAT features will be limited")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from dataclasses import dataclass, field


# ============================================================
# Inlined Config (from config.py)
# ============================================================

@dataclass
class ModelConfig:
    bert_model_name: str = 'bert-base-chinese'
    bert_max_length: int = 256
    label_embedding_dim: int = 128
    label_hidden_dim: int = 256
    num_gat_layers: int = 3
    num_gat_heads: int = 4
    max_label_depth: int = 8
    use_cross_attention: bool = True
    cross_attn_heads: int = 4
    fusion_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.3
    struct_feat_dim: int = 53
    use_feature_importance: bool = True
    has_struct_features: bool = True

@dataclass
class PretrainConfig:
    stage1_epochs: int = 30
    stage1_lr: float = 5e-5
    stage1_mask_prob: float = 0.15
    use_span_masking: bool = True
    span_mask_length: int = 3
    span_mask_prob: float = 0.3
    stage2_epochs: int = 20
    stage2_lr: float = 3e-5
    use_contrastive: bool = False
    contrastive_temperature: float = 0.5
    contrastive_loss_weight: float = 0.3
    use_global_graph_pretrain: bool = False
    global_graph_epochs: int = 10
    global_graph_lr: float = 1e-4
    use_node_prediction: bool = True
    use_link_prediction: bool = True
    use_subgraph_classification: bool = True
    pretrain_batch_size: int = 32
    save_steps: int = 500
    eval_steps: int = 500

@dataclass
class TrainingConfig:
    data_file: str = '小案例ai问询.xlsx'
    large_data_file: str = '多模态初始表_数据标签.xlsx'
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    use_curriculum_learning: bool = True
    stage1_single_modal_epochs: int = 10
    stage1_lr: float = 2e-5
    stage2_dual_modal_epochs: int = 10
    stage2_lr: float = 1e-5
    stage3_full_epochs: int = 20
    stage3_lr: float = 5e-6
    num_epochs: int = 30
    learning_rate: float = 2e-5
    batch_size: int = 16
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 5.0
    warmup_steps: int = 500
    scheduler_type: str = 'cosine'
    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.001
    use_modal_balance_loss: bool = False
    modal_balance_weight: float = 0.1
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    class_weight: list = None
    use_label_smoothing: bool = False
    label_smoothing: float = 0.1
    use_data_augmentation: bool = True
    augmentation_prob: float = 0.5
    device: str = 'cuda'
    num_workers: int = 0
    save_dir: str = './models'
    log_dir: str = './logs'
    save_steps: int = 1000
    log_interval: int = 10
    pretrain_save_dir: str = './pretrained_complaint_bert_improved'
    label_pretrain_save_dir: str = './pretrained_label_graph'

@dataclass
class DataConfig:
    max_text_length: int = 256
    max_label_nodes: int = 10
    user_dict_file: str = 'new_user_dict.txt'
    struct_start_col: int = 5
    struct_end_col: int = 57
    text_augment_prob: float = 0.5
    synonym_replace_prob: float = 0.3
    random_delete_prob: float = 0.1
    random_swap_prob: float = 0.1

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        import torch as _torch
        if self.training.device == 'cuda' and not _torch.cuda.is_available():
            print("⚠️ CUDA不可用，切换到CPU")
            self.training.device = 'cpu'
        os.makedirs(self.training.save_dir, exist_ok=True)
        os.makedirs(self.training.log_dir, exist_ok=True)
        os.makedirs(self.training.pretrain_save_dir, exist_ok=True)
        os.makedirs(self.training.label_pretrain_save_dir, exist_ok=True)

def get_taiwan_restaurant_config() -> Config:
    config = Config()
    config.model.struct_feat_dim = 4
    config.model.bert_model_name = 'bert-base-chinese'
    config.training.data_file = 'Restaurant Complaint balanced.xlsx'
    config.training.class_weight = [1.0, 3.03]
    print("🍜 使用台湾餐厅数据集配置 (struct_dim=4, class_weight=[1.0, 3.03])")
    return config


# ============================================================
# Inlined from data_processor.py
# ============================================================

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
# Inlined from model.py - MultiModalComplaintModel and dependencies
# ============================================================

class CrossModalAttention_Full(nn.Module):
    """跨模态注意力模块 - 方向四核心创新"""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )

    def forward(self, query, key_value):
        """
        Args:
            query: [batch, 1, dim]
            key_value: [batch, 1, dim]
        Returns:
            enhanced: [batch, 1, dim]
            attn_weights: 注意力权重
        """
        # Cross-Attention
        attn_output, attn_weights = self.attention(query, key_value, key_value)

        # 残差连接
        query = self.layer_norm1(query + attn_output)

        # FFN
        ffn_output = self.ffn(query)
        output = self.layer_norm2(query + ffn_output)

        return output, attn_weights

class TextMultiTokenGenerator(nn.Module):
    """从BERT多层输出生成多个语义Token"""

    def __init__(self, bert_hidden_size=768, output_dim=256, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens

        self.layer_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(bert_hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            ) for _ in range(num_tokens)
        ])

        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_tokens, output_dim) * 0.02
        )

    def forward(self, bert_hidden_states):
        """
        Args:
            bert_hidden_states: tuple，BERT所有层输出
        Returns:
            text_tokens: [batch, num_tokens, output_dim]
        """
        batch_size = bert_hidden_states[-1].size(0)

        tokens = []
        for i, proj in enumerate(self.layer_projections):
            layer_idx = -(self.num_tokens - i)  # -4, -3, -2, -1
            cls_token = bert_hidden_states[layer_idx][:, 0, :]  # [batch, 768]
            projected = proj(cls_token)  # [batch, 256]
            tokens.append(projected.unsqueeze(1))

        text_tokens = torch.cat(tokens, dim=1)  # [batch, 4, 256]
        text_tokens = text_tokens + self.position_embeddings.expand(batch_size, -1, -1)

        return text_tokens

class StructMultiTokenGenerator(nn.Module):
    """将结构化特征生成多个Token"""

    def __init__(self, input_dim=53, output_dim=256, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens

        self.token_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, output_dim),
                nn.LayerNorm(output_dim)
            ) for _ in range(num_tokens)
        ])

        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_tokens, output_dim) * 0.02
        )

    def forward(self, struct_features):
        """
        Args:
            struct_features: [batch, input_dim]
        Returns:
            struct_tokens: [batch, num_tokens, output_dim]
        """
        batch_size = struct_features.size(0)

        tokens = []
        for generator in self.token_generators:
            token = generator(struct_features)  # [batch, 256]
            tokens.append(token.unsqueeze(1))

        struct_tokens = torch.cat(tokens, dim=1)  # [batch, 4, 256]
        struct_tokens = struct_tokens + self.position_embeddings.expand(batch_size, -1, -1)

        return struct_tokens

class TextLedCrossModalAttention(nn.Module):
    """
    文本主导的跨模态注意力

    设计：
    - 文本对标签和结构做跨模态注意力
    - 各模态自注意力增强表示
    - 门控融合，文本权重更高
    """

    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim

        # 文本对其他模态的注意力
        self.text_to_label_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_struct_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # 各模态自注意力
        self.text_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.label_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.struct_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # 层归一化
        self.text_norm = nn.LayerNorm(dim)
        self.label_norm = nn.LayerNorm(dim)
        self.struct_norm = nn.LayerNorm(dim)

        # 门控融合
        self.modal_gate = nn.Sequential(
            nn.Linear(dim * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )

        # 文本偏置（确保文本权重更高）
        self.text_bias = nn.Parameter(torch.tensor(0.5))
        # 可学习的跨模态融合权重
        self.cross_modal_weight_label = nn.Parameter(torch.tensor(0.0))
        self.cross_modal_weight_struct = nn.Parameter(torch.tensor(0.0))

    def forward(self, text_tokens, label_tokens, struct_tokens,
                label_mask=None, return_attention=True):
        """
        Args:
            text_tokens: [batch, 4, 256]
            label_tokens: [batch, N, 256]
            struct_tokens: [batch, 4, 256]
            label_mask: [batch, N] - True表示padding位置
        """
        attention_weights = {}

        # 1. 自注意力
        text_self, _ = self.text_self_attn(text_tokens, text_tokens, text_tokens)
        text_tokens = self.text_norm(text_tokens + text_self)

        label_self, attn_l = self.label_self_attn(
            label_tokens, label_tokens, label_tokens,
            key_padding_mask=label_mask,
            need_weights=return_attention,
            average_attn_weights=False
        )
        label_tokens = self.label_norm(label_tokens + label_self)

        struct_self, _ = self.struct_self_attn(struct_tokens, struct_tokens, struct_tokens)
        struct_tokens = self.struct_norm(struct_tokens + struct_self)

        # 2. 文本主导的跨模态注意力
        text_to_label, attn_t2l = self.text_to_label_attn(
            text_tokens, label_tokens, label_tokens,
            key_padding_mask=label_mask,
            need_weights=return_attention,
            average_attn_weights=False
        )

        text_to_struct, attn_t2s = self.text_to_struct_attn(
            text_tokens, struct_tokens, struct_tokens,
            need_weights=return_attention,
            average_attn_weights=False
        )

        # 3. 特征增强
        weight_label = torch.sigmoid(self.cross_modal_weight_label)
        weight_struct = torch.sigmoid(self.cross_modal_weight_struct)
        text_enhanced = text_tokens + weight_label * text_to_label + weight_struct * text_to_struct
        label_enhanced = label_tokens
        struct_enhanced = struct_tokens

        # 4. 池化
        text_pooled = text_enhanced.mean(dim=1)  # [batch, 256]
        label_pooled = label_enhanced.mean(dim=1)  # [batch, 256]
        struct_pooled = struct_enhanced.mean(dim=1)  # [batch, 256]

        # 5. 门控融合（文本主导）
        gate_input = torch.cat([text_pooled, label_pooled, struct_pooled], dim=-1)
        gate_logits = self.modal_gate(gate_input)  # [batch, 3]
        gate_logits[:, 0] = gate_logits[:, 0] + self.text_bias  # 文本偏置
        gate_weights = F.softmax(gate_logits, dim=-1)  # [batch, 3]

        if return_attention:
            attention_weights = {
                'text_to_label': attn_t2l,  # [batch, heads, 4, N]
                'text_to_struct': attn_t2s,  # [batch, heads, 4, 4]
                'label_self': attn_l,  # [batch, heads, N, N]
                'modal_weights': gate_weights,  # [batch, 3]
            }

        return text_pooled, label_pooled, struct_pooled, attention_weights


class GATLabelEncoder(nn.Module):
    """GAT标签编码器 - 支持全局图预训练"""

    def __init__(self, vocab_size, hidden_dim=256, num_layers=3, num_heads=4, max_level=8):
        super().__init__()

        # 节点嵌入
        self.node_embedding = nn.Embedding(vocab_size, 128)

        # 层级嵌入 - 方向二改进
        self.level_embedding = nn.Embedding(max_level + 1, 32)

        # 层级权重 - 使用递增初始化，让深层节点有更大初始权重
        level_init = torch.zeros(max_level + 1)
        for i in range(max_level + 1):
            level_init[i] = i * 0.1  # 深层权重更大
        self.level_weights = nn.Parameter(level_init)

        # GAT层
        self.gat_layers = nn.ModuleList()
        input_dim = 128 + 32

        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=0.3)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=1, dropout=0.3)
                )

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_ids, edge_index, node_levels, batch=None):
        """前向传播"""
        # 节点嵌入
        x = self.node_embedding(node_ids)  # [num_nodes, 128]

        # 层级嵌入
        level_emb = self.level_embedding(node_levels)  # [num_nodes, 32]

        # 拼接
        x = torch.cat([x, level_emb], dim=-1)  # [num_nodes, 160]

        # GAT层
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = F.elu(x)

        # 层级加权 - 方向二改进
        level_weights = torch.softmax(self.level_weights, dim=0)
        weighted_x = x * level_weights[node_levels].unsqueeze(-1)

        # 输出投影和归一化（对所有节点）
        weighted_x = self.output_proj(weighted_x)
        weighted_x = self.layer_norm(weighted_x)

        # ========== 返回所有节点特征（修复P01）==========
        max_nodes = 8  # 标签路径最大节点数

        if batch is not None:
            batch_features = []
            batch_masks = []
            unique_batches = torch.unique(batch)

            for b in unique_batches:
                node_mask = (batch == b)
                nodes = weighted_x[node_mask]  # [num_nodes, hidden_dim]
                num_nodes = nodes.size(0)

                # Padding到max_nodes
                if num_nodes < max_nodes:
                    pad_size = max_nodes - num_nodes
                    padding = torch.zeros(pad_size, nodes.size(-1), device=nodes.device)
                    nodes = torch.cat([nodes, padding], dim=0)
                    attn_mask = torch.cat([
                        torch.zeros(num_nodes, dtype=torch.bool, device=nodes.device),
                        torch.ones(pad_size, dtype=torch.bool, device=nodes.device)
                    ])
                else:
                    nodes = nodes[:max_nodes]
                    attn_mask = torch.zeros(max_nodes, dtype=torch.bool, device=nodes.device)

                batch_features.append(nodes.unsqueeze(0))
                batch_masks.append(attn_mask.unsqueeze(0))

            node_features = torch.cat(batch_features, dim=0)  # [batch, max_nodes, dim]
            node_masks = torch.cat(batch_masks, dim=0)  # [batch, max_nodes]
            return node_features, node_masks

        # 单样本情况
        num_nodes = weighted_x.size(0)
        if num_nodes < max_nodes:
            pad_size = max_nodes - num_nodes
            padding = torch.zeros(pad_size, weighted_x.size(-1), device=weighted_x.device)
            weighted_x = torch.cat([weighted_x, padding], dim=0)
            attn_mask = torch.cat([
                torch.zeros(num_nodes, dtype=torch.bool, device=weighted_x.device),
                torch.ones(pad_size, dtype=torch.bool, device=weighted_x.device)
            ])
        else:
            weighted_x = weighted_x[:max_nodes]
            attn_mask = torch.zeros(max_nodes, dtype=torch.bool, device=weighted_x.device)

        return weighted_x.unsqueeze(0), attn_mask.unsqueeze(0)

class MultiModalComplaintModel(nn.Module):
    """多模态投诉预测模型 - 完整版"""

    def __init__(self, config, vocab_size, mode='full', pretrained_path=None, No_pretrain_bert=False, use_flat_label=False):

        """
        Args:
            config: 配置对象
            vocab_size: 标签词汇表大小
            mode: 模型模式
                - 'full': 完整三模态
                - 'text_only': 仅文本
                - 'label_only': 仅标签
                - 'struct_only': 仅结构化
                - 'text_label': 文本+标签
                - 'text_struct': 文本+结构化
                - 'label_struct': 标签+结构化
            pretrained_path: 预训练模型路径
        """
        super().__init__()

        self.config = config
        self.mode = mode
        self.device = config.training.device

        # ========== 文本编码器 (BERT) - 方向一改进 ==========
        if mode in ['full', 'text_only', 'text_label', 'text_struct']:
            if No_pretrain_bert:
                # 【新增】完全随机初始化BERT（真正从零训练）
                print("🔄 使用随机初始化的BERT（从零训练）")
                from transformers import BertConfig
                bert_config = BertConfig.from_pretrained(config.model.bert_model_name)
                self.text_encoder = BertModel(bert_config)  # 随机初始化权重
            elif pretrained_path and os.path.exists(pretrained_path):
                print(f"✅ 加载领域预训练BERT: {pretrained_path}")
                self.text_encoder = BertModel.from_pretrained(pretrained_path)
            else:
                print("📦 使用原始BERT预训练权重（无领域预训练）")
                self.text_encoder = BertModel.from_pretrained(config.model.bert_model_name)

            # ✅ 核心修复：添加投影层（统一维度到256）
            self.text_proj = nn.Linear(768, 256)

            # 对比学习投影头（如果需要）
            self.text_contrast_proj = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        else:
            self.text_encoder = None
            self.text_proj = None  # ✅ 明确设置为None
            self.text_contrast_proj = None

        # ========== 标签编码器 (GAT/Flat) - 方向二改进 ==========
        self.use_flat_label = use_flat_label
        if mode in ['full', 'label_only', 'text_label', 'label_struct']:
            if use_flat_label:
                # Flat MLP编码（对照组） - 不使用图结构
                print("📋 使用Flat MLP标签编码（对照组，无图结构）")
                self.label_encoder = None
                self.flat_label_encoder = nn.Sequential(
                    nn.Embedding(vocab_size, 128, padding_idx=0),
                )
                self.flat_label_mlp = nn.Sequential(
                    nn.Linear(128 * 8, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
            else:
                # GAT图编码（实验组） - 使用图注意力网络
                self.label_encoder = GATLabelEncoder(
                    vocab_size=vocab_size,
                    hidden_dim=256,
                    num_layers=3,
                    num_heads=4
                )
                self.flat_label_encoder = None
                self.flat_label_mlp = None

            # 加载全局图预训练权重（仅GAT模式）
            if pretrained_path and not use_flat_label:
                label_pretrain_path = os.path.join(
                    config.training.label_pretrain_save_dir,
                    'label_global_pretrain.pth'
                )
                if os.path.exists(label_pretrain_path):
                    try:
                        state_dict = torch.load(label_pretrain_path, map_location=self.device)
                        self.label_encoder.load_state_dict(state_dict, strict=False)
                        print("✅ 加载全局图预训练权重")
                    except Exception as e:
                        print(f"⚠️ 全局图预训练权重加载失败: {e}")
        else:
            self.label_encoder = None

        # ========== 结构化特征编码器 - 方向三改进 ==========
        if mode in ['full', 'struct_only', 'text_struct', 'label_struct']:
            struct_dim = config.model.struct_feat_dim  # 动态适配不同数据集
            print(f"📐 结构化特征维度: {struct_dim}")

            self.struct_encoder = nn.Sequential(
                nn.Linear(struct_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

            # 特征重要性权重 - 使用正态分布初始化以打破对称性
            self.feature_importance = nn.Parameter(torch.randn(struct_dim) * 0.1)
        else:
            self.struct_encoder = None
            self.feature_importance = None

        # ========== 跨模态注意力 - 文本主导方案（修复P01）==========
        if mode == 'full':
            # Token生成器
            self.text_token_gen = TextMultiTokenGenerator(
                bert_hidden_size=768, output_dim=256, num_tokens=4
            )
            self.struct_token_gen = StructMultiTokenGenerator(
                input_dim=config.model.struct_feat_dim, output_dim=256, num_tokens=4
            )
            # 文本主导的跨模态注意力
            self.text_led_cross_modal = TextLedCrossModalAttention(
                dim=256, num_heads=4, dropout=0.1
            )
        elif mode in ['text_label', 'text_struct', 'label_struct']:
            # 双模态交互
            self.modal_attn_1 = CrossModalAttention_Full(256, num_heads=4)
            self.modal_attn_2 = CrossModalAttention_Full(256, num_heads=4)

        # ========== 融合层 ==========
        # 计算融合层输入维度
        if mode == 'full':
            fusion_input_dim = 256 * 3  # text + label + struct
        elif mode in ['text_label', 'text_struct', 'label_struct']:
            fusion_input_dim = 256 * 2
        else:
            fusion_input_dim = 256  # 单模态统一256维

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.model.fusion_dim),
            nn.LayerNorm(config.model.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.fusion_dim, config.model.hidden_dim),
            nn.LayerNorm(config.model.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.model.dropout)
        )

        # ========== 分类头 ==========
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(config.model.hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        # ========== 源头预防: 权重初始化 ==========
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化模型权重 - 源头预防数值问题
        使用Xavier初始化确保训练稳定
        ⚠️ 跳过text_encoder(BERT)，保留其预训练权重
        """
        for name, module in self.named_modules():
            # ✅ 跳过BERT预训练权重，不要重置
            if 'text_encoder' in name:
                continue
            # ✅ 跳过对比学习投影头（如果从预训练加载）
            if 'text_contrast_proj' in name:
                continue

            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.MultiheadAttention):
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in param_name:
                        nn.init.zeros_(param)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        print("✅ 模型权重初始化完成（已保护BERT预训练权重）")

    def forward(self, input_ids=None, attention_mask=None,
                node_ids_list=None, edges_list=None, node_levels_list=None,
                struct_features=None, return_attention=False):
        """
        前向传播
        Args:
            input_ids: 文本输入 [batch, seq_len]
            attention_mask: 注意力mask [batch, seq_len]
            node_ids_list: 节点ID列表 (list of lists)
            edges_list: 边列表 (list of lists)
            node_levels_list: 节点层级列表 (list of lists)
            struct_features: 结构化特征 [batch, 53]
            return_attention: 是否返回注意力权重
        Returns:
            logits: 分类logits [batch, 2]
            attention_weights: 注意力权重字典（如果return_attention=True）
        """
        attention_weights = {}

        # ========== 文本特征 ==========
        bert_hidden_states = None  # 新增：保存hidden_states用于多Token生成
        if self.text_encoder is not None and input_ids is not None:
            text_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # 保存所有层的hidden_states（用于多Token生成）
            bert_hidden_states = text_output.hidden_states  # tuple of [batch, seq, 768]

            # 原有逻辑保持（用于非full模式的兼容）
            text_feat = text_output.last_hidden_state[:, 0, :]  # [batch, 768]

            if self.text_proj is not None:
                text_feat_proj = self.text_proj(text_feat)  # [batch, 256]
                text_feat_proj = text_feat_proj.unsqueeze(1)  # [batch, 1, 256]
            else:
                raise ValueError("text_proj不应为None！")
        else:
            text_feat_proj = None
            bert_hidden_states = None

        # ========== 标签特征 ==========
        if self.use_flat_label and self.flat_label_encoder is not None and node_ids_list is not None:
            # Flat MLP编码路径（不使用图结构）
            batch_label_feats = []
            device = next(self.parameters()).device
            for node_ids in node_ids_list:
                if isinstance(node_ids, list):
                    node_ids_t = torch.tensor(node_ids, dtype=torch.long, device=device)
                else:
                    node_ids_t = node_ids.to(device)
                # 截断/填充到 max_path_len=8
                max_len = 8
                if len(node_ids_t) > max_len:
                    node_ids_t = node_ids_t[:max_len]
                elif len(node_ids_t) < max_len:
                    padding = torch.zeros(max_len - len(node_ids_t), dtype=torch.long, device=device)
                    node_ids_t = torch.cat([node_ids_t, padding])
                emb = self.flat_label_encoder[0](node_ids_t)  # [max_len, 128]
                batch_label_feats.append(emb.view(-1))  # [128*8]
            label_flat_input = torch.stack(batch_label_feats)  # [batch, 128*8]
            label_feat = self.flat_label_mlp(label_flat_input)  # [batch, 256]
        elif self.label_encoder is not None and node_ids_list is not None:
            batch_data = []
            for i in range(len(node_ids_list)):
                node_ids = torch.tensor(node_ids_list[i], dtype=torch.long, device=self.device)
                node_levels = torch.tensor(node_levels_list[i], dtype=torch.long, device=self.device)

                # 构建边
                if edges_list[i]:
                    edges = torch.tensor(edges_list[i], dtype=torch.long, device=self.device).t()
                else:
                    # 自环
                    num_nodes = len(node_ids)
                    edges = torch.tensor([[j, j] for j in range(num_nodes)], device=self.device).t()

                data = Data(
                    x=node_ids,
                    edge_index=edges,
                    node_levels=node_levels,
                    batch=torch.full((len(node_ids),), i, dtype=torch.long, device=self.device)
                )
                batch_data.append(data)

            graph_batch = Batch.from_data_list(batch_data).to(self.device)
            # 获取标签多节点特征（修复P01）
            label_feat = self.label_encoder(
                graph_batch.x,
                graph_batch.edge_index,
                graph_batch.node_levels,
                graph_batch.batch
            )  # 返回 (node_features, node_masks)
            # 注意：label_feat现在是tuple，在跨模态交互中处理
        else:
            label_feat = None

        # ========== 结构化特征 ==========
        if self.struct_encoder is not None and struct_features is not None:
            # 特征重要性加权 - 方向三
            if hasattr(self, 'feature_importance'):
                importance_weights = torch.softmax(self.feature_importance, dim=0)
                struct_features = struct_features * importance_weights

            struct_feat = self.struct_encoder(struct_features)  # [batch, 256]
            struct_feat = struct_feat.unsqueeze(1)  # [batch, 1, 256]
        else:
            struct_feat = None

        # ========== 跨模态交互 - 文本主导方案 ==========
        if self.mode == 'full':
            if text_feat_proj is not None and label_feat is not None and struct_feat is not None:
                # 处理标签返回值（现在是多节点版本）
                if isinstance(label_feat, tuple):
                    label_node_feats, label_mask = label_feat
                else:
                    # 兼容旧版本或Flat MLP编码(2D→3D)
                    label_node_feats = label_feat
                    if label_node_feats.dim() == 2:
                        label_node_feats = label_node_feats.unsqueeze(1)  # [batch, 1, 256]
                    label_mask = None

                # 生成文本多Token
                if bert_hidden_states is not None:
                    text_tokens = self.text_token_gen(bert_hidden_states)  # [batch, 4, 256]
                else:
                    # 回退：复制text_feat_proj
                    text_tokens = text_feat_proj.expand(-1, 4, -1)

                # 生成结构多Token
                struct_tokens = self.struct_token_gen(struct_features)  # [batch, 4, 256]

                # 文本主导的跨模态注意力
                text_pooled, label_pooled, struct_pooled, cross_attn = \
                    self.text_led_cross_modal(
                        text_tokens, label_node_feats, struct_tokens,
                        label_mask=label_mask,
                        return_attention=return_attention
                    )

                if return_attention and cross_attn:
                    attention_weights.update(cross_attn)

                # 拼接三个模态的池化特征
                combined_feat = torch.cat([text_pooled, label_pooled, struct_pooled], dim=-1)
            else:
                # 回退：部分模态缺失
                features = []
                if text_feat_proj is not None:
                    features.append(text_feat_proj.squeeze(1))
                if label_feat is not None:
                    if isinstance(label_feat, tuple):
                        features.append(label_feat[0].mean(dim=1))
                    else:
                        features.append(label_feat.squeeze(1))
                if struct_feat is not None:
                    features.append(struct_feat.squeeze(1))
                combined_feat = torch.cat(features, dim=-1)

        elif self.mode in ['text_label', 'text_struct', 'label_struct']:
            # 双模态交互
            feat1, feat2 = None, None

            if self.mode == 'text_label':
                feat1 = text_feat_proj
                # 修复: label_encoder返回的是元组(node_features, node_masks)
                if isinstance(label_feat, tuple):
                    node_feats, node_mask = label_feat
                    if node_mask is not None:
                        valid_mask = ~node_mask
                        mask_expanded = valid_mask.unsqueeze(-1).float()
                        label_pooled = (node_feats * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
                    else:
                        label_pooled = node_feats.mean(dim=1)
                    feat2 = label_pooled.unsqueeze(1)
                else:
                    feat2 = label_feat
                    if feat2 is not None and feat2.dim() == 2:
                        feat2 = feat2.unsqueeze(1)  # [batch, 1, 256]
            elif self.mode == 'text_struct':
                feat1, feat2 = text_feat_proj, struct_feat
            elif self.mode == 'label_struct':
                # 修复: label_encoder返回的是元组(node_features, node_masks)
                if isinstance(label_feat, tuple):
                    node_feats, node_mask = label_feat
                    if node_mask is not None:
                        valid_mask = ~node_mask
                        mask_expanded = valid_mask.unsqueeze(-1).float()
                        label_pooled = (node_feats * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
                    else:
                        label_pooled = node_feats.mean(dim=1)
                    feat1 = label_pooled.unsqueeze(1)
                else:
                    feat1 = label_feat
                    if feat1 is not None and feat1.dim() == 2:
                        feat1 = feat1.unsqueeze(1)  # [batch, 1, 256]
                feat2 = struct_feat

            if feat1 is not None and feat2 is not None:
                feat1_enhanced, attn1 = self.modal_attn_1(feat1, feat2)
                feat2_enhanced, attn2 = self.modal_attn_2(feat2, feat1)

                if return_attention:
                    attention_weights['modal1_to_modal2'] = attn1
                    attention_weights['modal2_to_modal1'] = attn2

                combined_feat = torch.cat([
                    feat1_enhanced.squeeze(1),
                    feat2_enhanced.squeeze(1)
                ], dim=-1)
            else:
                # 如果某个模态缺失，使用可用的
                available = [f for f in [feat1, feat2] if f is not None]
                if available:
                    combined_feat = torch.cat([f.squeeze(1) for f in available], dim=-1)
                else:
                    raise ValueError("至少需要一个模态的输入")

        else:
            # 单模态
            if text_feat_proj is not None:
                combined_feat = text_feat_proj.squeeze(1)
            elif label_feat is not None:
                if isinstance(label_feat, tuple):
                    node_feats, node_mask = label_feat
                    if node_mask is not None:
                        valid_mask = ~node_mask  # True表示padding，取反得到有效位置
                        mask_expanded = valid_mask.unsqueeze(-1).float()
                        combined_feat = (node_feats * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
                    else:
                        combined_feat = node_feats.mean(dim=1)
                else:
                    combined_feat = label_feat.squeeze(1)
            elif struct_feat is not None:
                combined_feat = struct_feat.squeeze(1)
            else:
                raise ValueError("至少需要一个模态的输入")

        # ========== 融合和分类 ==========
        fused_feat = self.fusion(combined_feat)  # [batch, hidden_dim]
        logits = self.classifier(fused_feat)  # [batch, 2]

        # ========== 源头预防: 统一返回格式 ==========
        if return_attention:
            # 确保attention_weights不为空
            if not attention_weights:
                attention_weights = self._get_default_attention_weights()
            return logits, attention_weights
        else:
            # 即使不需要attention，也返回None保持格式统一
            return logits, None

    def _get_default_attention_weights(self):
        """获取默认的注意力权重 - 使用有意义的初始化"""
        max_nodes = 8
        num_heads = 4
        text_seq_len = 4
        struct_seq_len = 4

        # 文本对标签: 递减权重（前面的标签节点更重要）
        t2l_weights = torch.zeros(1, num_heads, text_seq_len, max_nodes, device=self.device)
        for i in range(max_nodes):
            t2l_weights[:, :, :, i] = 1.0 / (i + 1)
        t2l_weights = t2l_weights / t2l_weights.sum(dim=-1, keepdim=True)

        # 文本对结构: 递减权重
        t2s_weights = torch.zeros(1, num_heads, text_seq_len, struct_seq_len, device=self.device)
        for i in range(struct_seq_len):
            t2s_weights[:, :, :, i] = 1.0 / (i + 1)
        t2s_weights = t2s_weights / t2s_weights.sum(dim=-1, keepdim=True)

        # 标签自注意力: 对角线强调（自注意力更强）
        l_self_weights = torch.zeros(1, num_heads, max_nodes, max_nodes, device=self.device)
        for i in range(max_nodes):
            l_self_weights[:, :, i, i] = 0.5  # 对角线权重
            for j in range(max_nodes):
                if i != j:
                    l_self_weights[:, :, i, j] = 0.5 / (max_nodes - 1)

        # 门控权重: 文本主导
        gate_weights = torch.tensor([[0.5, 0.25, 0.25]], device=self.device)

        return {
            'text_to_label': t2l_weights,
            'text_to_struct': t2s_weights,
            'label_self': l_self_weights,
            'modal_weights': gate_weights,
        }




# ============================================================
# Inlined from ablation_study.py - train_and_evaluate
# ============================================================

def taiwan_train_and_evaluate(model, train_loader, val_loader, test_loader, config, device, exp_name):
    """训练和评估模型"""

    # ✅ 修复：针对性训练策略
    # 所有包含TEXT模态的实验都使用分层学习率，保护预训练的BERT参数

    if exp_name == 'full_model':
        # full_model: 三模态完整模型 - 保持最优参数
        bert_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'text_encoder' in name:
                bert_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': 1e-5, 'weight_decay': 0.01},
            {'params': other_params, 'lr': 5e-5}
        ])
        num_epochs = 20
        print(f"✅ {exp_name}: 分层学习率(BERT=1e-5, 其他=5e-5), {num_epochs}轮")



    elif exp_name == 'text_only':

        # text_only: 冻结BERT底层0-8，保留9-11可训练

        for name, param in model.text_encoder.named_parameters():

            if 'embeddings' in name:

                param.requires_grad = False

            elif 'encoder.layer.' in name:

                layer_num = int(name.split('encoder.layer.')[1].split('.')[0])

                if layer_num <= 8:
                    param.requires_grad = False

        bert_trainable = [p for n, p in model.text_encoder.named_parameters() if p.requires_grad]

        other_params = [p for n, p in model.named_parameters() if 'text_encoder' not in n]

        optimizer = torch.optim.AdamW([

            {'params': bert_trainable, 'lr': 3e-6, 'weight_decay': 0.01},

            {'params': other_params, 'lr': 3e-5}

        ])

        num_epochs = 17

        print(f"✅ {exp_name}: 冻结BERT层0-8, 保留9-11, BERT_lr=3e-6, {num_epochs}轮")


    elif exp_name == 'text_label':

        # text_label: 冻结BERT底层0-8，保留9-11可训练

        for name, param in model.text_encoder.named_parameters():

            if 'embeddings' in name:

                param.requires_grad = False

            elif 'encoder.layer.' in name:

                layer_num = int(name.split('encoder.layer.')[1].split('.')[0])

                if layer_num <= 8:
                    param.requires_grad = False

        bert_trainable = [p for n, p in model.text_encoder.named_parameters() if p.requires_grad]

        other_params = [p for n, p in model.named_parameters() if 'text_encoder' not in n]

        optimizer = torch.optim.AdamW([

            {'params': bert_trainable, 'lr': 5e-6, 'weight_decay': 0.01},

            {'params': other_params, 'lr': 4e-5}

        ])

        num_epochs = 10

        print(f"✅ {exp_name}: 冻结BERT层0-8, 保留9-11, BERT_lr=5e-6, {num_epochs}轮")


    elif exp_name == 'text_struct':

        # text_struct: 冻结BERT底层0-8，保留9-11可训练

        for name, param in model.text_encoder.named_parameters():

            if 'embeddings' in name:

                param.requires_grad = False

            elif 'encoder.layer.' in name:

                layer_num = int(name.split('encoder.layer.')[1].split('.')[0])

                if layer_num <= 8:
                    param.requires_grad = False

        bert_trainable = [p for n, p in model.text_encoder.named_parameters() if p.requires_grad]

        other_params = [p for n, p in model.named_parameters() if 'text_encoder' not in n]

        optimizer = torch.optim.AdamW([

            {'params': bert_trainable, 'lr': 5e-6, 'weight_decay': 0.01},

            {'params': other_params, 'lr': 4e-5}

        ])

        num_epochs = 10

        print(f"✅ {exp_name}: 冻结BERT层0-8, 保留9-11, BERT_lr=5e-6, {num_epochs}轮")


    elif exp_name == 'label_only':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        num_epochs = 20
        print(f"✅ {exp_name}: 学习率1e-4, {num_epochs}轮")


    elif exp_name == 'struct_only':

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        num_epochs = 25

        print(f"✅ {exp_name}: 学习率1e-4, {num_epochs}轮")


    elif exp_name == 'label_struct':

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        num_epochs = 15

        print(f"✅ {exp_name}: 学习率5e-5, {num_epochs}轮")
        print(f"✅ {exp_name}: 学习率5e-5, {num_epochs}轮")

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
        num_epochs = config.training.num_epochs

    _cw = getattr(config.training, 'class_weight', None)
    if _cw is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(_cw, dtype=torch.float32).to(device))
        print(f"  使用类别加权损失: {_cw}")
    else:
        criterion = nn.CrossEntropyLoss()
    best_val_auc = 0
    best_state = None
    patience = 0
    max_patience = 5

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            node_ids_list = batch['node_ids']
            edges_list = batch['edges']
            node_levels_list = batch['node_levels']
            struct_features = batch['struct_features'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()

            # 根据实验类型选择输入
            if exp_name == 'text_only':
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            elif exp_name == 'label_only':
                logits, _ = model(node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list)
            elif exp_name == 'struct_only':
                logits, _ = model(struct_features=struct_features)
            elif exp_name == 'text_label':
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                             node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list)
            elif exp_name == 'text_struct':
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                             struct_features=struct_features)
            elif exp_name == 'label_struct':
                logits, _ = model(node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list, struct_features=struct_features)
            else:  # full_model, No_pretrain
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                             node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list, struct_features=struct_features)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_preds = []
        val_targets = []
        val_probs = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                node_ids_list = batch['node_ids']
                edges_list = batch['edges']
                node_levels_list = batch['node_levels']
                struct_features = batch['struct_features'].to(device)
                targets = batch['target'].to(device)

                # 根据实验类型选择输入
                if exp_name == 'text_only':
                    # ✅ 正确代码
                    logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                elif exp_name == 'label_only':
                    logits, _ = model(node_ids_list=node_ids_list, edges_list=edges_list,
                                 node_levels_list=node_levels_list)
                elif exp_name == 'struct_only':
                    logits, _ = model(struct_features=struct_features)
                elif exp_name == 'text_label':
                    logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                                 node_ids_list=node_ids_list, edges_list=edges_list,
                                 node_levels_list=node_levels_list)
                elif exp_name == 'text_struct':
                    logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                                 struct_features=struct_features)
                elif exp_name == 'label_struct':
                    logits, _ = model(node_ids_list=node_ids_list, edges_list=edges_list,
                                 node_levels_list=node_levels_list, struct_features=struct_features)
                else:
                    logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                                 node_ids_list=node_ids_list, edges_list=edges_list,
                                 node_levels_list=node_levels_list, struct_features=struct_features)

                probs = torch.softmax(logits, dim=1)
                val_probs.extend(probs[:, 1].cpu().numpy())
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_auc = roc_auc_score(val_targets, val_probs)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val AUC={val_auc:.4f}")

        # 早停
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"早停触发，最佳验证AUC: {best_val_auc:.4f}")
                break

        # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        print(f"✅ 已恢复最佳模型 (Val AUC: {best_val_auc:.4f})")

    # 测试
    model.eval()
    test_preds = []
    test_targets = []
    test_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            node_ids_list = batch['node_ids']
            edges_list = batch['edges']
            node_levels_list = batch['node_levels']
            struct_features = batch['struct_features'].to(device)
            targets = batch['target'].to(device)

            # 根据实验类型选择输入
            if exp_name == 'text_only':
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            elif exp_name == 'label_only':
                logits, _ = model(node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list)
            elif exp_name == 'struct_only':
                logits, _ = model(struct_features=struct_features)
            elif exp_name == 'text_label':
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                             node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list)
            elif exp_name == 'text_struct':
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                             struct_features=struct_features)
            elif exp_name == 'label_struct':
                logits, _ = model(node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list, struct_features=struct_features)
            else:
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                             node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list, struct_features=struct_features)

            probs = torch.softmax(logits, dim=1)
            test_probs.extend(probs[:, 1].cpu().numpy())
            test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_targets.extend(targets.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(test_targets, test_preds)
    precision = precision_score(test_targets, test_preds, zero_division=0)
    recall = recall_score(test_targets, test_preds, zero_division=0)
    f1 = f1_score(test_targets, test_preds, average='macro')
    auc = roc_auc_score(test_targets, test_probs)

    return accuracy, precision, recall, f1, auc


# ============================================================
# 共享的Dataset和Model类（与baseline_all_methods.py完全一致）
# ============================================================

class BaselineDataset(Dataset):
    def __init__(self, texts, struct_features, targets, tokenizer, max_length=256):
        self.texts = texts
        self.struct_features = struct_features
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]), max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'struct_features': torch.tensor(self.struct_features[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.long)
        }

class SimpleLabelProcessor:
    def __init__(self):
        self.node_to_id = {'[PAD]': 0, '[UNK]': 1}
    def build_vocab(self, labels):
        for label in labels:
            label_str = str(label).strip()
            if not label_str or label_str.lower() in ['nan', 'none', '']:
                continue
            if '→' in label_str:
                parts = label_str.split('→')
            elif '->' in label_str:
                parts = label_str.split('->')
            else:
                parts = [label_str]
            cumulative = ''
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                cumulative = f"{cumulative}→{part}" if cumulative else part
                if cumulative not in self.node_to_id:
                    self.node_to_id[cumulative] = len(self.node_to_id)
        print(f"  标签词表大小: {len(self.node_to_id)}")
    def encode_label_path_as_graph(self, label):
        label_str = str(label).strip()
        if not label_str or label_str.lower() in ['nan', 'none', '']:
            return [0], [], [0]
        if '→' in label_str:
            parts = label_str.split('→')
        elif '->' in label_str:
            parts = label_str.split('->')
        else:
            parts = [label_str]
        node_ids, node_levels = [], []
        cumulative = ''
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            cumulative = f"{cumulative}→{part}" if cumulative else part
            nid = self.node_to_id.get(cumulative, 1)
            node_ids.append(nid)
            node_levels.append(i)
        if not node_ids:
            return [0], [], [0]
        edges = []
        for i in range(len(node_ids) - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        return node_ids, edges, node_levels

class FullModalDataset(Dataset):
    def __init__(self, texts, struct_features, targets, tokenizer,
                 labels=None, processor=None, max_length=256):
        self.texts = texts
        self.struct_features = struct_features
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels if labels is not None else [''] * len(texts)
        self.processor = processor
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]), max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'struct_features': torch.tensor(self.struct_features[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.long)
        }
        label = self.labels[idx] if idx < len(self.labels) else ''
        if label and self.processor and hasattr(self.processor, 'node_to_id') and self.processor.node_to_id:
            node_ids, edges, node_levels = self.processor.encode_label_path_as_graph(label)
        else:
            node_ids, edges, node_levels = [0], [], [0]
        item['node_ids'] = node_ids
        item['edges'] = edges
        item['node_levels'] = node_levels
        return item

def full_modal_collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'struct_features': torch.stack([item['struct_features'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch]),
        'node_ids': [item['node_ids'] for item in batch],
        'edges': [item['edges'] for item in batch],
        'node_levels': [item['node_levels'] for item in batch],
    }

# ============================================================
# DL Models（与baseline_all_methods.py完全一致）
# ============================================================

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, num_filters=50,
                 filter_sizes=[2,3,4,5], num_classes=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, fs) for fs in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    def forward(self, input_ids, attention_mask=None, struct_features=None):
        x = self.embedding(input_ids).transpose(1, 2)
        x = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(x, dim=1)
        return self.fc(self.dropout(x))

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=256,
                 num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, input_ids, attention_mask=None, struct_features=None):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        attn = torch.softmax(self.attention(x).squeeze(-1), dim=1)
        if attention_mask is not None:
            attn = attn * attention_mask.float()
            attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        return self.fc(self.dropout(x))

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )
    def forward(self, input_ids, attention_mask=None, struct_features=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

class BERTStructClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', struct_dim=53, num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size
        self.struct_encoder = nn.Sequential(
            nn.Linear(struct_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 256)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(bert_hidden + 256, 256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )
    def forward(self, input_ids, attention_mask=None, struct_features=None):
        text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        if struct_features is not None:
            struct_feat = self.struct_encoder(struct_features)
            combined = torch.cat([text_feat, struct_feat], dim=-1)
        else:
            combined = text_feat
        return self.classifier(combined)

class EarlyFusionModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', struct_dim=53, num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size
        self.struct_encoder = nn.Sequential(
            nn.Linear(struct_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 256)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(bert_hidden + 256, 256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )
    def forward(self, input_ids, attention_mask=None, struct_features=None):
        text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        struct_feat = self.struct_encoder(struct_features) if struct_features is not None else torch.zeros(text_feat.size(0), 256, device=text_feat.device)
        return self.classifier(torch.cat([text_feat, struct_feat], dim=-1))

class LateFusionModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', struct_dim=53, num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size
        self.text_classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(bert_hidden, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes)
        )
        self.struct_classifier = nn.Sequential(
            nn.Linear(struct_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes)
        )
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
    def forward(self, input_ids, attention_mask=None, struct_features=None):
        text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_logits = self.text_classifier(text_feat)
        if struct_features is not None:
            struct_logits = self.struct_classifier(struct_features)
            w = torch.sigmoid(self.fusion_weight)
            text_prob = F.softmax(text_logits, dim=-1)
            struct_prob = F.softmax(struct_logits, dim=-1)
            combined = w * text_prob + (1 - w) * struct_prob
            return torch.log(combined + 1e-8)
        return text_logits

class AttentionFusionModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', struct_dim=53, num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size
        self.struct_encoder = nn.Sequential(
            nn.Linear(struct_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, bert_hidden)
        )
        self.attention = nn.MultiheadAttention(bert_hidden, num_heads=4, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(bert_hidden * 2, bert_hidden), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(bert_hidden, 256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )
    def forward(self, input_ids, attention_mask=None, struct_features=None):
        text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        if struct_features is not None:
            struct_feat = self.struct_encoder(struct_features)
            # 拼成2-token序列: [text, struct]
            seq = torch.stack([text_feat, struct_feat], dim=1)  # (B, 2, 768)
            attn_out, _ = self.attention(seq, seq, seq)         # 自注意力
            attn_text = attn_out[:, 0, :]                       # 取text位置的输出
            # 门控残差：保留原始text信息
            g = self.gate(torch.cat([text_feat, attn_text], dim=-1))
            fused = g * text_feat + (1 - g) * attn_text
        else:
            fused = text_feat
        return self.classifier(fused)

class TextLabelEarlyFusion(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', vocab_size=1000, num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size
        self.label_embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.label_encoder = nn.Sequential(nn.Linear(128 * 8, 256), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(bert_hidden + 256, 256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )
    def forward(self, input_ids, attention_mask=None, label_ids=None, **kwargs):
        text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        if label_ids is not None:
            label_emb = self.label_embedding(label_ids)
            label_feat = self.label_encoder(label_emb.view(label_emb.size(0), -1))
            combined = torch.cat([text_feat, label_feat], dim=-1)
        else:
            combined = torch.cat([text_feat, torch.zeros(text_feat.size(0), 256, device=text_feat.device)], dim=-1)
        return self.classifier(combined)

class ProfileLabelEarlyFusion(nn.Module):
    def __init__(self, struct_dim=53, vocab_size=1000, num_classes=2, dropout=0.3):
        super().__init__()
        self.profile_encoder = nn.Sequential(
            nn.Linear(struct_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 256)
        )
        self.label_embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.label_encoder = nn.Sequential(nn.Linear(128 * 8, 256), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )
    def forward(self, struct_features, label_ids=None, **kwargs):
        profile_feat = self.profile_encoder(struct_features)
        if label_ids is not None:
            label_emb = self.label_embedding(label_ids)
            label_feat = self.label_encoder(label_emb.view(label_emb.size(0), -1))
        else:
            label_feat = torch.zeros(profile_feat.size(0), 256, device=profile_feat.device)
        combined = torch.cat([profile_feat, label_feat], dim=-1)
        return self.classifier(combined)

# ============================================================
# Training/Eval（与baseline_all_methods.py完全一致）
# ============================================================
def train_dl_model(model, train_loader, val_loader, device, num_epochs=10, lr=5e-4, model_type='bert', class_weights=None):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr if 'bert' in model_type.lower() else lr * 5)
    if class_weights is not None:
        weight_tensor = torch.tensor([class_weights.get(i, 1.0) for i in range(max(class_weights.keys()) + 1)], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    best_auc, best_state = 0, None
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            optimizer.zero_grad()
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['struct_features'].to(device))
            loss = criterion(logits, batch['target'].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        preds, targets, probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['struct_features'].to(device))
                p = F.softmax(logits, dim=-1)
                preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                targets.extend(batch['target'].numpy())
                probs.extend(p[:, 1].cpu().numpy())
        val_auc = roc_auc_score(targets, probs)
        scheduler.step(val_auc)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
    if best_state:
        model.load_state_dict(best_state)
    return model, best_auc

def evaluate_model_fn(model, test_loader, device):
    model.eval()
    preds, targets, probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['struct_features'].to(device))
            p = F.softmax(logits, dim=-1)
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            targets.extend(batch['target'].numpy())
            probs.extend(p[:, 1].cpu().numpy())
    fpr, tpr, _ = roc_curve(targets, probs)
    cm = confusion_matrix(targets, preds)
    return {
        'accuracy': accuracy_score(targets, preds),
        'precision': precision_score(targets, preds, zero_division=0),
        'recall': recall_score(targets, preds, zero_division=0),
        'f1': f1_score(targets, preds, average='weighted'),
        'auc': roc_auc_score(targets, probs),
        'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
        'confusion_matrix': cm.tolist(),
        'predictions': preds, 'targets': targets, 'probs': probs
    }

# ============================================================
# Taiwan Baseline Experiment
# ============================================================

class TaiwanBaselineExperiment:
    def __init__(self, data_file):
        self.config = Config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {}
        self.dataset_name = 'taiwan'
        self.has_struct = True
        self.class_weights = None
        self.bert_model_name = 'bert-base-chinese'
        print(f"📂 Loading data: {data_file}")
        self.df = pd.read_excel(data_file) if data_file.endswith('.xlsx') else pd.read_csv(data_file)
        self._prepare_data()

    def _prepare_data(self):
        target_col = None
        for col in ['Repeat complaint', 'satisfaction_binary', 'disputed']:
            if col in self.df.columns:
                target_col = col
                break
        if target_col is None:
            raise ValueError("未找到目标变量列")
        print(f"  目标变量: {target_col}")
        text_col = 'biz_cntt' if 'biz_cntt' in self.df.columns else self.df.columns[0]
        self.texts = self.df[text_col].fillna('').astype(str).tolist()
        label_col = None
        for col in ['Complaint label', 'Complaint_label']:
            if col in self.df.columns:
                label_col = col
                break
        self.labels = self.df[label_col].fillna('').astype(str).tolist() if label_col else [''] * len(self.texts)
        self._extract_struct_features(target_col)
        pos_count = (self.df[target_col] == 1).sum()
        neg_count = (self.df[target_col] == 0).sum()
        total = len(self.df)
        if pos_count > 0 and neg_count > 0 and abs(pos_count - neg_count) / total > 0.2:
            self.class_weights = {0: total / (2 * neg_count), 1: total / (2 * pos_count)}
            print(f"  类别不平衡: 0={neg_count}, 1={pos_count}, 权重={self.class_weights}")
        self.targets = self.df[target_col].values
        self.X_train_idx, self.X_test_idx = train_test_split(
            range(len(self.texts)), test_size=0.2, random_state=42, stratify=self.targets)
        self.vectorizer = TfidfVectorizer(max_features=300)
        train_texts = [self.texts[i] for i in self.X_train_idx]
        test_texts = [self.texts[i] for i in self.X_test_idx]
        self.X_train_tfidf = self.vectorizer.fit_transform(train_texts)
        self.X_test_tfidf = self.vectorizer.transform(test_texts)
        if self.has_struct:
            self.scaler = StandardScaler()
            self.X_train_struct = self.scaler.fit_transform([self.struct_features[i] for i in self.X_train_idx])
            self.X_test_struct = self.scaler.transform([self.struct_features[i] for i in self.X_test_idx])
        else:
            self.X_train_struct = np.zeros((len(self.X_train_idx), 1))
            self.X_test_struct = np.zeros((len(self.X_test_idx), 1))
        self.y_train = self.targets[self.X_train_idx]
        self.y_test = self.targets[self.X_test_idx]
        print(f"  Loading BERT tokenizer: {self.bert_model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        train_dataset = BaselineDataset([self.texts[i] for i in self.X_train_idx], self.X_train_struct, self.y_train, self.tokenizer)
        test_dataset = BaselineDataset([self.texts[i] for i in self.X_test_idx], self.X_test_struct, self.y_test, self.tokenizer)
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        print(f"  训练集: {len(self.X_train_idx)}, 测试集: {len(self.X_test_idx)}, struct_dim: {self.struct_dim}")

    def _extract_struct_features(self, target_col):
        col_names = self.df.columns.tolist()
        exclude_cols = {'new_code', 'biz_cntt', 'Complaint label', 'Complaint_label',
                        'Repeat complaint', 'satisfaction_binary', 'disputed', 'is_synthetic'}
        if 'Complaint label' in col_names or 'Complaint_label' in col_names:
            label_col = 'Complaint label' if 'Complaint label' in col_names else 'Complaint_label'
            target_idx = col_names.index(target_col)
            label_idx = col_names.index(label_col)
            struct_cols = col_names[label_idx + 1: target_idx]
        else:
            struct_cols = [col for col in col_names if col not in exclude_cols and self.df[col].dtype in ['int64', 'float64']]
        struct_cols = [col for col in struct_cols if col not in exclude_cols]
        if not struct_cols:
            self.struct_features = [[0.0] for _ in range(len(self.df))]
            self.struct_dim = 0
            self.has_struct = False
            return
        self.struct_dim = len(struct_cols)
        print(f"  结构化特征列: {struct_cols} (共{len(struct_cols)}列)")
        self.struct_features = []
        for _, row in self.df.iterrows():
            features = []
            for col in struct_cols:
                try:
                    val = pd.to_numeric(row[col], errors='coerce')
                    features.append(0.0 if pd.isna(val) else float(val))
                except Exception:
                    features.append(0.0)
            self.struct_features.append(features)

    def _m(self, y_true, y_pred, y_prob):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0), 'f1': f1_score(y_true, y_pred, average='weighted'),
                'auc': roc_auc_score(y_true, y_prob), 'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

    # ===== Layer 1: Text Unimodal =====
    def run_tfidf_lr(self):
        print("\n▶️ TF-IDF + LR...")
        m = LogisticRegression(max_iter=500, random_state=42, C=0.1,
                               class_weight='balanced' if self.class_weights else None)
        m.fit(self.X_train_tfidf, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_tfidf), m.predict_proba(self.X_test_tfidf)[:, 1])
        self.results['TF-IDF + LR'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_tfidf_rf(self):
        print("\n▶️ TF-IDF + RF...")
        m = RandomForestClassifier(n_estimators=20, max_depth=20, random_state=42, class_weight='balanced' if self.class_weights else None)
        m.fit(self.X_train_tfidf, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_tfidf), m.predict_proba(self.X_test_tfidf)[:, 1])
        self.results['TF-IDF + RF'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_tfidf_gbdt(self):
        print("\n▶️ TF-IDF + GBDT...")
        m = GradientBoostingClassifier(n_estimators=20, max_depth=10, random_state=42)
        m.fit(self.X_train_tfidf.toarray(), self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_tfidf.toarray()), m.predict_proba(self.X_test_tfidf.toarray())[:, 1])
        self.results['TF-IDF + GBDT'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_tfidf_svm(self):
        print("\n▶️ TF-IDF + SVM...")
        m = SVC(kernel='rbf', probability=True, random_state=42, C=0.01, class_weight='balanced' if self.class_weights else None)
        m.fit(self.X_train_tfidf, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_tfidf), m.predict_proba(self.X_test_tfidf)[:, 1])
        self.results['TF-IDF + SVM'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_tfidf_xgboost(self):
        if not XGBOOST_AVAILABLE: return
        print("\n▶️ TF-IDF + XGBoost...")
        m = xgb.XGBClassifier(n_estimators=20, max_depth=20, random_state=42, learning_rate=0.01, use_label_encoder=False, eval_metric='logloss',
                               scale_pos_weight=(self.class_weights[1] / self.class_weights[0]) if self.class_weights else 1.0)
        m.fit(self.X_train_tfidf, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_tfidf), m.predict_proba(self.X_test_tfidf)[:, 1])
        self.results['TF-IDF + XGBoost'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    # ===== Layer 2: Profile Unimodal =====
    def run_struct_lr(self):
        print("\n▶️ Struct + LR...")
        m = LogisticRegression(max_iter=50, random_state=42, C=0.01, class_weight='balanced' if self.class_weights else None)
        m.fit(self.X_train_struct, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_struct), m.predict_proba(self.X_test_struct)[:, 1])
        self.results['Struct + LR'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_struct_rf(self):
        print("\n▶️ Struct + RF...")
        m = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, class_weight='balanced' if self.class_weights else None)
        m.fit(self.X_train_struct, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_struct), m.predict_proba(self.X_test_struct)[:, 1])
        self.results['Struct + RF'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_struct_gbdt(self):
        print("\n▶️ Struct + GBDT...")
        m = GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=42)
        m.fit(self.X_train_struct, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_struct), m.predict_proba(self.X_test_struct)[:, 1])
        self.results['Struct + GBDT'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_struct_xgboost(self):
        if not XGBOOST_AVAILABLE: return
        print("\n▶️ Struct + XGBoost...")
        m = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42, learning_rate=0.05, subsample=0.8,
                               use_label_encoder=False, eval_metric='logloss',
                               scale_pos_weight=(self.class_weights[1] / self.class_weights[0]) if self.class_weights else 1.0)
        m.fit(self.X_train_struct, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_struct), m.predict_proba(self.X_test_struct)[:, 1])
        self.results['Struct + XGBoost'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_struct_mlp(self):
        print("\n▶️ Struct + MLP...")
        m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
        m.fit(self.X_train_struct, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_struct), m.predict_proba(self.X_test_struct)[:, 1])
        self.results['Struct + MLP'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    # ===== Layer 3: Label =====
    def run_label_mlp(self, num_epochs=1):
        print("\n▶️ Label + MLP (Flat)...")
        # ✅ 修复: 对拆分后的每一层分别fit，而非完整路径字符串
        all_parts = set()
        for l in self.labels:
            if l:
                parts = [p.strip() for p in l.split('→') if p.strip()]
                all_parts.update(parts)
        if not all_parts: return
        le = LabelEncoder()
        le.fit(list(all_parts))
        n = len(le.classes_)

        def enc(s, md=4):
            if not s: return np.zeros(min(n, 200) * md)
            parts = s.split('→') if '→' in s else (s.split('->') if '->' in s else [s])
            parts = [p.strip() for p in parts if p.strip()][:md]
            fd = min(n, 200);
            f = np.zeros(fd * md)
            for i, p in enumerate(parts):
                if p in le.classes_: f[i * fd + le.transform([p])[0] % fd] = 1.0
            return f

        Xtr = np.array([enc(self.labels[i]) for i in self.X_train_idx])
        Xte = np.array([enc(self.labels[i]) for i in self.X_test_idx])
        m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
        m.fit(Xtr, self.y_train)
        metrics = self._m(self.y_test, m.predict(Xte), m.predict_proba(Xte)[:, 1])
        self.results['Label + MLP (Flat)'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_label_gat(self, num_epochs=1):
        print("\n▶️ Label + GAT (Graph)...")
        # ✅ 修复: 对拆分后的每一层分别fit，而非完整路径字符串
        all_parts = set()
        for l in self.labels:
            if l:
                parts = [p.strip() for p in l.split('→') if p.strip()]
                all_parts.update(parts)
        if not all_parts: return
        le = LabelEncoder()
        le.fit(list(all_parts))
        def enc(s, md=4):
            parts = s.split('→') if '→' in s else (s.split('->') if '->' in s else [s])
            r = []
            for p in parts[:md]:
                p = p.strip()
                r.append(le.transform([p])[0] if p in le.classes_ else 0)
            while len(r) < md: r.append(0)
            return r
        Xtr = np.array([enc(self.labels[i]) for i in self.X_train_idx])
        Xte = np.array([enc(self.labels[i]) for i in self.X_test_idx])
        m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
        m.fit(Xtr, self.y_train)
        metrics = self._m(self.y_test, m.predict(Xte), m.predict_proba(Xte)[:, 1])
        self.results['Label + GAT (Graph)'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    # ===== DL Models =====
    def run_textcnn(self, ep=2):
        print("\n▶️ TextCNN...")
        model = TextCNN(vocab_size=self.tokenizer.vocab_size)
        model, _ = train_dl_model(model, self.train_loader, self.test_loader, self.device, ep, lr=1e-5, model_type='textcnn', class_weights=self.class_weights)
        self.results['TextCNN'] = evaluate_model_fn(model, self.test_loader, self.device)
        print(f"  AUC: {self.results['TextCNN']['auc']:.4f}")

    def run_bilstm(self, ep=5):
        print("\n▶️ BiLSTM...")
        model = BiLSTM(vocab_size=self.tokenizer.vocab_size)
        model, _ = train_dl_model(model, self.train_loader, self.test_loader, self.device, ep, lr=1e-5, model_type='bilstm', class_weights=self.class_weights)
        self.results['BiLSTM'] = evaluate_model_fn(model, self.test_loader, self.device)
        print(f"  AUC: {self.results['BiLSTM']['auc']:.4f}")

    def run_bert_base(self, ep=3, lr=1.4e-5, freeze_layers=11):
        print(f"\n▶️ BERT-base (ep={ep}, lr={lr}, freeze>={freeze_layers})...")
        model = BERTClassifier(bert_model_name=self.bert_model_name)
        for name, param in model.bert.named_parameters():
            if 'encoder.layer.' in name:
                layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                param.requires_grad = (layer_num >= freeze_layers)
            elif 'pooler' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        model, _ = train_dl_model(model, self.train_loader, self.test_loader,
                                  self.device, ep, lr=lr,
                                  model_type='bert', class_weights=self.class_weights)
        self.results['BERT-base'] = evaluate_model_fn(model, self.test_loader, self.device)
        print(f"  AUC: {self.results['BERT-base']['auc']:.4f}")

    def run_bert_struct(self, ep=10):
        print("\n▶️ BERT + Struct...")
        model = BERTStructClassifier(bert_model_name=self.bert_model_name, struct_dim=self.struct_dim)
        model, _ = train_dl_model(model, self.train_loader, self.test_loader, self.device, ep, model_type='bert', class_weights=self.class_weights)
        self.results['BERT + Struct'] = evaluate_model_fn(model, self.test_loader, self.device)
        print(f"  AUC: {self.results['BERT + Struct']['auc']:.4f}")

    def run_early_fusion(self, ep=3, lr=1e-5, freeze_layers=11):
        print(f"\n▶️ Early Fusion (ep={ep}, lr={lr}, freeze>={freeze_layers})...")
        model = EarlyFusionModel(bert_model_name=self.bert_model_name, struct_dim=self.struct_dim)
        for name, param in model.bert.named_parameters():
            if 'encoder.layer.' in name:
                layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                param.requires_grad = (layer_num >= freeze_layers)
            elif 'pooler' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        model, _ = train_dl_model(model, self.train_loader, self.test_loader, self.device, ep, lr=lr, model_type='bert', class_weights=self.class_weights)
        self.results['Early Fusion'] = evaluate_model_fn(model, self.test_loader, self.device)
        print(f"  AUC: {self.results['Early Fusion']['auc']:.4f}")

    def run_late_fusion(self, ep=3, lr=1.4e-5, freeze_layers=11):
        print(f"\n▶️ Late Fusion (ep={ep}, lr={lr}, freeze>={freeze_layers})...")
        model = LateFusionModel(bert_model_name=self.bert_model_name, struct_dim=self.struct_dim)
        for name, param in model.bert.named_parameters():
            if 'encoder.layer.' in name:
                layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                param.requires_grad = (layer_num >= freeze_layers)
            elif 'pooler' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        model, _ = train_dl_model(model, self.train_loader, self.test_loader, self.device, ep, lr=lr, model_type='bert', class_weights=self.class_weights)
        self.results['Late Fusion'] = evaluate_model_fn(model, self.test_loader, self.device)
        print(f"  AUC: {self.results['Late Fusion']['auc']:.4f}")

    def run_attention_fusion(self, ep=3, lr=1.4e-5, freeze_layers=11):
        print(f"\n▶️ Attention Fusion (ep={ep}, lr={lr}, freeze>={freeze_layers})...")
        model = AttentionFusionModel(bert_model_name=self.bert_model_name, struct_dim=self.struct_dim)
        for name, param in model.bert.named_parameters():
            if 'encoder.layer.' in name:
                layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                param.requires_grad = (layer_num >= freeze_layers)
            elif 'pooler' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        model, _ = train_dl_model(model, self.train_loader, self.test_loader, self.device, ep, lr=lr, model_type='bert', class_weights=self.class_weights)
        self.results['Attention Fusion'] = evaluate_model_fn(model, self.test_loader, self.device)
        print(f"  AUC: {self.results['Attention Fusion']['auc']:.4f}")

    # ===== Text+Label, Profile+Label (Layer 4) =====
    def run_text_label(self, num_epochs=1):
        print("\n▶️ Text + Label...")
        le = LabelEncoder()
        all_labels = [l for l in self.labels if l]
        if not all_labels: return
        # ✅ 修复: 对拆分后的每一层分别fit
        all_parts = set()
        for l in all_labels:
            parts = l.split('→') if '→' in l else (l.split('->') if '->' in l else [l])
            for p in parts:
                p = p.strip()
                if p:
                    all_parts.add(p)
        le.fit(list(all_parts))
        n_classes = len(le.classes_)
        def enc_path(s, mx=8):
            parts = s.split('→') if '→' in s else (s.split('->') if '->' in s else [s])
            parts = [p.strip() for p in parts if p.strip()][:mx]
            ids = [le.transform([p])[0]+1 if p in le.classes_ else 0 for p in parts]
            while len(ids) < mx: ids.append(0)
            return ids
        tr_ids = torch.tensor([enc_path(self.labels[i]) for i in self.X_train_idx], dtype=torch.long)
        te_ids = torch.tensor([enc_path(self.labels[i]) for i in self.X_test_idx], dtype=torch.long)
        model = TextLabelEarlyFusion(bert_model_name=self.bert_model_name, vocab_size=n_classes+1).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-7)
        crit = nn.CrossEntropyLoss()
        bs = 16
        for ep in range(num_epochs):
            model.train()
            for i in range(0, len(self.X_train_idx), bs):
                idx = list(range(i, min(i+bs, len(self.X_train_idx))))
                texts = [self.texts[self.X_train_idx[j]] for j in idx]
                enc = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
                opt.zero_grad()
                logits = model(enc['input_ids'].to(self.device), enc['attention_mask'].to(self.device), tr_ids[idx].to(self.device))
                loss = crit(logits, torch.tensor([self.y_train[j] for j in idx], device=self.device))
                loss.backward(); opt.step()
        model.eval(); all_probs = []
        with torch.no_grad():
            for i in range(0, len(self.X_test_idx), bs):
                idx = list(range(i, min(i+bs, len(self.X_test_idx))))
                texts = [self.texts[self.X_test_idx[j]] for j in idx]
                enc = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
                logits = model(enc['input_ids'].to(self.device), enc['attention_mask'].to(self.device), te_ids[i:i+bs].to(self.device))
                all_probs.extend(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        preds = [1 if p > 0.5 else 0 for p in all_probs]
        fpr, tpr, _ = roc_curve(self.y_test, all_probs)
        self.results['Text + Label'] = {'accuracy': accuracy_score(self.y_test, preds), 'precision': precision_score(self.y_test, preds, zero_division=0),
            'recall': recall_score(self.y_test, preds, zero_division=0), 'f1': f1_score(self.y_test, preds, average='weighted'),
            'auc': roc_auc_score(self.y_test, all_probs), 'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        print(f"  AUC: {self.results['Text + Label']['auc']:.4f}")

    def run_profile_label(self, num_epochs=10):
        print("\n▶️ Profile + Label...")
        le = LabelEncoder()
        all_labels = [l for l in self.labels if l]
        if not all_labels: return
        # ✅ 修复: 对拆分后的每一层分别fit
        all_parts = set()
        for l in all_labels:
            parts = l.split('→') if '→' in l else (l.split('->') if '->' in l else [l])
            for p in parts:
                p = p.strip()
                if p:
                    all_parts.add(p)
        le.fit(list(all_parts))
        n_classes = len(le.classes_)
        def enc_path(s, mx=8):
            parts = s.split('→') if '→' in s else (s.split('->') if '->' in s else [s])
            parts = [p.strip() for p in parts if p.strip()][:mx]
            ids = [le.transform([p])[0]+1 if p in le.classes_ else 0 for p in parts]
            while len(ids) < mx: ids.append(0)
            return ids
        tr_ids = torch.tensor([enc_path(self.labels[i]) for i in self.X_train_idx], dtype=torch.long)
        te_ids = torch.tensor([enc_path(self.labels[i]) for i in self.X_test_idx], dtype=torch.long)
        tr_struct = torch.tensor(self.X_train_struct, dtype=torch.float32)
        te_struct = torch.tensor(self.X_test_struct, dtype=torch.float32)
        model = ProfileLabelEarlyFusion(struct_dim=self.struct_dim, vocab_size=n_classes+1).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        bs = 32
        for ep in range(num_epochs):
            model.train()
            indices = np.random.permutation(len(self.X_train_idx))
            for i in range(0, len(indices), bs):
                idx = indices[i:i+bs]
                opt.zero_grad()
                logits = model(tr_struct[idx].to(self.device), tr_ids[idx].to(self.device))
                loss = crit(logits, torch.tensor([self.y_train[j] for j in idx], device=self.device))
                loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(te_struct.to(self.device), te_ids.to(self.device))
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds = [1 if p > 0.5 else 0 for p in probs]
        fpr, tpr, _ = roc_curve(self.y_test, probs)
        self.results['Profile + Label'] = {'accuracy': accuracy_score(self.y_test, preds), 'precision': precision_score(self.y_test, preds, zero_division=0),
            'recall': recall_score(self.y_test, preds, zero_division=0), 'f1': f1_score(self.y_test, preds, average='weighted'),
            'auc': roc_auc_score(self.y_test, probs), 'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        print(f"  AUC: {self.results['Profile + Label']['auc']:.4f}")

    # ===== TM-CRPP (Ours) =====
    def _run_tmcrpp_real(self, num_epochs=10):
        """运行TM-CRPP - 使用ablation管道（与主文件baseline_all_methods.py完全一致）"""
        print("\n▶️ TM-CRPP (Ours) - Using ablation pipeline for consistency...")
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        try:
            # ✅ Using inlined MultiModalComplaintModel
            # ✅ Using inlined ComplaintDataProcessor, ComplaintDataset, custom_collate_fn
            # ✅ Using inlined taiwan_train_and_evaluate
            import tempfile

            model_config = get_taiwan_restaurant_config()
            model_config.model.bert_model_name = self.bert_model_name
            model_config.training.batch_size = 16

            mode = 'full'
            model_name = 'Ours (TM-CRPP)'

            # 列名适配（与主文件完全一致）
            data_file = model_config.training.data_file
            temp_file = None
            if os.path.exists(data_file):
                df_tmp = pd.read_excel(data_file)
                rename_map = {}
                if 'Complaint_label' in df_tmp.columns and 'Complaint label' not in df_tmp.columns:
                    rename_map['Complaint_label'] = 'Complaint label'
                if 'satisfaction_binary' in df_tmp.columns and 'Repeat complaint' not in df_tmp.columns:
                    rename_map['satisfaction_binary'] = 'Repeat complaint'
                if rename_map:
                    df_tmp = df_tmp.rename(columns=rename_map)
                    temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
                    df_tmp.to_excel(temp_file.name, index=False)
                    data_file = temp_file.name
                    print(f"  台湾数据集列名适配: {rename_map}")

            if not os.path.exists(data_file):
                print(f"  ⚠️ 数据文件不存在: {data_file}")
                return

            # 创建processor（与ablation一致）
            processor = ComplaintDataProcessor(
                config=model_config,
                user_dict_file=model_config.data.user_dict_file
            )

            # 准备数据（与ablation一致）
            data = processor.prepare_datasets(train_file=data_file, for_pretrain=False)
            vocab_size = data.get('vocab_size', len(processor.node_to_id) + 1)

            # 数据划分（与ablation完全一致: seed=42, 60/20/20）
            total_size = len(data['targets'])
            torch.manual_seed(42)
            indices = torch.randperm(total_size).tolist()
            train_size = int(total_size * 0.6)
            val_size = int(total_size * 0.2)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

            def split_data_abl(d, idx_list):
                return {
                    'text_data': {
                        'input_ids': d['text_data']['input_ids'][idx_list],
                        'attention_mask': d['text_data']['attention_mask'][idx_list],
                    },
                    'node_ids_list': [d['node_ids_list'][i] for i in idx_list],
                    'edges_list': [d['edges_list'][i] for i in idx_list],
                    'node_levels_list': [d['node_levels_list'][i] for i in idx_list],
                    'struct_features': d['struct_features'][idx_list],
                    'targets': d['targets'][idx_list],
                }

            train_d = split_data_abl(data, train_indices)
            val_d = split_data_abl(data, val_indices)
            test_d = split_data_abl(data, test_indices)

            def make_loader(d, shuffle=False):
                ds = ComplaintDataset(
                    d['text_data'], d['node_ids_list'], d['edges_list'],
                    d['node_levels_list'], d['struct_features'], d['targets']
                )
                return DataLoader(ds, batch_size=16, shuffle=shuffle,
                                  collate_fn=custom_collate_fn, drop_last=shuffle)

            _train_loader = make_loader(train_d, shuffle=True)
            _val_loader = make_loader(val_d)
            _test_loader = make_loader(test_d)
            print(f"  ✅ 训练: {len(train_indices)}, 验证: {len(val_indices)}, 测试: {len(test_indices)}")

            # 创建模型（与ablation full_model一致）
            model = MultiModalComplaintModel(
                config=model_config, vocab_size=vocab_size,
                mode=mode, pretrained_path=None
            )
            model = model.to(self.device)

            # 使用ablation的train_and_evaluate函数
            exp_name = 'full_model'
            accuracy, precision_val, recall_val, f1_val, auc_val = taiwan_train_and_evaluate(
                model, _train_loader, _val_loader, _test_loader,
                model_config, self.device, exp_name
            )

            metrics = {
                'accuracy': accuracy,
                'precision': precision_val,
                'recall': recall_val,
                'f1': f1_val,
                'auc': auc_val,
                'fpr': [], 'tpr': []
            }
            print(f"  ✅ {model_name}: Acc={accuracy:.4f}, F1={f1_val:.4f}, AUC={auc_val:.4f}")
            self.results[model_name] = metrics

            # 保存TM-CRPP模型供跨文件复用
            _tw_save_dir = './outputs/baseline_comparison/taiwan/tmcrpp_models'
            os.makedirs(_tw_save_dir, exist_ok=True)
            _tw_save_path = os.path.join(_tw_save_dir, 'tmcrpp_taiwan.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model_config,
                'vocab_size': vocab_size,
                'mode': mode,
                'dataset_name': 'taiwan',
                'metrics': metrics,
            }, _tw_save_path)
            print(f"  💾 TM-CRPP模型已保存: {_tw_save_path}")

            # 清理
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if temp_file is not None:
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass

        except Exception as e:
            print(f"  ⚠️ TM-CRPP失败: {e}")
            import traceback
            traceback.print_exc()

    # ===== Run All Baseline =====
    def run_baseline(self):
        print("\n" + "="*60 + "\n🍜 Taiwan Restaurant Baseline\n" + "="*60)
        ep_s, ep_b = 1, 3

        def _safe(fn, *a, **kw):
            try:
                fn(*a, **kw)
            except Exception as e:
                print(f"  ⚠️ {fn.__name__} failed: {e}, skipping...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Layer 1: Text unimodal (TF-IDF)
        for fn in [self.run_tfidf_lr, self.run_tfidf_rf, self.run_tfidf_gbdt,
                    self.run_tfidf_svm, self.run_tfidf_xgboost]:
            _safe(fn)

        # ✅ 释放Layer 1的TF-IDF数据
        if hasattr(self, 'X_train_tfidf'):
            del self.X_train_tfidf, self.X_test_tfidf, self.vectorizer
            gc.collect()
            print("  🗑️ TF-IDF数据已释放")

        # Layer 2: Profile unimodal (Struct)
        for fn in [self.run_struct_lr, self.run_struct_rf, self.run_struct_gbdt,
                    self.run_struct_xgboost, self.run_struct_mlp]:
            _safe(fn)

        # Layer 3-5: DL models (需要GPU清理)
        _safe(self.run_label_mlp, ep_s); _safe(self.run_label_gat, ep_s)
        _safe(self.run_textcnn, 2); _safe(self.run_bilstm, 2);_safe(self.run_bert_base, 3, 1.4e-5, 11)
        _safe(self.run_bert_struct, ep_b); _safe(self.run_text_label, ep_b); _safe(self.run_profile_label, ep_s)
        _safe(self.run_early_fusion, 5, 1e-5, 11); _safe(self.run_late_fusion, 5, 1.4e-5, 11); _safe(self.run_attention_fusion, 5, 1.4e-5, 11)
        _safe(self._run_tmcrpp_real, ep_b)

        self._save_results(); self._print_summary()
        return self.results

    def _save_results(self):
        sd = f'./outputs/baseline_comparison/taiwan'
        os.makedirs(sd, exist_ok=True)
        dd = {n: {k: m.get(k, 0) for k in ['accuracy','precision','recall','f1','auc']} for n, m in self.results.items()}
        df = pd.DataFrame(dd).T; df.index.name = 'Method'; df = df.reset_index()
        for c in ['accuracy','precision','recall','f1','auc']: df[c] = df[c].round(4)
        df.to_excel(f'{sd}/baseline_5level_results.xlsx', index=False)
        df.to_csv(f'{sd}/baseline_5level_results.csv', index=False)
        jr = {n: {k: m.get(k, 0) for k in ['accuracy','precision','recall','f1','auc','fpr','tpr']} for n, m in self.results.items()}
        with open(f'{sd}/baseline_5level_results.json', 'w') as f: json.dump(jr, f, indent=2)
        print(f"\n✅ Results saved to {sd}/")

    def _print_summary(self):
        print("\n" + "="*80 + "\n📊 Taiwan Restaurant Results\n" + "="*80)
        print(f"\n{'Method':<30} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}\n" + "-"*70)
        for n, m in sorted(self.results.items(), key=lambda x: x[1].get('auc', 0), reverse=True):
            print(f"{n:<30} {m['accuracy']:.4f}  {m['precision']:.4f}  {m['recall']:.4f}  {m['f1']:.4f}  {m['auc']:.4f}")


# ============================================================
# Taiwan 消融实验
# ============================================================
def run_taiwan_ablation(data_file='Restaurant Complaint balanced.xlsx'):
    # ✅ Using inlined ComplaintDataProcessor, ComplaintDataset, custom_collate_fn
    # ✅ Using inlined MultiModalComplaintModel
    # ✅ Using inlined taiwan_train_and_evaluate
    import tempfile

    config = get_taiwan_restaurant_config()
    config.training.num_epochs = 10
    config.training.batch_size = 16

    experiments = [
        ('full_model', 'full'), ('text_only', 'text_only'), ('label_only', 'label_only'),
        ('struct_only', 'struct_only'), ('text_label', 'text_label'), ('text_struct', 'text_struct'),
        ('label_struct', 'label_struct'), ('No_pretrain', 'full'), ('label_gat', 'label_only'), ('label_flat', 'label_only'),
    ]
    experiment_seeds = {'full_model': 42, 'text_only': 43, 'label_only': 44, 'struct_only': 45,
        'text_label': 46, 'text_struct': 47, 'label_struct': 48, 'No_pretrain': 50, 'label_gat': 51, 'label_flat': 52}
    results = {}
    temp_file = None
    if os.path.exists(data_file):
        df = pd.read_excel(data_file)
        rm = {}
        if 'Complaint_label' in df.columns and 'Complaint label' not in df.columns:
            rm['Complaint_label'] = 'Complaint label'
        if 'satisfaction_binary' in df.columns and 'Repeat complaint' not in df.columns:
            rm['satisfaction_binary'] = 'Repeat complaint'
        if rm:
            df = df.rename(columns=rm)
            temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
            df.to_excel(temp_file.name, index=False)
            data_file = temp_file.name
            print(f"  列名适配: {rm}")

    for exp_name, mode in experiments:
        print(f"\n运行: {exp_name}\n" + "-"*40)
        seed = experiment_seeds.get(exp_name, 42)
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        use_No_pretrain = (exp_name == 'No_pretrain')
        processor = ComplaintDataProcessor(config=config, user_dict_file=config.data.user_dict_file)
        data = processor.prepare_datasets(train_file=data_file, for_pretrain=False)
        total_size = len(data['targets'])
        torch.manual_seed(seed)
        indices = torch.randperm(total_size).tolist()
        ts = int(total_size * 0.6); vs = int(total_size * 0.2)
        def split_data(data, idx):
            return {'text_data': {'input_ids': data['text_data']['input_ids'][idx], 'attention_mask': data['text_data']['attention_mask'][idx]},
                'node_ids_list': [data['node_ids_list'][i] for i in idx], 'edges_list': [data['edges_list'][i] for i in idx],
                'node_levels_list': [data['node_levels_list'][i] for i in idx], 'struct_features': data['struct_features'][idx], 'targets': data['targets'][idx]}
        trd = split_data(data, indices[:ts]); vld = split_data(data, indices[ts:ts+vs]); ted = split_data(data, indices[ts+vs:])
        trds = ComplaintDataset(trd['text_data'], trd['node_ids_list'], trd['edges_list'], trd['node_levels_list'], trd['struct_features'], trd['targets'])
        vlds = ComplaintDataset(vld['text_data'], vld['node_ids_list'], vld['edges_list'], vld['node_levels_list'], vld['struct_features'], vld['targets'])
        teds = ComplaintDataset(ted['text_data'], ted['node_ids_list'], ted['edges_list'], ted['node_levels_list'], ted['struct_features'], ted['targets'])
        trl = DataLoader(trds, batch_size=config.training.batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
        vll = DataLoader(vlds, batch_size=config.training.batch_size, collate_fn=custom_collate_fn)
        tel = DataLoader(teds, batch_size=config.training.batch_size, collate_fn=custom_collate_fn)
        use_flat = (exp_name == 'label_flat')
        model = MultiModalComplaintModel(config=config, vocab_size=data['vocab_size'], mode=mode, pretrained_path=None,
                                          No_pretrain_bert=use_No_pretrain, use_flat_label=use_flat)
        model = model.to(config.training.device)
        acc, prec, rec, f1v, aucv = taiwan_train_and_evaluate(model, trl, vll, tel, config, config.training.device, exp_name)
        results[exp_name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1v, 'auc': aucv}
        print(f"  {exp_name}: Acc={acc:.4f}, F1={f1v:.4f}, AUC={aucv:.4f}")
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if temp_file:
        try: os.unlink(temp_file.name)
        except Exception: pass
    print("\n" + "="*60 + f"\n台湾餐厅消融实验结果\n" + "="*60)
    print(f"\n{'实验':<15} {'Acc':<10} {'Prec':<10} {'Rec':<10} {'F1':<10} {'AUC':<10}\n" + "-"*70)
    for e, _ in experiments:
        if e in results:
            r = results[e]
            print(f"{e:<15} {r['accuracy']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f} {r['auc']:<10.4f}")
    with open('ablation_results_taiwan.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # 保存到Excel和CSV
    sd = './outputs/baseline_comparison/taiwan'
    os.makedirs(sd, exist_ok=True)
    df_data = {
        name: {k: round(v, 4) for k, v in metrics.items() if k in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
        for name, metrics in results.items()}
    df = pd.DataFrame(df_data).T
    df.index.name = 'Experiment'
    df = df.reset_index()
    df.to_excel(f'{sd}/ablation_results_taiwan.xlsx', index=False)
    df.to_csv(f'{sd}/ablation_results_taiwan.csv', index=False)
    print(f"\n✅ 消融结果已保存到: {sd}/")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'baseline', 'ablation'])
    parser.add_argument('--data_file', type=str, default='Restaurant Complaint balanced.xlsx')
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if args.mode in ['baseline', 'all']:
        exp = TaiwanBaselineExperiment(args.data_file)
        exp.run_baseline()
    if args.mode in ['ablation', 'all']:
        run_taiwan_ablation(args.data_file)
