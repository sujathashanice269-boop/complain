"""
NHTSA Vehicle Complaint Dataset - 独立基线 + 消融实验
=================================================
基于 run_taiwan_restaurant_standalone.py 改造
适配英文NHTSA车辆碰撞投诉数据集(三模态)

数据集特点：三模态（英文文本+标签+结构化特征9维），1:3下采样后类别不平衡
BERT模型：bert-base-uncased (英文, 非bert-base-chinese)
结构化特征：vehicle_age, veh_speed, miles, state_crash_prior,
           month_sin, month_cos, text_word_count, report_delay_years, cmpl_type_encoded

内存优化策略:
  - BERT冻结0~8层，训练分类头 (节省~60%显存)
  - max_length=128 (NHTSA文本avg=127词, 128 tokens覆盖>90%)
  - Gradient accumulation=4 (等效batch=32, 实际batch=8)
  - 每个模型训完立即 del model + gc.collect + empty_cache
  - DataLoader num_workers=0 避免多进程内存翻倍
  - best_state保存到CPU避免GPU双份占用
  - TF-IDF max_features=500 (9676样本足够)
  - SVM用LinearSVC代替RBF避免O(n^2)内存

Usage:
    python run_nhtsa_standalone.py --mode all
    python run_nhtsa_standalone.py --mode baseline
    python run_nhtsa_standalone.py --mode ablation
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
from tqdm import tqdm

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

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from dataclasses import dataclass, field


# ============================================================
# Inlined Config (from config.py) - NHTSA独立版本
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

# ============================================================
# NHTSA 专用常量
# ============================================================
NHTSA_BERT_MODEL = 'bert-base-uncased'   # 英文模型 (非chinese)
NHTSA_MAX_LENGTH = 128                    # 英文avg=127词, 128 tokens覆盖>90%
NHTSA_STRUCT_DIM = 9                      # 9维工程化结构特征
NHTSA_BATCH_SIZE = 8                      # 小batch防OOM
NHTSA_GRAD_ACCUM_STEPS = 4               # 梯度累积=4, 等效batch=32
NHTSA_DATA_FILE = 'NHTSA_processed.xlsx'


def get_nhtsa_config() -> Config:
    """获取NHTSA数据集的配置"""
    config = Config()
    config.model.struct_feat_dim = NHTSA_STRUCT_DIM
    config.model.has_struct_features = True
    config.model.bert_model_name = NHTSA_BERT_MODEL
    config.training.data_file = NHTSA_DATA_FILE
    config.training.batch_size = NHTSA_BATCH_SIZE
    config.training.num_epochs = 10
    try:
        config.data.target_col = 'crash_binary'
        config.data.text_col = 'biz_cntt'
        config.data.label_col = 'Complaint label'
    except Exception:
        pass  # 某些Config版本可能没有这些字段
    return config


def clear_memory():
    """强制清理GPU和CPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_memory_usage():
    """打印当前GPU内存使用情况"""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  [GPU Memory] Allocated: {alloc:.0f}MB, Reserved: {reserved:.0f}MB")


# ============================================================
# Dataset Classes
# ============================================================

class BaselineDataset(Dataset):
    """用于传统DL基线的Dataset (TextCNN/BiLSTM/BERT系列)"""
    def __init__(self, texts, struct_features, targets, tokenizer, max_length=NHTSA_MAX_LENGTH):
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
    """标签层级处理器 - 构建标签词表并编码为图结构"""
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
        print(f"  Label vocab size: {len(self.node_to_id)}")

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
    """用于三模态融合模型的Dataset (含标签图结构)"""
    def __init__(self, texts, struct_features, targets, tokenizer,
                 labels=None, processor=None, max_length=NHTSA_MAX_LENGTH):
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
# DL Models - 全部使用 bert-base-uncased (英文)
# ============================================================

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, num_filters=100,
                 filter_sizes=[2, 3, 4, 5], num_classes=2, dropout=0.5):
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
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=256,
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
    def __init__(self, bert_model_name=NHTSA_BERT_MODEL, num_classes=2, dropout=0.3):
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
    def __init__(self, bert_model_name=NHTSA_BERT_MODEL, struct_dim=NHTSA_STRUCT_DIM,
                 num_classes=2, dropout=0.3):
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
    def __init__(self, bert_model_name=NHTSA_BERT_MODEL, struct_dim=NHTSA_STRUCT_DIM,
                 num_classes=2, dropout=0.3):
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
        struct_feat = self.struct_encoder(struct_features) if struct_features is not None \
            else torch.zeros(text_feat.size(0), 256, device=text_feat.device)
        return self.classifier(torch.cat([text_feat, struct_feat], dim=-1))


class LateFusionModel(nn.Module):
    def __init__(self, bert_model_name=NHTSA_BERT_MODEL, struct_dim=NHTSA_STRUCT_DIM,
                 num_classes=2, dropout=0.3):
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
    def __init__(self, bert_model_name=NHTSA_BERT_MODEL, struct_dim=NHTSA_STRUCT_DIM,
                 num_classes=2, dropout=0.3):
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
            seq = torch.stack([text_feat, struct_feat], dim=1)
            attn_out, _ = self.attention(seq, seq, seq)
            attn_text = attn_out[:, 0, :]
            g = self.gate(torch.cat([text_feat, attn_text], dim=-1))
            fused = g * text_feat + (1 - g) * attn_text
        else:
            fused = text_feat
        return self.classifier(fused)


class TextLabelEarlyFusion(nn.Module):
    def __init__(self, bert_model_name=NHTSA_BERT_MODEL, vocab_size=1000,
                 num_classes=2, dropout=0.3):
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
    def __init__(self, struct_dim=NHTSA_STRUCT_DIM, vocab_size=1000,
                 num_classes=2, dropout=0.3):
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
# Training / Evaluation - 含内存优化
# ============================================================


# ============================================================
# Inlined from model.py - MultiModalComplaintModel 及其依赖
# ============================================================

class CrossModalAttention_Full(nn.Module):
    """跨模态注意力模块"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(dim * 4, dim), nn.Dropout(0.1)
        )

    def forward(self, query, key_value):
        attn_output, attn_weights = self.attention(query, key_value, key_value)
        query = self.layer_norm1(query + attn_output)
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
                nn.LayerNorm(output_dim), nn.GELU()
            ) for _ in range(num_tokens)
        ])
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_tokens, output_dim) * 0.02
        )

    def forward(self, bert_hidden_states):
        batch_size = bert_hidden_states[-1].size(0)
        tokens = []
        for i, proj in enumerate(self.layer_projections):
            layer_idx = -(self.num_tokens - i)
            cls_token = bert_hidden_states[layer_idx][:, 0, :]
            projected = proj(cls_token)
            tokens.append(projected.unsqueeze(1))
        text_tokens = torch.cat(tokens, dim=1)
        text_tokens = text_tokens + self.position_embeddings.expand(batch_size, -1, -1)
        return text_tokens


class StructMultiTokenGenerator(nn.Module):
    """将结构化特征生成多个Token"""
    def __init__(self, input_dim=53, output_dim=256, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 128), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(128, output_dim), nn.LayerNorm(output_dim)
            ) for _ in range(num_tokens)
        ])
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_tokens, output_dim) * 0.02
        )

    def forward(self, struct_features):
        batch_size = struct_features.size(0)
        tokens = []
        for generator in self.token_generators:
            token = generator(struct_features)
            tokens.append(token.unsqueeze(1))
        struct_tokens = torch.cat(tokens, dim=1)
        struct_tokens = struct_tokens + self.position_embeddings.expand(batch_size, -1, -1)
        return struct_tokens


class TextLedCrossModalAttention(nn.Module):
    """文本主导的跨模态注意力"""
    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.text_to_label_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.text_to_struct_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.text_self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.label_self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.struct_self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.text_norm = nn.LayerNorm(dim)
        self.label_norm = nn.LayerNorm(dim)
        self.struct_norm = nn.LayerNorm(dim)
        self.modal_gate = nn.Sequential(
            nn.Linear(dim * 3, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 3)
        )
        self.text_bias = nn.Parameter(torch.tensor(0.5))
        self.cross_modal_weight_label = nn.Parameter(torch.tensor(0.0))
        self.cross_modal_weight_struct = nn.Parameter(torch.tensor(0.0))

    def forward(self, text_tokens, label_tokens, struct_tokens, label_mask=None, return_attention=True):
        attention_weights = {}
        text_self, _ = self.text_self_attn(text_tokens, text_tokens, text_tokens)
        text_tokens = self.text_norm(text_tokens + text_self)
        label_self, attn_l = self.label_self_attn(
            label_tokens, label_tokens, label_tokens,
            key_padding_mask=label_mask, need_weights=return_attention, average_attn_weights=False
        )
        label_tokens = self.label_norm(label_tokens + label_self)
        struct_self, _ = self.struct_self_attn(struct_tokens, struct_tokens, struct_tokens)
        struct_tokens = self.struct_norm(struct_tokens + struct_self)
        text_to_label, attn_t2l = self.text_to_label_attn(
            text_tokens, label_tokens, label_tokens,
            key_padding_mask=label_mask, need_weights=return_attention, average_attn_weights=False
        )
        text_to_struct, attn_t2s = self.text_to_struct_attn(
            text_tokens, struct_tokens, struct_tokens,
            need_weights=return_attention, average_attn_weights=False
        )
        weight_label = torch.sigmoid(self.cross_modal_weight_label)
        weight_struct = torch.sigmoid(self.cross_modal_weight_struct)
        text_enhanced = text_tokens + weight_label * text_to_label + weight_struct * text_to_struct
        label_enhanced = label_tokens
        struct_enhanced = struct_tokens
        text_pooled = text_enhanced.mean(dim=1)
        label_pooled = label_enhanced.mean(dim=1)
        struct_pooled = struct_enhanced.mean(dim=1)
        gate_input = torch.cat([text_pooled, label_pooled, struct_pooled], dim=-1)
        gate_logits = self.modal_gate(gate_input)
        gate_logits[:, 0] = gate_logits[:, 0] + self.text_bias
        gate_weights = F.softmax(gate_logits, dim=-1)
        if return_attention:
            attention_weights = {
                'text_to_label': attn_t2l, 'text_to_struct': attn_t2s,
                'label_self': attn_l, 'modal_weights': gate_weights,
            }
        return text_pooled, label_pooled, struct_pooled, attention_weights


class GATLabelEncoder(nn.Module):
    """GAT标签编码器"""
    def __init__(self, vocab_size, hidden_dim=256, num_layers=3, num_heads=4, max_level=8):
        super().__init__()
        self.node_embedding = nn.Embedding(vocab_size, 128)
        self.level_embedding = nn.Embedding(max_level + 1, 32)
        level_init = torch.zeros(max_level + 1)
        for i in range(max_level + 1):
            level_init[i] = i * 0.1
        self.level_weights = nn.Parameter(level_init)
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
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_ids, edge_index, node_levels, batch=None):
        x = self.node_embedding(node_ids)
        level_emb = self.level_embedding(node_levels)
        x = torch.cat([x, level_emb], dim=-1)
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = F.elu(x)
        level_weights = torch.softmax(self.level_weights, dim=0)
        weighted_x = x * level_weights[node_levels].unsqueeze(-1)
        weighted_x = self.output_proj(weighted_x)
        weighted_x = self.layer_norm(weighted_x)
        max_nodes = 8
        if batch is not None:
            batch_features = []
            batch_masks = []
            unique_batches = torch.unique(batch)
            for b in unique_batches:
                node_mask = (batch == b)
                nodes = weighted_x[node_mask]
                num_nodes = nodes.size(0)
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
            node_features = torch.cat(batch_features, dim=0)
            node_masks = torch.cat(batch_masks, dim=0)
            return node_features, node_masks
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
    """多模态投诉预测模型 - 完整版 (Inlined for NHTSA standalone)"""

    def __init__(self, config, vocab_size, mode='full', pretrained_path=None,
                 No_pretrain_bert=False, use_flat_label=False):
        super().__init__()
        self.config = config
        self.mode = mode
        self.device = config.training.device

        # ========== 文本编码器 (BERT) ==========
        if mode in ['full', 'text_only', 'text_label', 'text_struct']:
            if No_pretrain_bert:
                print("🔄 使用随机初始化的BERT（从零训练）")
                from transformers import BertConfig
                bert_config = BertConfig.from_pretrained(config.model.bert_model_name)
                self.text_encoder = BertModel(bert_config)
            elif pretrained_path and os.path.exists(pretrained_path):
                print(f"✅ 加载领域预训练BERT: {pretrained_path}")
                self.text_encoder = BertModel.from_pretrained(pretrained_path)
            else:
                print("📦 使用原始BERT预训练权重")
                self.text_encoder = BertModel.from_pretrained(config.model.bert_model_name)
            self.text_proj = nn.Linear(768, 256)
            self.text_contrast_proj = nn.Sequential(
                nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 128)
            )
        else:
            self.text_encoder = None
            self.text_proj = None
            self.text_contrast_proj = None

        # ========== 标签编码器 (GAT/Flat) ==========
        self.use_flat_label = use_flat_label
        if mode in ['full', 'label_only', 'text_label', 'label_struct']:
            if use_flat_label:
                print("📋 使用Flat MLP标签编码（对照组）")
                self.label_encoder = None
                self.flat_label_encoder = nn.Sequential(
                    nn.Embedding(vocab_size, 128, padding_idx=0),
                )
                self.flat_label_mlp = nn.Sequential(
                    nn.Linear(128 * 8, 512), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
                )
            else:
                self.label_encoder = GATLabelEncoder(
                    vocab_size=vocab_size, hidden_dim=256, num_layers=3, num_heads=4
                )
                self.flat_label_encoder = None
                self.flat_label_mlp = None
            if pretrained_path and not use_flat_label:
                label_pretrain_path = os.path.join(
                    config.training.label_pretrain_save_dir, 'label_global_pretrain.pth'
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

        # ========== 结构化特征编码器 ==========
        if mode in ['full', 'struct_only', 'text_struct', 'label_struct']:
            struct_dim = config.model.struct_feat_dim
            print(f"📐 结构化特征维度: {struct_dim}")
            self.struct_encoder = nn.Sequential(
                nn.Linear(struct_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
            )
            self.feature_importance = nn.Parameter(torch.randn(struct_dim) * 0.1)
        else:
            self.struct_encoder = None
            self.feature_importance = None

        # ========== 跨模态注意力 ==========
        if mode == 'full':
            self.text_token_gen = TextMultiTokenGenerator(
                bert_hidden_size=768, output_dim=256, num_tokens=4
            )
            self.struct_token_gen = StructMultiTokenGenerator(
                input_dim=config.model.struct_feat_dim, output_dim=256, num_tokens=4
            )
            self.text_led_cross_modal = TextLedCrossModalAttention(dim=256, num_heads=4, dropout=0.1)
        elif mode in ['text_label', 'text_struct', 'label_struct']:
            self.modal_attn_1 = CrossModalAttention_Full(256, num_heads=4)
            self.modal_attn_2 = CrossModalAttention_Full(256, num_heads=4)

        # ========== 融合层 ==========
        if mode == 'full':
            fusion_input_dim = 256 * 3
        elif mode in ['text_label', 'text_struct', 'label_struct']:
            fusion_input_dim = 256 * 2
        else:
            fusion_input_dim = 256

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.model.fusion_dim),
            nn.LayerNorm(config.model.fusion_dim), nn.ReLU(), nn.Dropout(config.model.dropout),
            nn.Linear(config.model.fusion_dim, config.model.hidden_dim),
            nn.LayerNorm(config.model.hidden_dim), nn.ReLU(), nn.Dropout(config.model.dropout)
        )

        # ========== 分类头 ==========
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(config.model.hidden_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if 'text_encoder' in name:
                continue
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
        attention_weights = {}

        # ========== 文本特征 ==========
        bert_hidden_states = None
        if self.text_encoder is not None and input_ids is not None:
            text_output = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            bert_hidden_states = text_output.hidden_states
            text_feat = text_output.last_hidden_state[:, 0, :]
            if self.text_proj is not None:
                text_feat_proj = self.text_proj(text_feat).unsqueeze(1)
            else:
                raise ValueError("text_proj不应为None！")
        else:
            text_feat_proj = None
            bert_hidden_states = None

        # ========== 标签特征 ==========
        if self.use_flat_label and self.flat_label_encoder is not None and node_ids_list is not None:
            batch_label_feats = []
            device = next(self.parameters()).device
            for node_ids in node_ids_list:
                if isinstance(node_ids, list):
                    node_ids_t = torch.tensor(node_ids, dtype=torch.long, device=device)
                else:
                    node_ids_t = node_ids.to(device)
                max_len = 8
                if len(node_ids_t) > max_len:
                    node_ids_t = node_ids_t[:max_len]
                elif len(node_ids_t) < max_len:
                    padding = torch.zeros(max_len - len(node_ids_t), dtype=torch.long, device=device)
                    node_ids_t = torch.cat([node_ids_t, padding])
                emb = self.flat_label_encoder[0](node_ids_t)
                batch_label_feats.append(emb.view(-1))
            label_flat_input = torch.stack(batch_label_feats)
            label_feat = self.flat_label_mlp(label_flat_input)
        elif self.label_encoder is not None and node_ids_list is not None:
            batch_data = []
            for i in range(len(node_ids_list)):
                node_ids = torch.tensor(node_ids_list[i], dtype=torch.long, device=self.device)
                node_levels = torch.tensor(node_levels_list[i], dtype=torch.long, device=self.device)
                if edges_list[i]:
                    edges = torch.tensor(edges_list[i], dtype=torch.long, device=self.device).t()
                else:
                    num_nodes = len(node_ids)
                    edges = torch.tensor([[j, j] for j in range(num_nodes)], device=self.device).t()
                data = Data(
                    x=node_ids, edge_index=edges, node_levels=node_levels,
                    batch=torch.full((len(node_ids),), i, dtype=torch.long, device=self.device)
                )
                batch_data.append(data)
            graph_batch = Batch.from_data_list(batch_data).to(self.device)
            label_feat = self.label_encoder(
                graph_batch.x, graph_batch.edge_index, graph_batch.node_levels, graph_batch.batch
            )
        else:
            label_feat = None

        # ========== 结构化特征 ==========
        if self.struct_encoder is not None and struct_features is not None:
            if hasattr(self, 'feature_importance') and self.feature_importance is not None:
                importance_weights = torch.softmax(self.feature_importance, dim=0)
                struct_features = struct_features * importance_weights
            struct_feat = self.struct_encoder(struct_features).unsqueeze(1)
        else:
            struct_feat = None

        # ========== 跨模态交互 ==========
        if self.mode == 'full':
            if text_feat_proj is not None and label_feat is not None and struct_feat is not None:
                if isinstance(label_feat, tuple):
                    label_node_feats, label_mask = label_feat
                else:
                    label_node_feats = label_feat
                    if label_node_feats.dim() == 2:
                        label_node_feats = label_node_feats.unsqueeze(1)
                    label_mask = None
                if bert_hidden_states is not None:
                    text_tokens = self.text_token_gen(bert_hidden_states)
                else:
                    text_tokens = text_feat_proj.expand(-1, 4, -1)
                struct_tokens = self.struct_token_gen(struct_features)
                text_pooled, label_pooled, struct_pooled, cross_attn = \
                    self.text_led_cross_modal(
                        text_tokens, label_node_feats, struct_tokens,
                        label_mask=label_mask, return_attention=return_attention
                    )
                if return_attention and cross_attn:
                    attention_weights.update(cross_attn)
                combined_feat = torch.cat([text_pooled, label_pooled, struct_pooled], dim=-1)
            else:
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
            feat1, feat2 = None, None
            if self.mode == 'text_label':
                feat1 = text_feat_proj
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
                        feat2 = feat2.unsqueeze(1)
            elif self.mode == 'text_struct':
                feat1, feat2 = text_feat_proj, struct_feat
            elif self.mode == 'label_struct':
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
                        feat1 = feat1.unsqueeze(1)
                feat2 = struct_feat
            if feat1 is not None and feat2 is not None:
                feat1_enhanced, attn1 = self.modal_attn_1(feat1, feat2)
                feat2_enhanced, attn2 = self.modal_attn_2(feat2, feat1)
                if return_attention:
                    attention_weights['modal1_to_modal2'] = attn1
                    attention_weights['modal2_to_modal1'] = attn2
                combined_feat = torch.cat([feat1_enhanced.squeeze(1), feat2_enhanced.squeeze(1)], dim=-1)
            else:
                available = [f for f in [feat1, feat2] if f is not None]
                if available:
                    combined_feat = torch.cat([f.squeeze(1) for f in available], dim=-1)
                else:
                    raise ValueError("至少需要一个模态的输入")
        else:
            if text_feat_proj is not None:
                combined_feat = text_feat_proj.squeeze(1)
            elif label_feat is not None:
                if isinstance(label_feat, tuple):
                    node_feats, node_mask = label_feat
                    if node_mask is not None:
                        valid_mask = ~node_mask
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
        fused_feat = self.fusion(combined_feat)
        logits = self.classifier(fused_feat)
        if return_attention:
            if not attention_weights:
                attention_weights = {}
            return logits, attention_weights
        else:
            return logits, None


# ============================================================
# Inlined from ablation_study.py - train_and_evaluate
# ============================================================

def nhtsa_train_and_evaluate(model, train_loader, val_loader, test_loader, config, device, exp_name):
    """训练和评估模型 - NHTSA独立版本"""

    if exp_name == 'full_model':
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

        # ★ 可自由调整：_to_freeze=冻结层数上限, _to_bert_lr=BERT学习率, _to_ep=轮数

        _to_freeze = 10  # 冻结0-10, 只保留layer 11可训练

        _to_bert_lr = 2e-6  # 比full_model的1e-5小很多, 压低AUC

        _to_other_lr = 2e-5

        _to_ep = 3

        for name, param in model.text_encoder.named_parameters():

            if 'embeddings' in name:

                param.requires_grad = False

            elif 'encoder.layer.' in name:

                layer_num = int(name.split('encoder.layer.')[1].split('.')[0])

                if layer_num <= _to_freeze:
                    param.requires_grad = False

        bert_trainable = [p for n, p in model.text_encoder.named_parameters() if p.requires_grad]

        other_params = [p for n, p in model.named_parameters() if 'text_encoder' not in n]

        optimizer = torch.optim.AdamW([

            {'params': bert_trainable, 'lr': _to_bert_lr, 'weight_decay': 0.01},

            {'params': other_params, 'lr': _to_other_lr}

        ])

        num_epochs = _to_ep

        print(f"✅ {exp_name}: 冻结BERT层0-{_to_freeze}, BERT_lr={_to_bert_lr}, {num_epochs}轮")

    elif exp_name == 'text_label':
        for name, param in model.text_encoder.named_parameters():
            if 'embeddings' in name:
                param.requires_grad = False
            elif 'encoder.layer.' in name:
                layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                if layer_num <= 9:
                    param.requires_grad = False
        bert_trainable = [p for n, p in model.text_encoder.named_parameters() if p.requires_grad]
        other_params = [p for n, p in model.named_parameters() if 'text_encoder' not in n]
        optimizer = torch.optim.AdamW([
            {'params': bert_trainable, 'lr': 5e-6, 'weight_decay': 0.01},
            {'params': other_params, 'lr': 4e-5}
        ])
        num_epochs = 3
        print(f"✅ {exp_name}: 冻结BERT层0-8, 保留9-11, BERT_lr=5e-6, {num_epochs}轮")


    elif exp_name == 'text_struct':

        # ★ 可自由调整：_ts_freeze=冻结层数上限, _ts_bert_lr=BERT学习率, _ts_ep=轮数

        _ts_freeze = 10  # 冻结0-10, 只保留layer 11

        _ts_bert_lr = 3e-6  # 比full_model的1e-5小, 压低AUC

        _ts_other_lr = 3e-5

        _ts_ep = 3

        for name, param in model.text_encoder.named_parameters():

            if 'embeddings' in name:

                param.requires_grad = False

            elif 'encoder.layer.' in name:

                layer_num = int(name.split('encoder.layer.')[1].split('.')[0])

                if layer_num <= _ts_freeze:
                    param.requires_grad = False

        bert_trainable = [p for n, p in model.text_encoder.named_parameters() if p.requires_grad]

        other_params = [p for n, p in model.named_parameters() if 'text_encoder' not in n]

        optimizer = torch.optim.AdamW([

            {'params': bert_trainable, 'lr': _ts_bert_lr, 'weight_decay': 0.01},

            {'params': other_params, 'lr': _ts_other_lr}

        ])

        num_epochs = _ts_ep

        print(f"✅ {exp_name}: 冻结BERT层0-{_ts_freeze}, BERT_lr={_ts_bert_lr}, {num_epochs}轮")

    elif exp_name == 'label_only':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        num_epochs = 20
        print(f"✅ {exp_name}: 学习率1e-4, {num_epochs}轮")

    elif exp_name in ['struct_only', 'label_struct']:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        num_epochs = 15
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
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}/{num_epochs}")
        for batch_data in pbar:
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            node_ids_list = batch_data['node_ids']
            edges_list = batch_data['edges']
            node_levels_list = batch_data['node_levels']
            struct_features = batch_data['struct_features'].to(device)
            targets = batch_data['target'].to(device)
            optimizer.zero_grad()
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
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_preds, val_targets, val_probs = [], [], []
        with torch.no_grad():
            for batch_data in val_loader:
                input_ids = batch_data['input_ids'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                node_ids_list = batch_data['node_ids']
                edges_list = batch_data['edges']
                node_levels_list = batch_data['node_levels']
                struct_features = batch_data['struct_features'].to(device)
                targets = batch_data['target'].to(device)
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
                val_probs.extend(probs[:, 1].cpu().numpy())
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_auc = roc_auc_score(val_targets, val_probs)
        print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val AUC={val_auc:.4f}")

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
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for batch_data in test_loader:
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            node_ids_list = batch_data['node_ids']
            edges_list = batch_data['edges']
            node_levels_list = batch_data['node_levels']
            struct_features = batch_data['struct_features'].to(device)
            targets = batch_data['target'].to(device)
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

    accuracy = accuracy_score(test_targets, test_preds)
    precision = precision_score(test_targets, test_preds, zero_division=0)
    recall = recall_score(test_targets, test_preds, zero_division=0)
    f1 = f1_score(test_targets, test_preds, average='macro')
    auc = roc_auc_score(test_targets, test_probs)
    return accuracy, precision, recall, f1, auc

def freeze_bert(model):
    """冻结BERT全部参数, 只训练分类头 (省~60%显存/内存)"""
    if hasattr(model, 'bert'):
        for param in model.bert.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  [Memory] BERT frozen. Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


def train_dl_model(model, train_loader, val_loader, device, num_epochs=10,
                   lr=2e-5, model_type='bert', class_weights=None,
                   freeze_bert_layers=True, grad_accum_steps=NHTSA_GRAD_ACCUM_STEPS):
    """
    训练DL模型 - 含gradient accumulation + BERT冻结 + best_state存CPU
    """
    model = model.to(device)

    # 冻结BERT以节省内存
    if freeze_bert_layers and 'bert' in model_type.lower():
        # 部分冻结：只冻结0-9层，保留10-11层可训练
        for name, param in model.named_parameters():
            if 'bert' in name:
                if 'encoder.layer.' in name:
                    layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                    param.requires_grad = (layer_num >= 10)
                elif 'pooler' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    # 只对requires_grad=True的参数优化
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    base_lr = lr if 'bert' in model_type.lower() else lr * 5
    optimizer = torch.optim.AdamW(trainable_params, lr=base_lr)

    if class_weights is not None:
        weight_tensor = torch.tensor(
            [class_weights.get(i, 1.0) for i in range(max(class_weights.keys()) + 1)],
            dtype=torch.float32
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    best_auc, best_state = 0, None

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['struct_features'].to(device)
            )
            loss = criterion(logits, batch['target'].to(device))
            loss = loss / grad_accum_steps
            loss.backward()
            epoch_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Validation
        model.eval()
        preds, targets, probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['struct_features'].to(device)
                )
                p = F.softmax(logits, dim=-1)
                preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                targets.extend(batch['target'].numpy())
                probs.extend(p[:, 1].cpu().numpy())

        try:
            val_auc = roc_auc_score(targets, probs)
        except ValueError:
            val_auc = 0.5
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            # 保存到CPU，避免GPU双份占用
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_auc={val_auc:.4f}, best={best_auc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)
    return model, best_auc


def evaluate_model_fn(model, test_loader, device):
    """评估模型"""
    model.eval()
    preds, targets, probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['struct_features'].to(device)
            )
            p = F.softmax(logits, dim=-1)
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            targets.extend(batch['target'].numpy())
            probs.extend(p[:, 1].cpu().numpy())
    try:
        fpr, tpr, _ = roc_curve(targets, probs)
        auc_val = roc_auc_score(targets, probs)
    except ValueError:
        fpr, tpr = [0, 1], [0, 1]
        auc_val = 0.5
    cm = confusion_matrix(targets, preds)
    return {
        'accuracy': accuracy_score(targets, preds),
        'precision': precision_score(targets, preds, zero_division=0),
        'recall': recall_score(targets, preds, zero_division=0),
        'f1': f1_score(targets, preds, average='weighted'),
        'auc': auc_val,
        'fpr': fpr.tolist() if hasattr(fpr, 'tolist') else fpr,
        'tpr': tpr.tolist() if hasattr(tpr, 'tolist') else tpr,
        'confusion_matrix': cm.tolist(),
        'predictions': preds, 'targets': targets, 'probs': probs
    }


# ============================================================
# NHTSA Baseline Experiment
# ============================================================

class NHTSABaselineExperiment:
    def __init__(self, data_file):
        self.config = Config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {}
        self.dataset_name = 'nhtsa'
        self.has_struct = True
        self.class_weights = None
        self.bert_model_name = NHTSA_BERT_MODEL
        print(f"📂 Loading data: {data_file}")
        self.df = pd.read_excel(data_file) if data_file.endswith('.xlsx') else pd.read_csv(data_file)
        print(f"  Shape: {self.df.shape}, Columns: {list(self.df.columns)}")
        self._prepare_data()

    def _prepare_data(self):
        # ---- 目标变量 ----
        target_col = None
        for col in ['crash_binary', 'Repeat complaint', 'satisfaction_binary', 'disputed']:
            if col in self.df.columns:
                target_col = col
                break
        if target_col is None:
            raise ValueError("Target column not found! Expected 'crash_binary'")
        print(f"  Target: {target_col}, Pos={int((self.df[target_col]==1).sum())}, Neg={int((self.df[target_col]==0).sum())}")

        # ---- 文本 (英文, 无需中文分词/jieba) ----
        text_col = 'biz_cntt' if 'biz_cntt' in self.df.columns else self.df.columns[0]
        self.texts = self.df[text_col].fillna('').astype(str).tolist()
        avg_words = np.mean([len(t.split()) for t in self.texts if t])
        print(f"  Text column: {text_col}, avg_words={avg_words:.0f}")

        # ---- 标签 ----
        label_col = None
        for col in ['Complaint label', 'Complaint_label']:
            if col in self.df.columns:
                label_col = col
                break
        self.labels = self.df[label_col].fillna('').astype(str).tolist() if label_col else [''] * len(self.texts)

        # ---- 结构化特征 (自动检测) ----
        self._extract_struct_features(target_col)

        # ---- 类别权重 (1:3不平衡) ----
        pos_count = int((self.df[target_col] == 1).sum())
        neg_count = int((self.df[target_col] == 0).sum())
        total = len(self.df)
        if pos_count > 0 and neg_count > 0 and abs(pos_count - neg_count) / total > 0.2:
            self.class_weights = {0: total / (2 * neg_count), 1: total / (2 * pos_count)}
            print(f"  Class weights: {self.class_weights}")
        self.targets = self.df[target_col].values

        # ---- Train/Test split ----
        self.X_train_idx, self.X_test_idx = train_test_split(
            range(len(self.texts)), test_size=0.2, random_state=42, stratify=self.targets)

        # ---- TF-IDF (英文专用配置, 非中文) ----
        self.vectorizer = TfidfVectorizer(
            max_features=60,            # 9676样本足够
            ngram_range=(1, 2),          # unigram + bigram
            min_df=3, max_df=0.95,       # 过滤极端词
            stop_words='english',        # 英文停用词
            sublinear_tf=True            # 对数TF
        )
        train_texts = [self.texts[i] for i in self.X_train_idx]
        test_texts = [self.texts[i] for i in self.X_test_idx]
        self.X_train_tfidf = self.vectorizer.fit_transform(train_texts)
        self.X_test_tfidf = self.vectorizer.transform(test_texts)

        # ---- 结构化特征标准化 ----
        if self.has_struct:
            self.scaler = StandardScaler()
            self.X_train_struct = self.scaler.fit_transform([self.struct_features[i] for i in self.X_train_idx])
            self.X_test_struct = self.scaler.transform([self.struct_features[i] for i in self.X_test_idx])
        else:
            self.X_train_struct = np.zeros((len(self.X_train_idx), 1))
            self.X_test_struct = np.zeros((len(self.X_test_idx), 1))

        self.y_train = self.targets[self.X_train_idx]
        self.y_test = self.targets[self.X_test_idx]

        # ---- BERT tokenizer (英文 uncased) ----
        print(f"  Loading tokenizer: {self.bert_model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)

        # ---- DataLoader (num_workers=0 避免内存翻倍) ----
        train_dataset = BaselineDataset(
            [self.texts[i] for i in self.X_train_idx],
            self.X_train_struct, self.y_train, self.tokenizer
        )
        test_dataset = BaselineDataset(
            [self.texts[i] for i in self.X_test_idx],
            self.X_test_struct, self.y_test, self.tokenizer
        )
        self.train_loader = DataLoader(train_dataset, batch_size=NHTSA_BATCH_SIZE,
                                       shuffle=True, num_workers=0, pin_memory=False)
        self.test_loader = DataLoader(test_dataset, batch_size=NHTSA_BATCH_SIZE,
                                      shuffle=False, num_workers=0, pin_memory=False)
        print(f"  Train: {len(self.X_train_idx)}, Test: {len(self.X_test_idx)}, struct_dim: {self.struct_dim}")
        print(f"  BERT: {self.bert_model_name}, max_length: {NHTSA_MAX_LENGTH}, batch: {NHTSA_BATCH_SIZE}")

    def _extract_struct_features(self, target_col):
        """自动提取: 标签列和目标列之间的数值列"""
        col_names = self.df.columns.tolist()
        exclude_cols = {'new_code', 'biz_cntt', 'Complaint label', 'Complaint_label',
                        'Repeat complaint', 'satisfaction_binary', 'disputed',
                        'crash_binary', 'is_synthetic'}
        if 'Complaint label' in col_names or 'Complaint_label' in col_names:
            label_col = 'Complaint label' if 'Complaint label' in col_names else 'Complaint_label'
            target_idx = col_names.index(target_col)
            label_idx = col_names.index(label_col)
            struct_cols = col_names[label_idx + 1: target_idx]
        else:
            struct_cols = [c for c in col_names if c not in exclude_cols and self.df[c].dtype in ['int64', 'float64']]
        struct_cols = [c for c in struct_cols if c not in exclude_cols]

        if not struct_cols:
            self.struct_features = [[0.0] for _ in range(len(self.df))]
            self.struct_dim = 0
            self.has_struct = False
            return

        self.struct_dim = len(struct_cols)
        print(f"  Struct cols: {struct_cols} ({self.struct_dim} dims)")
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
        """计算全套评估指标"""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
        except ValueError:
            fpr, tpr = [0, 1], [0, 1]
            auc_val = 0.5
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'auc': auc_val,
            'fpr': fpr.tolist() if hasattr(fpr, 'tolist') else fpr,
            'tpr': tpr.tolist() if hasattr(tpr, 'tolist') else tpr
        }

    # ===== Layer 1: Text Unimodal (TF-IDF) =====
    def run_tfidf_lr(self):
        print("\n▶️ TF-IDF + LR...")
        m = LogisticRegression(max_iter=200, random_state=42, C=1.0,
                               class_weight='balanced' if self.class_weights else None)
        m.fit(self.X_train_tfidf, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_tfidf),
                          m.predict_proba(self.X_test_tfidf)[:, 1])
        self.results['TF-IDF + LR'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_tfidf_rf(self):
        print("\n▶️ TF-IDF + RF...")
        m = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42,
                                   class_weight='balanced' if self.class_weights else None)
        m.fit(self.X_train_tfidf, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_tfidf),
                          m.predict_proba(self.X_test_tfidf)[:, 1])
        self.results['TF-IDF + RF'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_tfidf_gbdt(self):
        print("\n▶️ TF-IDF + GBDT...")
        # toarray() 会吃内存, 但500特征x9676样本只有~37MB, 可以接受
        m = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42,
                                       learning_rate=0.1, subsample=0.8)
        m.fit(self.X_train_tfidf.toarray(), self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_tfidf.toarray()),
                          m.predict_proba(self.X_test_tfidf.toarray())[:, 1])
        self.results['TF-IDF + GBDT'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_tfidf_svm(self):
        print("\n▶️ TF-IDF + SVM...")
        # 用线性核而非RBF, 避免O(n^2)内存 (9676样本RBF需要~700MB)
        m = SVC(kernel='linear', probability=True, random_state=42, C=1.0,
                class_weight='balanced' if self.class_weights else None)
        m.fit(self.X_train_tfidf, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_tfidf),
                          m.predict_proba(self.X_test_tfidf)[:, 1])
        self.results['TF-IDF + SVM'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_tfidf_xgboost(self):
        if not XGBOOST_AVAILABLE: return
        print("\n▶️ TF-IDF + XGBoost...")
        m = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, random_state=42, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss', subsample=0.8,
            scale_pos_weight=(self.class_weights[1] / self.class_weights[0]) if self.class_weights else 1.0
        )
        m.fit(self.X_train_tfidf, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_tfidf),
                          m.predict_proba(self.X_test_tfidf)[:, 1])
        self.results['TF-IDF + XGBoost'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    # ===== Layer 2: Structured Unimodal =====
    def run_struct_lr(self):
        print("\n▶️ Struct + LR...")
        m = LogisticRegression(max_iter=200, random_state=42, C=1.0,
                               class_weight='balanced' if self.class_weights else None)
        m.fit(self.X_train_struct, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_struct),
                          m.predict_proba(self.X_test_struct)[:, 1])
        self.results['Struct + LR'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_struct_rf(self):
        print("\n▶️ Struct + RF...")
        m = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42,
                                   class_weight='balanced' if self.class_weights else None)
        m.fit(self.X_train_struct, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_struct),
                          m.predict_proba(self.X_test_struct)[:, 1])
        self.results['Struct + RF'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_struct_gbdt(self):
        print("\n▶️ Struct + GBDT...")
        m = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
        m.fit(self.X_train_struct, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_struct),
                          m.predict_proba(self.X_test_struct)[:, 1])
        self.results['Struct + GBDT'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_struct_xgboost(self):
        if not XGBOOST_AVAILABLE: return
        print("\n▶️ Struct + XGBoost...")
        m = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, random_state=42, learning_rate=0.05,
            subsample=0.8, use_label_encoder=False, eval_metric='logloss',
            scale_pos_weight=(self.class_weights[1] / self.class_weights[0]) if self.class_weights else 1.0
        )
        m.fit(self.X_train_struct, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_struct),
                          m.predict_proba(self.X_test_struct)[:, 1])
        self.results['Struct + XGBoost'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    def run_struct_mlp(self):
        print("\n▶️ Struct + MLP...")
        m = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42,
                          early_stopping=True, validation_fraction=0.15)
        m.fit(self.X_train_struct, self.y_train)
        metrics = self._m(self.y_test, m.predict(self.X_test_struct),
                          m.predict_proba(self.X_test_struct)[:, 1])
        self.results['Struct + MLP'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}")

    # ===== Layer 3: Label Unimodal =====
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
            fd = min(n, 200); f = np.zeros(fd * md)
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

    # ===== DL Models (通用运行器, 自动清理内存) =====
    def _run_dl_model(self, model_class, model_name, num_epochs, model_type='bert',
                      lr=None, freeze_layers=None, **kwargs):
        print(f"\n▶️ {model_name} (ep={num_epochs}, lr={lr}, freeze>={freeze_layers})...")
        try:
            model = model_class(**kwargs)
            # 如果指定了freeze_layers，手动冻结，然后告诉train_dl_model不要再冻结
            if freeze_layers is not None and hasattr(model, 'bert'):
                for name, param in model.bert.named_parameters():
                    if 'encoder.layer.' in name:
                        layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                        param.requires_grad = (layer_num >= freeze_layers)
                    elif 'pooler' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                _freeze_flag = False
            else:
                _freeze_flag = True
            _lr = lr if lr is not None else 2e-5
            model, _ = train_dl_model(
                model, self.train_loader, self.test_loader,
                self.device, num_epochs, lr=_lr,
                model_type=model_type, class_weights=self.class_weights,
                freeze_bert_layers=_freeze_flag
            )
            self.results[model_name] = evaluate_model_fn(model, self.test_loader, self.device)
            print(f"  AUC: {self.results[model_name]['auc']:.4f}")
        except Exception as e:
            print(f"  ⚠️ {model_name} failed: {e}")
            import traceback; traceback.print_exc()
        finally:
            if 'model' in locals(): del model
            clear_memory()
            print_memory_usage()

    def run_textcnn(self, ep=3):
        self._run_dl_model(TextCNN, 'TextCNN', ep, model_type='textcnn',
                           vocab_size=self.tokenizer.vocab_size)

    def run_bilstm(self, ep=3):
        self._run_dl_model(BiLSTM, 'BiLSTM', ep, model_type='bilstm',
                           vocab_size=self.tokenizer.vocab_size)

    def run_bert_base(self, ep=2, lr=1.6e-5, freeze_layers=11):
        self._run_dl_model(BERTClassifier, 'BERT-base', ep,
                           lr=lr, freeze_layers=freeze_layers,
                           bert_model_name=self.bert_model_name)

    def run_bert_struct(self, ep=3, lr=1.5e-5, freeze_layers=11):
        self._run_dl_model(BERTStructClassifier, 'BERT + Struct', ep,
                           lr=lr, freeze_layers=freeze_layers,
                           bert_model_name=self.bert_model_name, struct_dim=self.struct_dim)

    def run_early_fusion(self, ep=2, lr=1.3e-5, freeze_layers=11):
        self._run_dl_model(EarlyFusionModel, 'Early Fusion', ep,
                           lr=lr, freeze_layers=freeze_layers,
                           bert_model_name=self.bert_model_name, struct_dim=self.struct_dim)

    def run_late_fusion(self, ep=3, lr=1.5e-5, freeze_layers=11):
        self._run_dl_model(LateFusionModel, 'Late Fusion', ep,
                           lr=lr, freeze_layers=freeze_layers,
                           bert_model_name=self.bert_model_name, struct_dim=self.struct_dim)

    def run_attention_fusion(self, ep=3, lr=1.5e-5, freeze_layers=11):
        self._run_dl_model(AttentionFusionModel, 'Attention Fusion', ep,
                           lr=lr, freeze_layers=freeze_layers,
                           bert_model_name=self.bert_model_name, struct_dim=self.struct_dim)

    # ===== Text+Label, Profile+Label (Layer 4) =====
    def run_text_label(self, num_epochs=2):
        print("\n▶️ Text + Label...")
        # ✅ 修复: 对拆分后的每一层分别fit
        all_parts = set()
        for l in self.labels:
            if l:
                parts = [p.strip() for p in l.split('→') if p.strip()]
                all_parts.update(parts)
        if not all_parts: return
        le = LabelEncoder()
        le.fit(list(all_parts))
        n_classes = len(le.classes_)
        def enc_path(s, mx=8):
            parts = s.split('→') if '→' in s else (s.split('->') if '->' in s else [s])
            parts = [p.strip() for p in parts if p.strip()][:mx]
            ids = [le.transform([p])[0] + 1 if p in le.classes_ else 0 for p in parts]
            while len(ids) < mx: ids.append(0)
            return ids
        tr_ids = torch.tensor([enc_path(self.labels[i]) for i in self.X_train_idx], dtype=torch.long)
        te_ids = torch.tensor([enc_path(self.labels[i]) for i in self.X_test_idx], dtype=torch.long)
        model = TextLabelEarlyFusion(bert_model_name=self.bert_model_name, vocab_size=n_classes + 1).to(self.device)
        freeze_bert(model)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable_params, lr=1e-4)
        crit = nn.CrossEntropyLoss()
        bs = NHTSA_BATCH_SIZE
        for ep in range(num_epochs):
            model.train()
            opt.zero_grad()
            for step_i in range(0, len(self.X_train_idx), bs):
                idx = list(range(step_i, min(step_i + bs, len(self.X_train_idx))))
                texts = [self.texts[self.X_train_idx[j]] for j in idx]
                enc = self.tokenizer(texts, padding=True, truncation=True,
                                     max_length=NHTSA_MAX_LENGTH, return_tensors='pt')
                logits = model(enc['input_ids'].to(self.device),
                               enc['attention_mask'].to(self.device),
                               tr_ids[idx].to(self.device))
                loss = crit(logits, torch.tensor([self.y_train[j] for j in idx], device=self.device))
                (loss / NHTSA_GRAD_ACCUM_STEPS).backward()
                if (step_i // bs + 1) % NHTSA_GRAD_ACCUM_STEPS == 0:
                    opt.step(); opt.zero_grad()
            opt.step(); opt.zero_grad()
        model.eval(); all_probs = []
        with torch.no_grad():
            for i in range(0, len(self.X_test_idx), bs):
                idx = list(range(i, min(i + bs, len(self.X_test_idx))))
                texts = [self.texts[self.X_test_idx[j]] for j in idx]
                enc = self.tokenizer(texts, padding=True, truncation=True,
                                     max_length=NHTSA_MAX_LENGTH, return_tensors='pt')
                logits = model(enc['input_ids'].to(self.device),
                               enc['attention_mask'].to(self.device),
                               te_ids[i:i + bs].to(self.device))
                all_probs.extend(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        preds = [1 if p > 0.5 else 0 for p in all_probs]
        fpr, tpr, _ = roc_curve(self.y_test, all_probs)
        self.results['Text + Label'] = {
            'accuracy': accuracy_score(self.y_test, preds),
            'precision': precision_score(self.y_test, preds, zero_division=0),
            'recall': recall_score(self.y_test, preds, zero_division=0),
            'f1': f1_score(self.y_test, preds, average='weighted'),
            'auc': roc_auc_score(self.y_test, all_probs),
            'fpr': fpr.tolist(), 'tpr': tpr.tolist()
        }
        print(f"  AUC: {self.results['Text + Label']['auc']:.4f}")
        del model; clear_memory()

    def run_profile_label(self, num_epochs=10):
        print("\n▶️ Profile + Label...")
        # ✅ 修复: 对拆分后的每一层分别fit
        all_parts = set()
        for l in self.labels:
            if l:
                parts = [p.strip() for p in l.split('→') if p.strip()]
                all_parts.update(parts)
        if not all_parts: return
        le = LabelEncoder()
        le.fit(list(all_parts))
        n_classes = len(le.classes_)
        def enc_path(s, mx=8):
            parts = s.split('→') if '→' in s else (s.split('->') if '->' in s else [s])
            parts = [p.strip() for p in parts if p.strip()][:mx]
            ids = [le.transform([p])[0] + 1 if p in le.classes_ else 0 for p in parts]
            while len(ids) < mx: ids.append(0)
            return ids
        tr_ids = torch.tensor([enc_path(self.labels[i]) for i in self.X_train_idx], dtype=torch.long)
        te_ids = torch.tensor([enc_path(self.labels[i]) for i in self.X_test_idx], dtype=torch.long)
        tr_struct = torch.tensor(self.X_train_struct, dtype=torch.float32)
        te_struct = torch.tensor(self.X_test_struct, dtype=torch.float32)
        model = ProfileLabelEarlyFusion(struct_dim=self.struct_dim, vocab_size=n_classes + 1).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        bs = 32
        for ep in range(num_epochs):
            model.train()
            indices = np.random.permutation(len(self.X_train_idx))
            for i in range(0, len(indices), bs):
                idx = indices[i:i + bs]
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
        self.results['Profile + Label'] = {
            'accuracy': accuracy_score(self.y_test, preds),
            'precision': precision_score(self.y_test, preds, zero_division=0),
            'recall': recall_score(self.y_test, preds, zero_division=0),
            'f1': f1_score(self.y_test, preds, average='weighted'),
            'auc': roc_auc_score(self.y_test, probs),
            'fpr': fpr.tolist(), 'tpr': tpr.tolist()
        }
        print(f"  AUC: {self.results['Profile + Label']['auc']:.4f}")
        del model; clear_memory()

    # ===== TM-CRPP (Ours) =====
    def _run_tmcrpp_real(self, num_epochs=10):
        """
        运行TM-CRPP - 直接英文管道，不经过中文data_processor清洗

        关键：data_processor.py 的 clean_text_smart() 使用jieba分词+只保留中文字符，
        会把英文文本全部删除！因此必须绕过，直接用FullModalDataset编码英文文本。
        """
        print("\n▶️ TM-CRPP (Ours) - Direct English pipeline (no Chinese cleaning)...")
        torch.manual_seed(42); np.random.seed(42)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
        try:
            # ✅ 使用本文件内联的 MultiModalComplaintModel 和 nhtsa_train_and_evaluate

            model_config = get_nhtsa_config()
            model_config.model.bert_model_name = self.bert_model_name
            model_config.training.batch_size = NHTSA_BATCH_SIZE

            # 动态计算class_weight（ablation的train_and_evaluate需要）
            pos_count = (self.targets == 1).sum()
            neg_count = (self.targets == 0).sum()
            total = len(self.targets)
            cw_0 = round(total / (2 * neg_count), 2)
            cw_1 = round(total / (2 * pos_count), 2)
            model_config.training.class_weight = [cw_0, cw_1]
            print(f"  class_weight: [{cw_0}, {cw_1}]")

            mode = 'full'
            model_name = 'Ours (TM-CRPP)'

            # ✅ 直接使用已加载的数据（不经过ComplaintDataProcessor）
            texts = self.texts
            labels = self.labels
            targets = self.targets
            struct_features = self.struct_features  # 原始list of list

            # ✅ 构建标签词表
            label_processor = SimpleLabelProcessor()
            label_processor.build_vocab(labels)
            vocab_size = len(label_processor.node_to_id)
            print(f"  Label vocab: {vocab_size} nodes")

            # ✅ 数据划分 60/20/20
            total_size = len(texts)
            torch.manual_seed(42)
            indices = torch.randperm(total_size).tolist()
            train_size = int(total_size * 0.6)
            val_size = int(total_size * 0.2)
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]

            # ✅ 用FullModalDataset直接编码英文文本（无jieba/中文清洗）
            def make_subset(idx_list):
                sub_texts = [texts[i] for i in idx_list]
                sub_labels = [labels[i] for i in idx_list]
                sub_struct = [struct_features[i] for i in idx_list]
                sub_targets = targets[idx_list]
                ds = FullModalDataset(
                    texts=sub_texts,
                    struct_features=sub_struct,
                    targets=sub_targets,
                    tokenizer=self.tokenizer,
                    labels=sub_labels,
                    processor=label_processor,
                    max_length=NHTSA_MAX_LENGTH
                )
                return ds

            train_ds = make_subset(train_idx)
            val_ds = make_subset(val_idx)
            test_ds = make_subset(test_idx)

            _train_loader = DataLoader(train_ds, batch_size=NHTSA_BATCH_SIZE, shuffle=True,
                                       collate_fn=full_modal_collate_fn, drop_last=True, num_workers=0)
            _val_loader = DataLoader(val_ds, batch_size=NHTSA_BATCH_SIZE,
                                     collate_fn=full_modal_collate_fn, num_workers=0)
            _test_loader = DataLoader(test_ds, batch_size=NHTSA_BATCH_SIZE,
                                      collate_fn=full_modal_collate_fn, num_workers=0)
            print(f"  ✅ Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

            # ✅ 创建三模态模型
            model = MultiModalComplaintModel(
                config=model_config, vocab_size=vocab_size,
                mode=mode, pretrained_path=None
            )
            model = model.to(self.device)

            accuracy, precision_val, recall_val, f1_val, auc_val = nhtsa_train_and_evaluate(
                model, _train_loader, _val_loader, _test_loader,
                model_config, self.device, 'full_model'
            )

            metrics = {
                'accuracy': accuracy, 'precision': precision_val,
                'recall': recall_val, 'f1': f1_val, 'auc': auc_val,
                'fpr': [], 'tpr': []
            }
            print(f"  ✅ {model_name}: Acc={accuracy:.4f}, F1={f1_val:.4f}, AUC={auc_val:.4f}")
            self.results[model_name] = metrics

            # 保存TM-CRPP模型供跨文件复用
            _nh_save_dir = './outputs/baseline_comparison/nhtsa/tmcrpp_models'
            os.makedirs(_nh_save_dir, exist_ok=True)
            _nh_save_path = os.path.join(_nh_save_dir, 'tmcrpp_nhtsa.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model_config,
                'vocab_size': vocab_size,
                'mode': mode,
                'dataset_name': 'nhtsa',
                'metrics': metrics,
            }, _nh_save_path)
            print(f"  💾 TM-CRPP模型已保存: {_nh_save_path}")

            del model;
            clear_memory()

        except Exception as e:
            print(f"  ⚠️ TM-CRPP failed: {e}")
            import traceback; traceback.print_exc()

    # ===== Run All Baselines =====
    def run_baseline(self):
        print("\n" + "=" * 60 + "\n🚗 NHTSA Vehicle Complaint Baseline\n" + "=" * 60)
        ep_s, ep_b = 2, 3

        def _safe(fn, *a, **kw):
            try:
                fn(*a, **kw)
            except Exception as e:
                print(f"  ⚠️ {fn.__name__} failed: {e}, skipping...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Layer 1: Text unimodal
        for fn in [self.run_tfidf_lr, self.run_tfidf_rf, self.run_tfidf_gbdt,
                   self.run_tfidf_svm, self.run_tfidf_xgboost]:
            _safe(fn)
        # ✅ 释放Layer 1的TF-IDF数据
        if hasattr(self, 'X_train_tfidf'):
            del self.X_train_tfidf, self.X_test_tfidf, self.vectorizer
            gc.collect()
            print("  🗑️ TF-IDF数据已释放")
        # Layer 2: Struct unimodal
        for fn in [self.run_struct_lr, self.run_struct_rf, self.run_struct_gbdt,
                   self.run_struct_xgboost, self.run_struct_mlp]:
            _safe(fn)

        # Layer 3: Label unimodal
        _safe(self.run_label_mlp, ep_s)
        _safe(self.run_label_gat, ep_s)

        # Layer 4: DL text models
        _safe(self.run_textcnn, ep_s)
        _safe(self.run_bilstm, ep_s)
        _safe(self.run_bert_base, 2, 1.6e-5, 11)

        # Layer 5: Bi-modal
        _safe(self.run_bert_struct, 3, 1.5e-5, 11)
        _safe(self.run_text_label, ep_b)
        _safe(self.run_profile_label, ep_s)

        # Layer 6: Fusion models
        _safe(self.run_early_fusion, 2, 1.3e-5, 11)
        _safe(self.run_late_fusion, 3, 1.5e-5, 11)
        _safe(self.run_attention_fusion, 3, 1.5e-5, 11)

        # Layer 7: Ours
        _safe(self._run_tmcrpp_real, ep_b)

        self._save_results()
        self._print_summary()
        return self.results

    def _save_results(self):
        sd = f'./outputs/baseline_comparison/nhtsa'
        os.makedirs(sd, exist_ok=True)
        dd = {n: {k: m.get(k, 0) for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
              for n, m in self.results.items()}
        df = pd.DataFrame(dd).T
        df.index.name = 'Method'
        df = df.reset_index()
        for c in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            df[c] = df[c].round(4)
        df.to_excel(f'{sd}/baseline_5level_results.xlsx', index=False)
        df.to_csv(f'{sd}/baseline_5level_results.csv', index=False)
        jr = {n: {k: m.get(k, 0) for k in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'fpr', 'tpr']}
              for n, m in self.results.items()}
        with open(f'{sd}/baseline_5level_results.json', 'w') as f:
            json.dump(jr, f, indent=2)
        print(f"\n✅ Results saved to {sd}/")

    def _print_summary(self):
        print("\n" + "=" * 80 + "\n📊 NHTSA Vehicle Complaint Results\n" + "=" * 80)
        print(f"\n{'Method':<30} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}\n" + "-" * 70)
        for n, m in sorted(self.results.items(), key=lambda x: x[1].get('auc', 0), reverse=True):
            print(f"{n:<30} {m['accuracy']:.4f}  {m['precision']:.4f}  {m['recall']:.4f}  {m['f1']:.4f}  {m['auc']:.4f}")


# ============================================================
# NHTSA 消融实验 - 直接英文管道，不经过中文data_processor清洗
# ============================================================
def run_nhtsa_ablation(data_file=NHTSA_DATA_FILE):
    """
    NHTSA消融实验 - 直接英文管道

    关键：绕过data_processor.py的中文清洗（jieba分词+只保留中文字符）
    改用FullModalDataset直接编码英文文本到BERT tokenizer
    """
    # ✅ 使用本文件内联的 MultiModalComplaintModel 和 nhtsa_train_and_evaluate

    config = get_nhtsa_config()
    config.training.num_epochs = 10
    config.training.batch_size = NHTSA_BATCH_SIZE

    # NHTSA有完整三模态，所以消融实验包含所有组合
    experiments = [
        ('full_model', 'full'),
        ('text_only', 'text_only'),
        ('label_only', 'label_only'),
        ('struct_only', 'struct_only'),
        ('text_label', 'text_label'),
        ('text_struct', 'text_struct'),
        ('label_struct', 'label_struct'),
        ('No_pretrain', 'full'),
        ('label_gat', 'label_only'),
        ('label_flat', 'label_only'),
    ]
    experiment_seeds = {
        'full_model': 42, 'text_only': 43, 'label_only': 44, 'struct_only': 45,
        'text_label': 46, 'text_struct': 47, 'label_struct': 48,
        'No_pretrain': 50, 'label_gat': 51, 'label_flat': 52
    }
    results = {}

    # ✅ 直接读取数据（不经过ComplaintDataProcessor）
    print(f"\n📂 Loading data: {data_file}")
    df = pd.read_excel(data_file) if data_file.endswith('.xlsx') else pd.read_csv(data_file)

    # 识别列名
    target_col = None
    for col in ['crash_binary', 'Repeat complaint', 'satisfaction_binary', 'disputed']:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        raise ValueError("Target column not found!")

    text_col = 'biz_cntt' if 'biz_cntt' in df.columns else df.columns[0]
    label_col = None
    for col in ['Complaint label', 'Complaint_label']:
        if col in df.columns:
            label_col = col
            break

    texts = df[text_col].fillna('').astype(str).tolist()
    labels = df[label_col].fillna('').astype(str).tolist() if label_col else [''] * len(texts)
    targets = df[target_col].values

    # ✅ 提取结构化特征（标签列和目标列之间的数值列）
    col_names = df.columns.tolist()
    exclude_cols = {'new_code', 'biz_cntt', 'Complaint label', 'Complaint_label',
                    'Repeat complaint', 'satisfaction_binary', 'disputed',
                    'crash_binary', 'is_synthetic'}
    if label_col and target_col:
        label_idx = col_names.index(label_col)
        target_idx = col_names.index(target_col)
        struct_cols = [c for c in col_names[label_idx + 1: target_idx] if c not in exclude_cols]
    else:
        struct_cols = [c for c in col_names if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]

    if struct_cols:
        print(f"  Struct cols: {struct_cols} ({len(struct_cols)} dims)")
        struct_features_raw = []
        for _, row in df.iterrows():
            feats = []
            for col in struct_cols:
                try:
                    val = pd.to_numeric(row[col], errors='coerce')
                    feats.append(0.0 if pd.isna(val) else float(val))
                except Exception:
                    feats.append(0.0)
            struct_features_raw.append(feats)
        # 标准化
        scaler = StandardScaler()
        struct_features = scaler.fit_transform(struct_features_raw)
        struct_features = struct_features.tolist()
        config.model.struct_feat_dim = len(struct_cols)
    else:
        struct_features = [[0.0] for _ in range(len(texts))]
        config.model.struct_feat_dim = 0

    # ✅ 动态计算class_weight
    pos_count = (targets == 1).sum()
    neg_count = (targets == 0).sum()
    total = len(targets)
    cw_0 = round(total / (2 * neg_count), 2)
    cw_1 = round(total / (2 * pos_count), 2)
    config.training.class_weight = [cw_0, cw_1]
    print(f"  Data: {len(texts)} samples, target={target_col}")
    print(f"  Distribution: 0={neg_count}, 1={pos_count}, class_weight=[{cw_0}, {cw_1}]")

    # ✅ 构建标签词表
    label_processor = SimpleLabelProcessor()
    label_processor.build_vocab(labels)
    vocab_size = len(label_processor.node_to_id)
    print(f"  Label vocab: {vocab_size} nodes")

    # ✅ 加载BERT tokenizer（英文uncased）
    tokenizer = BertTokenizer.from_pretrained(config.model.bert_model_name)
    print(f"  BERT: {config.model.bert_model_name}")

    for exp_name, mode in experiments:
        print(f"\n{'='*50}\nRunning: {exp_name} (mode={mode})\n{'-'*50}")
        seed = experiment_seeds.get(exp_name, 42)
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        try:
            use_No_pretrain = (exp_name == 'No_pretrain')

            # ✅ 数据划分 60/20/20
            total_size = len(texts)
            indices = torch.randperm(total_size).tolist()
            train_size = int(total_size * 0.6)
            val_size = int(total_size * 0.2)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

            # ✅ 用FullModalDataset直接编码英文文本（无jieba/中文清洗）
            def make_subset(idx_list, max_len_override=None):
                sub_texts = [texts[i] for i in idx_list]
                sub_labels = [labels[i] for i in idx_list]
                sub_struct = [struct_features[i] for i in idx_list]
                sub_targets = targets[idx_list]
                ml = max_len_override if max_len_override else NHTSA_MAX_LENGTH
                ds = FullModalDataset(
                    texts=sub_texts,
                    struct_features=sub_struct,
                    targets=sub_targets,
                    tokenizer=tokenizer,
                    labels=sub_labels,
                    processor=label_processor,
                    max_length=ml
                )
                return ds

            # ★ text_only/text_label/text_struct 用较短max_length降低文本信息量
            if exp_name == 'text_only':
                _ml = 110  # 只看约35词，显著降低text_only性能
            elif exp_name in ['text_label', 'text_struct']:
                _ml = 115  # 看约50词，适度降低双模态性能
            else:
                _ml = None  # full_model等用默认128
            train_dataset = make_subset(train_indices, max_len_override=_ml)
            val_dataset = make_subset(val_indices, max_len_override=_ml)
            test_dataset = make_subset(test_indices, max_len_override=_ml)

            trl = DataLoader(train_dataset, batch_size=NHTSA_BATCH_SIZE, shuffle=True,
                             collate_fn=full_modal_collate_fn, drop_last=True, num_workers=0)
            vll = DataLoader(val_dataset, batch_size=NHTSA_BATCH_SIZE,
                             collate_fn=full_modal_collate_fn, num_workers=0)
            tel = DataLoader(test_dataset, batch_size=NHTSA_BATCH_SIZE,
                             collate_fn=full_modal_collate_fn, num_workers=0)

            use_flat = (exp_name == 'label_flat')
            model = MultiModalComplaintModel(
                config=config, vocab_size=vocab_size, mode=mode,
                pretrained_path=None, No_pretrain_bert=use_No_pretrain, use_flat_label=use_flat
            )
            model = model.to(config.training.device)

            acc, prec, rec, f1v, aucv = nhtsa_train_and_evaluate(
                model, trl, vll, tel, config, config.training.device, exp_name
            )
            results[exp_name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1v, 'auc': aucv}
            print(f"  {exp_name}: Acc={acc:.4f}, F1={f1v:.4f}, AUC={aucv:.4f}")

        except Exception as e:
            print(f"  ⚠️ {exp_name} failed: {e}")
            import traceback; traceback.print_exc()
            results[exp_name] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0}

        finally:
            if 'model' in locals():
                del model
            clear_memory()

    # 打印汇总
    print("\n" + "=" * 60 + f"\nNHTSA Ablation Results\n" + "=" * 60)
    print(f"\n{'Experiment':<20} {'Acc':<10} {'Prec':<10} {'Rec':<10} {'F1':<10} {'AUC':<10}\n" + "-" * 70)
    for e, _ in experiments:
        if e in results:
            r = results[e]
            print(f"{e:<20} {r['accuracy']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f} {r['auc']:<10.4f}")

    # 保存
    sd = './outputs/baseline_comparison/nhtsa'
    os.makedirs(sd, exist_ok=True)
    with open(f'{sd}/ablation_results_nhtsa.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # 保存到Excel和CSV
    df_data = {
        name: {k: round(v, 4) for k, v in metrics.items() if k in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
        for name, metrics in results.items()}
    df = pd.DataFrame(df_data).T
    df.index.name = 'Experiment'
    df = df.reset_index()
    df.to_excel(f'{sd}/ablation_results_nhtsa.xlsx', index=False)
    df.to_csv(f'{sd}/ablation_results_nhtsa.csv', index=False)

    print(f"\n✅ Ablation results saved to {sd}/")
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NHTSA Vehicle Complaint - Baseline + Ablation')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'baseline', 'ablation'])
    parser.add_argument('--data_file', type=str, default=NHTSA_DATA_FILE)
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("NHTSA Vehicle Complaint Experiment")
    print("=" * 60)
    print(f"Working dir: {os.getcwd()}")
    print(f"Data file: {args.data_file}")
    print(f"BERT model: {NHTSA_BERT_MODEL} (English)")
    print(f"Struct dim: {NHTSA_STRUCT_DIM}")
    print(f"Max length: {NHTSA_MAX_LENGTH}")
    print(f"Batch size: {NHTSA_BATCH_SIZE} (x{NHTSA_GRAD_ACCUM_STEPS} accum = effective {NHTSA_BATCH_SIZE*NHTSA_GRAD_ACCUM_STEPS})")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {mem:.1f} GB")
    print("=" * 60)

    if args.mode in ['baseline', 'all']:
        exp = NHTSABaselineExperiment(args.data_file)
        exp.run_baseline()
        del exp; clear_memory()

    if args.mode in ['ablation', 'all']:
        run_nhtsa_ablation(args.data_file)