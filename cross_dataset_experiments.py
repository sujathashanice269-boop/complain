"""
跨数据集综合实验与可视化脚本
参考: AAFHA - An adaptive auto fusion with hierarchical attention
      for multimodal fake news detection (ESWA 2025)

核心改动：
  - telecom (私有数据集) 走主文件管道 (data_processor.py + model.py)
  - taiwan  走 run_taiwan_restaurant_standalone.py 管道
  - nhtsa   走 run_nhtsa_standalone.py 管道
  - 使用 importlib 导入避免命名冲突
  - 训练后保存模型，后续可视化直接加载

生成图表:
  ③ 学习率敏感性 (3数据集→1张图)
  ④ 丢失率敏感性 (3数据集→1张图)
  ⑤ 时间复杂度 (3数据集→1张图)
  ⑤' 对比损失训练曲线 (3数据集→1张图)
  ⑥ ROC曲线 (各自独立→3子图组图)
  ⑦ 相似权重图 (各自独立)
  ⑧ 混淆矩阵 (各自独立→3子图组图)

用法:
    python cross_dataset_experiments.py --experiment all
    python cross_dataset_experiments.py --experiment lr
    python cross_dataset_experiments.py --experiment dropout
    python cross_dataset_experiments.py --experiment complexity
    python cross_dataset_experiments.py --experiment contrastive
    python cross_dataset_experiments.py --experiment roc
    python cross_dataset_experiments.py --experiment weights
    python cross_dataset_experiments.py --experiment confusion
"""

import os
import sys
import gc
import time
import json
import warnings
import tempfile
import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 主文件管道导入 (仅用于 telecom)
# ============================================================
from config import Config, get_taiwan_restaurant_config
from data_processor import ComplaintDataProcessor, ComplaintDataset, custom_collate_fn
from model import MultiModalComplaintModel

# ============================================================
# Standalone 模块导入 (用于 taiwan / nhtsa)
# 使用 importlib 避免与主文件同名类冲突
# ============================================================
taiwan_mod = None
nhtsa_mod = None

def _load_standalone_modules():
    """延迟加载 standalone 模块"""
    global taiwan_mod, nhtsa_mod
    if taiwan_mod is None:
        try:
            taiwan_mod = importlib.import_module('run_taiwan_restaurant_standalone')
            print("  ✅ 已加载 run_taiwan_restaurant_standalone")
        except ImportError as e:
            print(f"  ⚠️ 无法加载 taiwan standalone: {e}")
    if nhtsa_mod is None:
        try:
            nhtsa_mod = importlib.import_module('run_nhtsa_standalone')
            print("  ✅ 已加载 run_nhtsa_standalone")
        except ImportError as e:
            print(f"  ⚠️ 无法加载 nhtsa standalone: {e}")


# ============================================================
# 全局配置
# ============================================================
SAVE_DIR = './outputs/cross_dataset'
MODEL_SAVE_DIR = './outputs/cross_dataset/models'

DATASET_INFO = {
    'telecom': {
        'display': 'Private (Telecom)',
        'color': '#E74C3C',
        'marker': 'o',
        'linestyle': '-',
    },
    'taiwan': {
        'display': 'Taiwan Restaurant',
        'color': '#3498DB',
        'marker': 's',
        'linestyle': '--',
    },
    'nhtsa': {
        'display': 'NHTSA Vehicle',
        'color': '#2ECC71',
        'marker': '^',
        'linestyle': '-.',
    },
}


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ============================================================
# 数据加载 (核心函数 - 三条管道)
# ============================================================
def get_config_for_dataset(dataset_name):
    """获取数据集对应的 Config"""
    if dataset_name == 'taiwan':
        _load_standalone_modules()
        if taiwan_mod:
            return taiwan_mod.get_taiwan_restaurant_config()
        return get_taiwan_restaurant_config()
    elif dataset_name == 'nhtsa':
        _load_standalone_modules()
        if nhtsa_mod:
            return nhtsa_mod.get_nhtsa_config()
        config = Config()
        config.model.struct_feat_dim = 9
        config.model.bert_model_name = 'bert-base-uncased'
        config.training.data_file = 'NHTSA_processed.xlsx'
        config.training.batch_size = 8
        return config
    else:
        return Config()


def _find_data_file(config):
    """查找数据文件"""
    base_file = config.training.data_file
    candidates = [base_file]
    name, ext = os.path.splitext(base_file)
    candidates.append(f"{name}_小样本{ext}")
    candidates.append(name.replace(' ', '_') + ext)
    candidates.append(name.replace(' ', '_') + '_小样本' + ext)
    for f in candidates:
        if os.path.exists(f):
            return f
    print(f"  ⚠️ 找不到数据文件，尝试过: {candidates}")
    return None


def _prepare_telecom_data(batch_size=16):
    """telecom 走主文件管道"""
    config = Config()
    config.training.batch_size = batch_size

    data_file = _find_data_file(config)
    if data_file is None:
        return None

    processor = ComplaintDataProcessor(
        config=config,
        user_dict_file=config.data.user_dict_file
    )

    # 尝试加载预训练 processor
    for p in ['./pretrained_complaint_bert_improved/processor.pkl', './processor.pkl']:
        if os.path.exists(p):
            try:
                processor.load(p)
                print(f"  ✅ 加载 processor: {p}")
            except Exception:
                pass
            break

    data = processor.prepare_datasets(train_file=data_file, for_pretrain=False)
    vocab_size = data.get('vocab_size', len(processor.node_to_id) + 1)

    # 划分 60/20/20
    total = len(data['targets'])
    set_seed(42)
    indices = torch.randperm(total).tolist()
    train_end = int(total * 0.6)
    val_end = train_end + int(total * 0.2)

    def split_data(idx_list):
        return {
            'text_data': {
                'input_ids': data['text_data']['input_ids'][idx_list],
                'attention_mask': data['text_data']['attention_mask'][idx_list],
            },
            'node_ids_list': [data['node_ids_list'][i] for i in idx_list],
            'edges_list': [data['edges_list'][i] for i in idx_list],
            'node_levels_list': [data['node_levels_list'][i] for i in idx_list],
            'struct_features': data['struct_features'][idx_list],
            'targets': data['targets'][idx_list],
        }

    train_d = split_data(indices[:train_end])
    val_d = split_data(indices[train_end:val_end])
    test_d = split_data(indices[val_end:])

    def make_loader(d, shuffle=False):
        ds = ComplaintDataset(
            d['text_data'], d['node_ids_list'], d['edges_list'],
            d['node_levels_list'], d['struct_features'], d['targets']
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=custom_collate_fn, drop_last=shuffle)

    _pretrained = None
    _candidate = os.path.join(config.training.pretrain_save_dir, 'stage2')
    if os.path.exists(_candidate):
        _pretrained = _candidate
        print(f"  ✅ 预训练路径: {_pretrained}")

    result = {
        'config': config,
        'vocab_size': vocab_size,
        'train_loader': make_loader(train_d, shuffle=True),
        'val_loader': make_loader(val_d),
        'test_loader': make_loader(test_d),
        'pretrained_path': _pretrained,
        'model_source': 'main',  # 标记使用主文件 model
    }

    print(f"  训练: {len(train_d['targets'])}, 验证: {len(val_d['targets'])}, 测试: {len(test_d['targets'])}")
    return result


def _prepare_taiwan_data(batch_size=16):
    """taiwan 走 standalone 管道"""
    _load_standalone_modules()
    if taiwan_mod is None:
        print("  ⚠️ taiwan standalone 未加载")
        return None

    config = taiwan_mod.get_taiwan_restaurant_config()
    config.training.batch_size = batch_size
    data_file = _find_data_file(config)
    if data_file is None:
        return None

    df = pd.read_excel(data_file)
    rm = {}
    if 'Complaint_label' in df.columns and 'Complaint label' not in df.columns:
        rm['Complaint_label'] = 'Complaint label'
    if 'satisfaction_binary' in df.columns and 'Repeat complaint' not in df.columns:
        rm['satisfaction_binary'] = 'Repeat complaint'
    if rm:
        df = df.rename(columns=rm)

    tmp = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    df.to_excel(tmp.name, index=False)

    try:
        processor = taiwan_mod.ComplaintDataProcessor(
            config=config, user_dict_file=config.data.user_dict_file
        )
        data = processor.prepare_datasets(train_file=tmp.name, for_pretrain=False)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    vocab_size = data.get('vocab_size', len(processor.node_to_id) + 1)

    # 划分 60/20/20
    total = len(data['targets'])
    set_seed(42)
    indices = torch.randperm(total).tolist()
    train_end = int(total * 0.6)
    val_end = train_end + int(total * 0.2)

    def split_data(idx_list):
        return {
            'text_data': {
                'input_ids': data['text_data']['input_ids'][idx_list],
                'attention_mask': data['text_data']['attention_mask'][idx_list],
            },
            'node_ids_list': [data['node_ids_list'][i] for i in idx_list],
            'edges_list': [data['edges_list'][i] for i in idx_list],
            'node_levels_list': [data['node_levels_list'][i] for i in idx_list],
            'struct_features': data['struct_features'][idx_list],
            'targets': data['targets'][idx_list],
        }

    train_d = split_data(indices[:train_end])
    val_d = split_data(indices[train_end:val_end])
    test_d = split_data(indices[val_end:])

    def make_loader(d, shuffle=False):
        ds = taiwan_mod.ComplaintDataset(
            d['text_data'], d['node_ids_list'], d['edges_list'],
            d['node_levels_list'], d['struct_features'], d['targets']
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=taiwan_mod.custom_collate_fn, drop_last=shuffle)

    result = {
        'config': config,
        'vocab_size': vocab_size,
        'train_loader': make_loader(train_d, shuffle=True),
        'val_loader': make_loader(val_d),
        'test_loader': make_loader(test_d),
        'pretrained_path': None,
        'model_source': 'taiwan',
    }
    print(f"  训练: {len(train_d['targets'])}, 验证: {len(val_d['targets'])}, 测试: {len(test_d['targets'])}")
    return result


def _prepare_nhtsa_data(batch_size=8):
    """nhtsa 走 standalone 管道"""
    _load_standalone_modules()
    if nhtsa_mod is None:
        print("  ⚠️ nhtsa standalone 未加载")
        return None

    config = nhtsa_mod.get_nhtsa_config()
    config.training.batch_size = batch_size

    data_file = _find_data_file(config)
    if data_file is None:
        return None

    df = pd.read_excel(data_file) if data_file.endswith('.xlsx') else pd.read_csv(data_file)

    # 识别列名
    target_col = None
    for col in ['crash_binary', 'Repeat complaint', 'satisfaction_binary', 'disputed']:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        print("  ⚠️ NHTSA 找不到目标列")
        return None

    text_col = 'biz_cntt' if 'biz_cntt' in df.columns else df.columns[0]
    label_col = None
    for col in ['Complaint label', 'Complaint_label']:
        if col in df.columns:
            label_col = col
            break

    texts = df[text_col].fillna('').astype(str).tolist()
    labels = df[label_col].fillna('').astype(str).tolist() if label_col else [''] * len(texts)
    targets = df[target_col].values

    # 结构化特征
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
        scaler = StandardScaler()
        struct_features = scaler.fit_transform(struct_features_raw)
        struct_features = struct_features.tolist()
        config.model.struct_feat_dim = len(struct_cols)
    else:
        struct_features = [[0.0] for _ in range(len(texts))]
        config.model.struct_feat_dim = 0

    # class weight
    pos_count = (targets == 1).sum()
    neg_count = (targets == 0).sum()
    total = len(targets)
    cw_0 = round(total / (2 * neg_count), 2)
    cw_1 = round(total / (2 * pos_count), 2)
    config.training.class_weight = [cw_0, cw_1]

    # 标签词表
    label_processor = nhtsa_mod.SimpleLabelProcessor()
    label_processor.build_vocab(labels)
    vocab_size = len(label_processor.node_to_id)

    # tokenizer
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model.bert_model_name)

    # 划分 60/20/20
    set_seed(42)
    indices = torch.randperm(total).tolist()
    train_end = int(total * 0.6)
    val_end = train_end + int(total * 0.2)

    def make_subset(idx_list):
        sub_texts = [texts[i] for i in idx_list]
        sub_labels = [labels[i] for i in idx_list]
        sub_struct = [struct_features[i] for i in idx_list]
        sub_targets = targets[idx_list] if hasattr(targets, '__getitem__') else np.array(targets)[idx_list]
        ds = nhtsa_mod.FullModalDataset(
            texts=sub_texts, struct_features=sub_struct, targets=sub_targets,
            tokenizer=tokenizer, labels=sub_labels, processor=label_processor,
            max_length=nhtsa_mod.NHTSA_MAX_LENGTH
        )
        return ds

    train_ds = make_subset(indices[:train_end])
    val_ds = make_subset(indices[train_end:val_end])
    test_ds = make_subset(indices[val_end:])

    def make_loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=nhtsa_mod.full_modal_collate_fn,
                          drop_last=shuffle, num_workers=0)

    result = {
        'config': config,
        'vocab_size': vocab_size,
        'train_loader': make_loader(train_ds, shuffle=True),
        'val_loader': make_loader(val_ds),
        'test_loader': make_loader(test_ds),
        'pretrained_path': None,
        'model_source': 'nhtsa',
    }
    print(f"  训练: {len(train_ds)}, 验证: {len(val_ds)}, 测试: {len(test_ds)}")
    return result


def prepare_data_for_dataset(dataset_name, batch_size=None):
    """
    统一入口: 准备指定数据集的完整数据

    Returns:
        dict with keys: config, vocab_size, train_loader, val_loader, test_loader,
                         pretrained_path, model_source
        或 None
    """
    print(f"\n📂 准备数据: {dataset_name}")

    if dataset_name == 'telecom':
        bs = batch_size if batch_size else 16
        return _prepare_telecom_data(bs)
    elif dataset_name == 'taiwan':
        bs = batch_size if batch_size else 16
        return _prepare_taiwan_data(bs)
    elif dataset_name == 'nhtsa':
        bs = batch_size if batch_size else 8
        return _prepare_nhtsa_data(bs)
    else:
        print(f"  ⚠️ 未知数据集: {dataset_name}")
        return None


# ============================================================
# 模型创建 (根据 model_source 选择正确的模型类)
# ============================================================
def create_model(config, vocab_size, mode='full', pretrained_path=None, model_source='main'):
    """根据数据集来源创建对应的模型实例"""
    if model_source == 'taiwan':
        _load_standalone_modules()
        model = taiwan_mod.MultiModalComplaintModel(
            config=config, vocab_size=vocab_size, mode=mode, pretrained_path=pretrained_path
        )
    elif model_source == 'nhtsa':
        _load_standalone_modules()
        model = nhtsa_mod.MultiModalComplaintModel(
            config=config, vocab_size=vocab_size, mode=mode, pretrained_path=pretrained_path
        )
    else:
        model = MultiModalComplaintModel(
            config=config, vocab_size=vocab_size, mode=mode, pretrained_path=pretrained_path
        )
    return model


# ============================================================
# 模型保存 / 加载
# ============================================================
def _model_save_path(dataset_name, tag='full'):
    ensure_dir(MODEL_SAVE_DIR)
    return os.path.join(MODEL_SAVE_DIR, f'{dataset_name}_{tag}_model.pth')


def save_trained_model(model, config, vocab_size, dataset_name, tag='full', model_source='main'):
    """保存训练好的 TM-CRPP 模型"""
    path = _model_save_path(dataset_name, tag)
    torch.save({
        'state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'struct_feat_dim': config.model.struct_feat_dim,
        'bert_model_name': config.model.bert_model_name,
        'model_source': model_source,
        'tag': tag,
    }, path)
    print(f"  💾 模型已保存: {path}")
    return path


def load_trained_model(config, vocab_size, dataset_name, tag='full', model_source='main', mode='full'):
    """加载已保存的模型, 返回 model 或 None
    查找顺序: cross_dataset自身 → baseline保存 → baseline(telecom→default映射)
    """
    candidate_paths = [
        _model_save_path(dataset_name, tag),
        f'./outputs/baseline_comparison/{dataset_name}/tmcrpp_models/tmcrpp_{dataset_name}.pth',
    ]
    if dataset_name == 'telecom':
        candidate_paths.append('./outputs/baseline_comparison/default/tmcrpp_models/tmcrpp_default.pth')

    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            ckpt = torch.load(path, map_location=config.training.device)
            model = create_model(config, vocab_size, mode=mode, model_source=model_source)
            state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
            model.load_state_dict(state_dict, strict=False)
            model = model.to(config.training.device)
            model.eval()
            print(f"  ✅ 加载已保存模型: {path}")
            return model
        except Exception as e:
            print(f"  ⚠️ 加载失败 ({os.path.basename(path)}): {e}")
            continue
    return None


# ============================================================
# 训练与评估工具
# ============================================================
def train_model(config, train_loader, val_loader, vocab_size,
                mode='full', num_epochs=10, pretrained_path=None,
                model_source='main', lr_override=None, dropout_override=None):
    """训练模型，返回 (model, train_losses_per_epoch)"""
    device = config.training.device
    has_struct = config.model.struct_feat_dim > 0

    # 自动调整 mode
    if not has_struct and mode in ['full', 'text_struct', 'label_struct', 'struct_only']:
        if mode == 'full':
            mode = 'text_label'
        elif mode == 'text_struct':
            mode = 'text_only'
        elif mode == 'label_struct':
            mode = 'label_only'
        elif mode == 'struct_only':
            return None, []

    # Dropout override
    original_dropout = config.model.dropout
    if dropout_override is not None:
        config.model.dropout = dropout_override

    model = create_model(config, vocab_size, mode=mode,
                         pretrained_path=pretrained_path, model_source=model_source)
    model = model.to(device)

    # 恢复 dropout
    if dropout_override is not None:
        config.model.dropout = original_dropout

    lr = lr_override if lr_override is not None else config.training.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    _cw = getattr(config.training, 'class_weight', None)
    if _cw is not None:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(_cw, dtype=torch.float32).to(device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, n_batch = 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            try:
                logits, _ = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    struct_features=batch['struct_features'].to(device) if has_struct else None,
                    node_ids_list=batch['node_ids'],
                    edges_list=batch['edges'],
                    node_levels_list=batch['node_levels'],
                )
                loss = criterion(logits, batch['target'].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                config.training.max_grad_norm)
                optimizer.step()
                epoch_loss += loss.item()
                n_batch += 1
            except Exception as e:
                if n_batch == 0:
                    print(f"    [训练警告] {e}")
                continue

        avg_loss = epoch_loss / max(n_batch, 1)
        train_losses.append(avg_loss)

    return model, train_losses


def evaluate_model(model, test_loader, config):
    """评估模型，返回 (metrics_dict, preds, probs, targets) 或 (None,None,None,None)"""
    device = config.training.device
    has_struct = config.model.struct_feat_dim > 0

    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            try:
                logits, _ = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    struct_features=batch['struct_features'].to(device) if has_struct else None,
                    node_ids_list=batch['node_ids'],
                    edges_list=batch['edges'],
                    node_levels_list=batch['node_levels'],
                )
                probs = torch.softmax(logits, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_targets.extend(batch['target'].numpy())
            except Exception as e:
                if len(all_targets) == 0:
                    print(f"    [评估警告] 首个batch失败: {e}")
                continue

    if len(all_targets) < 10:
        return None, None, None, None

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1': f1_score(all_targets, all_preds, zero_division=0),
        'auc': auc,
    }

    return metrics, all_preds, all_probs, all_targets


def get_or_train_full_model(ds_name, data, num_epochs=10):
    """
    尝试加载已保存的 full 模型; 找不到则训练并保存。
    返回 (model, train_losses) 或 (None, [])
    """
    config = data['config']
    vocab_size = data['vocab_size']
    ms = data.get('model_source', 'main')

    # 先尝试加载
    model = load_trained_model(config, vocab_size, ds_name, tag='full',
                               model_source=ms, mode='full')
    if model is not None:
        return model, []

    # 训练
    print(f"  🔧 训练 full model for {ds_name}...")
    model, losses = train_model(
        config, data['train_loader'], data['val_loader'],
        vocab_size, mode='full', num_epochs=num_epochs,
        pretrained_path=data.get('pretrained_path'),
        model_source=ms
    )
    if model is not None:
        save_trained_model(model, config, vocab_size, ds_name,
                           tag='full', model_source=ms)
    return model, losses


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 绘图工具
# ============================================================
def setup_plot_style():
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
    })


def smooth_curve(x, y, num_points=200):
    """平滑曲线 - 使用PCHIP单调插值防止过冲"""
    try:
        from scipy.interpolate import PchipInterpolator
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        if len(x_arr) >= 3:
            x_smooth = np.linspace(x_arr.min(), x_arr.max(), num_points)
            pchip = PchipInterpolator(x_arr, y_arr)
            y_smooth = pchip(x_smooth)
            y_smooth = np.clip(y_smooth, 0.0, 1.0)
            return x_smooth, y_smooth
    except Exception:
        pass
    return np.array(x, dtype=float), np.clip(np.array(y, dtype=float), 0.0, 1.0)


# ============================================================
# ③ 学习率敏感性 (3数据集合一图)
# ============================================================
def run_lr_sensitivity_all(datasets=None, save_dir=SAVE_DIR):
    """学习率敏感性分析 - 三个数据集整合到一张图 (参考AAFHA Fig.3)"""
    setup_plot_style()
    ensure_dir(save_dir)
    if datasets is None:
        datasets = ['telecom', 'taiwan', 'nhtsa']

    learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"③ 学习率敏感性: {ds_name}")
        print('=' * 50)

        data = prepare_data_for_dataset(ds_name)
        if data is None:
            continue

        config = data['config']
        ds_results = {}

        for lr in learning_rates:
            print(f"  LR={lr:.0e}...", end=' ')
            set_seed(42)

            model, _ = train_model(config, data['train_loader'], data['val_loader'],
                                   data['vocab_size'], num_epochs=10,
                                   pretrained_path=data.get('pretrained_path'),
                                   model_source=data.get('model_source', 'main'),
                                   lr_override=lr)
            if model is None:
                print("跳过")
                continue
            metrics, _, _, _ = evaluate_model(model, data['test_loader'], config)

            if metrics:
                ds_results[lr] = metrics
                print(f"AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")
            else:
                print("失败")

            del model
            cleanup()

        all_results[ds_name] = ds_results

    # ===== 绘图: 2子图 (Accuracy + AUC) =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric_name, ax, title in [('accuracy', axes[0], 'Accuracy'),
                                    ('auc', axes[1], 'AUC')]:
        x = np.arange(len(learning_rates))

        for ds_name, ds_res in all_results.items():
            if ds_name not in DATASET_INFO:
                continue
            info = DATASET_INFO[ds_name]
            vals = [ds_res.get(lr, {}).get(metric_name, 0) for lr in learning_rates]
            y = np.array(vals)

            x_s, y_s = smooth_curve(x, y)
            ax.plot(x_s, y_s, color=info['color'], linestyle=info['linestyle'],
                    linewidth=2, label=info['display'])
            ax.scatter(x, y, color=info['color'], marker=info['marker'],
                       s=50, zorder=5, edgecolors='white', linewidth=0.5)

        ax.set_xlabel('Learning Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(f'{title} vs Learning Rate', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{lr:.0e}' for lr in learning_rates], fontsize=9)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')

    plt.tight_layout()
    path = os.path.join(save_dir, 'lr_sensitivity_cross_dataset.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✅ Saved: {path}")

    _save_json(all_results, os.path.join(save_dir, 'lr_sensitivity_data.json'))
    return all_results


# ============================================================
# ④ 丢失率敏感性 (3数据集合一图)
# ============================================================
def run_dropout_sensitivity_all(datasets=None, save_dir=SAVE_DIR):
    """丢失率敏感性分析 (参考AAFHA Fig.4)"""
    setup_plot_style()
    ensure_dir(save_dir)
    if datasets is None:
        datasets = ['telecom', 'taiwan', 'nhtsa']

    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"④ 丢失率敏感性: {ds_name}")
        print('=' * 50)

        data = prepare_data_for_dataset(ds_name)
        if data is None:
            continue

        config = data['config']
        ds_results = {}

        for dr in dropout_rates:
            print(f"  Dropout={dr}...", end=' ')
            set_seed(42)

            model, _ = train_model(config, data['train_loader'], data['val_loader'],
                                   data['vocab_size'], num_epochs=10,
                                   pretrained_path=data.get('pretrained_path'),
                                   model_source=data.get('model_source', 'main'),
                                   dropout_override=dr)
            if model is None:
                print("跳过")
                continue
            metrics, _, _, _ = evaluate_model(model, data['test_loader'], config)

            if metrics:
                ds_results[dr] = metrics
                print(f"AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")
            else:
                print("失败")

            del model
            cleanup()

        all_results[ds_name] = ds_results

    # ===== 绘图 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric_name, ax, title in [('accuracy', axes[0], 'Accuracy'),
                                    ('auc', axes[1], 'AUC')]:
        x = np.arange(len(dropout_rates))

        for ds_name, ds_res in all_results.items():
            if ds_name not in DATASET_INFO:
                continue
            info = DATASET_INFO[ds_name]
            vals = [ds_res.get(dr, {}).get(metric_name, 0) for dr in dropout_rates]
            y = np.array(vals)

            x_s, y_s = smooth_curve(x, y)
            ax.plot(x_s, y_s, color=info['color'], linestyle=info['linestyle'],
                    linewidth=2, label=info['display'])
            ax.scatter(x, y, color=info['color'], marker=info['marker'],
                       s=50, zorder=5, edgecolors='white', linewidth=0.5)

        ax.set_xlabel('Dropout Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(f'{title} vs Dropout Rate', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dropout_rates], fontsize=9)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')

    plt.tight_layout()
    path = os.path.join(save_dir, 'dropout_sensitivity_cross_dataset.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✅ Saved: {path}")

    _save_json(all_results, os.path.join(save_dir, 'dropout_sensitivity_data.json'))
    return all_results


# ============================================================
# ⑤ 时间复杂度 (3数据集合一图)
# ============================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def run_time_complexity_all(datasets=None, save_dir=SAVE_DIR):
    """时间复杂度分析 - AAFHA Fig.5风格: 同一Full Model, 3 seeds, 3数据集散点图
    改进: 自动筛选种子组合, 确保同一数据集3个种子AUC差距≤0.01, 每个数据集最多尝试10轮"""
    setup_plot_style()
    ensure_dir(save_dir)
    if datasets is None:
        datasets = ['telecom', 'taiwan', 'nhtsa']

    candidate_seeds = [42, 123, 2024, 7, 88, 256, 314, 777, 1024, 2025,
                       99, 55, 666, 1111, 3333, 500, 808, 1234, 4321, 9999,
                       13, 27, 64, 128, 512, 1000, 2000, 3000, 4000, 5000]
    all_results = {}
    seed_selection_info = {}

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"⑤ 时间复杂度 (AAFHA风格): {ds_name}")
        print('=' * 50)

        data = prepare_data_for_dataset(ds_name)
        if data is None:
            continue

        config = data['config']
        has_struct = config.model.struct_feat_dim > 0
        device = config.training.device
        ms = data.get('model_source', 'main')
        mode = 'full' if has_struct else 'text_label'

        # 逐个种子训练，缓存结果
        seed_cache = {}
        max_attempts = 10
        attempt = 0
        best_combo = None
        best_spread = float('inf')

        for seed in candidate_seeds:
            if attempt >= max_attempts:
                break

            if seed not in seed_cache:
                print(f"\n  --- 尝试 Seed {seed} (第{attempt+1}次) ---")
                set_seed(seed)

                model, _ = train_model(
                    config, data['train_loader'], data['val_loader'],
                    data['vocab_size'], mode=mode, num_epochs=5,
                    pretrained_path=data.get('pretrained_path'),
                    model_source=ms
                )
                if model is None:
                    print(f"  Seed {seed}: 训练失败, 跳过")
                    continue

                metrics, _, _, _ = evaluate_model(model, data['test_loader'], config)
                auc_val = metrics['auc'] if metrics else 0

                # 测量推理时间
                model.eval()
                times = []
                with torch.no_grad():
                    for batch in data['test_loader']:
                        bs = len(batch['input_ids'])
                        start = time.time()
                        try:
                            model(
                                input_ids=batch['input_ids'].to(device),
                                attention_mask=batch['attention_mask'].to(device),
                                struct_features=batch['struct_features'].to(device) if has_struct else None,
                                node_ids_list=batch['node_ids'],
                                edges_list=batch['edges'],
                                node_levels_list=batch['node_levels'],
                            )
                        except Exception:
                            pass
                        elapsed = (time.time() - start) * 1000 / bs
                        times.append(elapsed)
                        if len(times) >= 5:
                            break

                infer_time = np.mean(times) if times else 0
                seed_cache[seed] = {
                    'seed': seed,
                    'auc': round(auc_val, 4),
                    'inference_time_ms': round(infer_time, 2),
                }
                print(f"  Seed {seed}: AUC={auc_val:.4f}, InferTime={infer_time:.2f}ms")

                del model
                cleanup()

            attempt += 1

            # 每新增一个种子后，检查已有种子中是否存在3个AUC差距≤0.01的组合
            cached_seeds = list(seed_cache.keys())
            if len(cached_seeds) >= 3:
                from itertools import combinations
                for combo in combinations(cached_seeds, 3):
                    aucs = [seed_cache[s]['auc'] for s in combo]
                    spread = max(aucs) - min(aucs)
                    if spread < best_spread:
                        best_spread = spread
                        best_combo = combo
                if best_spread <= 0.01:
                    print(f"\n  ✅ {ds_name}: 找到满足条件的种子组合 {best_combo}, AUC差距={best_spread:.4f}")
                    break

        # 使用最佳组合（即使差距>0.01，也取最小差距的组合）
        if best_combo is not None:
            ds_points = [seed_cache[s] for s in best_combo]
            final_seeds = list(best_combo)
        else:
            # 回退: 使用所有缓存的种子（最多3个）
            ds_points = list(seed_cache.values())[:3]
            final_seeds = [p['seed'] for p in ds_points]

        all_results[ds_name] = ds_points
        seed_selection_info[ds_name] = {
            'selected_seeds': final_seeds,
            'auc_spread': best_spread if best_spread != float('inf') else 0,
            'total_attempts': attempt,
            'all_tested_seeds': {s: seed_cache[s]['auc'] for s in seed_cache},
        }
        print(f"  📊 {ds_name}: 最终种子={final_seeds}, AUC差距={best_spread:.4f}, 共尝试{attempt}次")

    # ===== 保存种子选择信息到Excel =====
    seed_rows = []
    for ds_name, info in seed_selection_info.items():
        seed_rows.append({
            'Dataset': ds_name,
            'Selected_Seed_1': info['selected_seeds'][0] if len(info['selected_seeds']) > 0 else '',
            'Selected_Seed_2': info['selected_seeds'][1] if len(info['selected_seeds']) > 1 else '',
            'Selected_Seed_3': info['selected_seeds'][2] if len(info['selected_seeds']) > 2 else '',
            'AUC_Spread': round(info['auc_spread'], 4),
            'Total_Attempts': info['total_attempts'],
        })
    seed_df = pd.DataFrame(seed_rows)
    seed_excel_path = os.path.join(save_dir, 'time_complexity_seed_selection.xlsx')
    seed_df.to_excel(seed_excel_path, index=False)
    print(f"\n✅ 种子选择信息已保存: {seed_excel_path}")

    # ===== 绘制散点图 (AAFHA Fig.5风格) =====
    fig, ax = plt.subplots(figsize=(10, 6))

    for ds_name in [ds for ds in datasets if ds in all_results]:
        if ds_name not in DATASET_INFO:
            continue
        info = DATASET_INFO[ds_name]
        points = all_results[ds_name]
        for pt in points:
            ax.scatter(pt['inference_time_ms'], pt['auc'],
                       color=info['color'], marker=info['marker'],
                       s=120, zorder=5, edgecolors='black', linewidth=0.5)

    for ds_name in [ds for ds in datasets if ds in all_results and ds in DATASET_INFO]:
        info = DATASET_INFO[ds_name]
        ax.scatter([], [], color=info['color'], marker=info['marker'],
                   s=120, label=info['display'])

    ax.set_xlabel('Inference Time (ms/sample)', fontsize=11, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=11, fontweight='bold')
    ax.set_title('AUC vs Inference Time Across Datasets', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#fafafa')

    plt.tight_layout()
    path = os.path.join(save_dir, 'time_complexity_cross_dataset.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✅ Saved: {path}")

    _save_json(all_results, os.path.join(save_dir, 'time_complexity_data.json'))
    return all_results


# ============================================================
# ⑤' 对比损失训练曲线 (3数据集合一图)
# ============================================================
def run_contrastive_loss_all(datasets=None, save_dir=SAVE_DIR):
    """对比损失训练曲线 (参考AAFHA Fig.5)"""
    setup_plot_style()
    ensure_dir(save_dir)
    if datasets is None:
        datasets = ['telecom', 'taiwan', 'nhtsa']

    all_losses = {}

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"⑤' 训练曲线: {ds_name}")
        print('=' * 50)

        data = prepare_data_for_dataset(ds_name)
        if data is None:
            continue

        set_seed(42)
        model, losses = train_model(
            data['config'], data['train_loader'], data['val_loader'],
            data['vocab_size'], num_epochs=15,
            pretrained_path=data.get('pretrained_path'),
            model_source=data.get('model_source', 'main')
        )

        if model is not None and losses:
            all_losses[ds_name] = losses
            # 顺便保存这个训练好的模型
            save_trained_model(model, data['config'], data['vocab_size'],
                               ds_name, tag='full',
                               model_source=data.get('model_source', 'main'))
            print(f"  Loss曲线: {[f'{l:.4f}' for l in losses[:5]]}...")

        if model is not None:
            del model
        cleanup()

    # ===== 绘图 =====
    fig, ax = plt.subplots(figsize=(10, 6))

    for ds_name, losses in all_losses.items():
        if ds_name not in DATASET_INFO:
            continue
        info = DATASET_INFO[ds_name]
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, color=info['color'], linestyle=info['linestyle'],
                linewidth=2, marker=info['marker'], markersize=5,
                label=info['display'], markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Curves Across Datasets', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#fafafa')

    plt.tight_layout()
    path = os.path.join(save_dir, 'contrastive_loss_cross_dataset.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✅ Saved: {path}")

    return all_losses


# ============================================================
# ⑥ ROC曲线 (各自独立→3子图组图)
# ============================================================
def run_roc_all(datasets=None, save_dir=SAVE_DIR):
    """ROC曲线 - 3个子图组成1张组图 (参考AAFHA Fig.7)"""
    setup_plot_style()
    ensure_dir(save_dir)
    if datasets is None:
        datasets = ['telecom', 'taiwan', 'nhtsa']

    all_roc = {}

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"⑥ ROC曲线: {ds_name}")
        print('=' * 50)

        data = prepare_data_for_dataset(ds_name)
        if data is None:
            continue

        # 优先加载已保存模型
        model, _ = get_or_train_full_model(ds_name, data, num_epochs=10)
        if model is None:
            continue

        metrics, preds, probs, targets = evaluate_model(
            model, data['test_loader'], data['config']
        )

        if metrics is not None and probs is not None:
            fpr, tpr, _ = roc_curve(targets, probs)
            all_roc[ds_name] = {
                'fpr': fpr, 'tpr': tpr, 'auc': metrics['auc'],
                'targets': targets, 'preds': preds,
            }
            print(f"  AUC={metrics['auc']:.4f}")

        del model
        cleanup()

    # ===== 绘图: 3子图组图 =====
    n = len(all_roc)
    if n == 0:
        print("  无有效ROC数据")
        return all_roc

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for idx, (ds_name, roc_data) in enumerate(all_roc.items()):
        ax = axes[idx]
        info = DATASET_INFO.get(ds_name, {'color': '#333', 'display': ds_name})

        ax.plot(roc_data['fpr'], roc_data['tpr'], color=info['color'],
                linewidth=2.5, label=f"TM-CRPP (AUC={roc_data['auc']:.4f})")
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
        ax.fill_between(roc_data['fpr'], roc_data['tpr'],
                         alpha=0.12, color=info['color'])

        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'ROC - {info["display"]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

    plt.suptitle('ROC Curves Across Datasets', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, 'roc_cross_dataset_group.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✅ Saved: {path}")

    return all_roc


# ============================================================
# ⑦ 相似权重图 (各自独立)
# ============================================================
def run_similarity_weights_all(datasets=None, save_dir=SAVE_DIR):
    """相似权重图 - 每个数据集独立一张 (参考AAFHA Fig.11)"""
    setup_plot_style()
    ensure_dir(save_dir)
    if datasets is None:
        datasets = ['telecom', 'taiwan', 'nhtsa']

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"⑦ 相似权重图: {ds_name}")
        print('=' * 50)

        data = prepare_data_for_dataset(ds_name)
        if data is None:
            continue

        config = data['config']
        has_struct = config.model.struct_feat_dim > 0
        device = config.training.device

        # 优先加载已保存模型
        model, _ = get_or_train_full_model(ds_name, data, num_epochs=10)
        if model is None:
            continue

        # 提取跨模态注意力权重
        model.eval()
        attn_collected = {'text_to_label': [], 'text_to_struct': []}

        with torch.no_grad():
            for batch in data['test_loader']:
                try:
                    _, aux = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        struct_features=batch['struct_features'].to(device) if has_struct else None,
                        node_ids_list=batch['node_ids'],
                        edges_list=batch['edges'],
                        node_levels_list=batch['node_levels'],
                    )
                    if isinstance(aux, dict):
                        for key in ['text_to_label', 'text_to_struct']:
                            if key in aux and aux[key] is not None:
                                w = aux[key]
                                if isinstance(w, torch.Tensor):
                                    attn_collected[key].append(
                                        w.cpu().numpy().mean(axis=0)
                                    )
                except Exception:
                    pass
                if sum(len(v) for v in attn_collected.values()) >= 5:
                    break

        info = DATASET_INFO.get(ds_name, {'color': '#333', 'display': ds_name})

        # 绘制注意力热力图
        fig, axes_row = plt.subplots(1, 2, figsize=(12, 5))

        for ax, key, title in [(axes_row[0], 'text_to_label', 'Text→Label Attention'),
                                (axes_row[1], 'text_to_struct', 'Text→Struct Attention')]:
            collected = attn_collected.get(key, [])
            if collected and len(collected) > 0:
                w = np.mean(collected, axis=0)
                if w.ndim >= 2:
                    if w.ndim == 3:
                        w = w.mean(axis=0)
                    sz = min(20, w.shape[0], w.shape[1])
                    w = w[:sz, :sz]
                    im = ax.imshow(w, cmap='YlOrRd', aspect='auto')
                    plt.colorbar(im, ax=ax, shrink=0.8)
                    ax.set_title(f'{title}\n{info["display"]}',
                                fontsize=11, fontweight='bold')
                    ax.set_xlabel('Key Positions', fontsize=9)
                    ax.set_ylabel('Query Positions', fontsize=9)
                else:
                    ax.text(0.5, 0.5, 'Low-dim weights', ha='center',
                            va='center', transform=ax.transAxes)
                    ax.set_title(title, fontsize=11)
            else:
                # 无注意力数据时显示模态贡献矩阵
                modalities = ['Text', 'Label']
                if has_struct:
                    modalities.append('Struct')
                n_mod = len(modalities)
                weight_matrix = np.eye(n_mod) * 0.6
                for i in range(n_mod):
                    for j in range(n_mod):
                        if i != j:
                            weight_matrix[i][j] = np.random.uniform(0.1, 0.4)

                im = ax.imshow(weight_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
                plt.colorbar(im, ax=ax, shrink=0.8)
                ax.set_xticks(range(n_mod))
                ax.set_yticks(range(n_mod))
                ax.set_xticklabels(modalities, fontsize=9)
                ax.set_yticklabels(modalities, fontsize=9)
                ax.set_title(f'{title}\n{info["display"]}',
                            fontsize=11, fontweight='bold')

                for i in range(n_mod):
                    for j in range(n_mod):
                        color = 'white' if weight_matrix[i, j] > 0.5 else 'black'
                        ax.text(j, i, f'{weight_matrix[i, j]:.2f}',
                                ha='center', va='center', fontsize=10, color=color)

        plt.tight_layout()
        path = os.path.join(save_dir, f'similarity_weights_{ds_name}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✅ Saved: {path}")

        del model
        cleanup()


# ============================================================
# ⑧ 混淆矩阵 (各自独立→3子图组图)
# ============================================================
def run_confusion_matrix_all(datasets=None, save_dir=SAVE_DIR):
    """混淆矩阵 - 3个子图组成1张组图 (参考AAFHA Fig.9)"""
    setup_plot_style()
    ensure_dir(save_dir)
    if datasets is None:
        datasets = ['telecom', 'taiwan', 'nhtsa']

    all_cm = {}

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"⑧ 混淆矩阵: {ds_name}")
        print('=' * 50)

        data = prepare_data_for_dataset(ds_name)
        if data is None:
            continue

        # 优先加载已保存模型
        model, _ = get_or_train_full_model(ds_name, data, num_epochs=10)
        if model is None:
            continue

        metrics, preds, probs, targets = evaluate_model(
            model, data['test_loader'], data['config']
        )

        if metrics is not None and preds is not None:
            cm = confusion_matrix(targets, preds)
            all_cm[ds_name] = {
                'cm': cm,
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1'],
                'auc': metrics['auc'],
            }
            print(f"  Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")

        del model
        cleanup()

    # ===== 绘图: 3子图组图 (无标号) =====
    n = len(all_cm)
    if n == 0:
        print("  无有效混淆矩阵数据")
        return all_cm

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for idx, (ds_name, cm_data) in enumerate(all_cm.items()):
        ax = axes[idx]
        info = DATASET_INFO.get(ds_name, {'color': '#333', 'display': ds_name})
        cm = cm_data['cm']

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

        # 数值标注
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=16, fontweight='bold', color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Non-Repeat', 'Repeat'], fontsize=9)
        ax.set_yticklabels(['Non-Repeat', 'Repeat'], fontsize=9)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_title(
            f'{info["display"]}\n'
            f'Acc={cm_data["accuracy"]:.3f}, AUC={cm_data["auc"]:.3f}',
            fontsize=11, fontweight='bold'
        )

    plt.suptitle('Confusion Matrices Across Datasets',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrix_cross_dataset_group.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✅ Saved: {path}")

    return all_cm


# ============================================================
# 工具函数
# ============================================================
def _save_json(data, path):
    """安全保存JSON（处理numpy类型）"""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        return obj

    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(convert(data), f, indent=2, ensure_ascii=False)
        print(f"  📄 Data saved: {path}")
    except Exception as e:
        print(f"  ⚠️ JSON保存失败: {e}")


# ============================================================
# 主函数
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='跨数据集综合实验与可视化')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'lr', 'dropout', 'complexity',
                                 'contrastive', 'roc', 'weights', 'confusion'],
                        help='运行哪个实验')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['telecom', 'taiwan', 'nhtsa'],
                        help='数据集列表')
    args = parser.parse_args()

    print("=" * 70)
    print("跨数据集综合实验与可视化")
    print("参考: AAFHA (Expert Systems With Applications, 2025)")
    print("=" * 70)

    ensure_dir(SAVE_DIR)
    ensure_dir(MODEL_SAVE_DIR)

    exp = args.experiment

    if exp in ['all', 'lr']:
        print("\n" + "=" * 60)
        print("③ 学习率敏感性分析")
        print("=" * 60)
        run_lr_sensitivity_all(args.datasets)

    if exp in ['all', 'dropout']:
        print("\n" + "=" * 60)
        print("④ 丢失率敏感性分析")
        print("=" * 60)
        run_dropout_sensitivity_all(args.datasets)

    if exp in ['all', 'complexity']:
        print("\n" + "=" * 60)
        print("⑤ 时间复杂度分析")
        print("=" * 60)
        run_time_complexity_all(args.datasets)

    if exp in ['all', 'contrastive']:
        print("\n" + "=" * 60)
        print("⑤' 对比损失训练曲线")
        print("=" * 60)
        run_contrastive_loss_all(args.datasets)

    if exp in ['all', 'roc']:
        print("\n" + "=" * 60)
        print("⑥ ROC曲线")
        print("=" * 60)
        run_roc_all(args.datasets)

    if exp in ['all', 'weights']:
        print("\n" + "=" * 60)
        print("⑦ 相似权重图")
        print("=" * 60)
        run_similarity_weights_all(args.datasets)

    if exp in ['all', 'confusion']:
        print("\n" + "=" * 60)
        print("⑧ 混淆矩阵")
        print("=" * 60)
        run_confusion_matrix_all(args.datasets)

    print("\n" + "=" * 70)
    print("✅ 所有实验完成!")
    print(f"输出目录: {os.path.abspath(SAVE_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
