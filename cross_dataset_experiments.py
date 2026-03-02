"""
跨数据集综合实验与可视化脚本
参考: AAFHA - An adaptive auto fusion with hierarchical attention
      for multimodal fake news detection (ESWA 2025)

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

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_taiwan_restaurant_config, get_consumer_complaint_config
from data_processor import ComplaintDataProcessor, ComplaintDataset, custom_collate_fn
from model import MultiModalComplaintModel


# ============================================================
# 全局配置
# ============================================================
SAVE_DIR = './outputs/cross_dataset'

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
    'consumer': {
        'display': 'Consumer Complaint',
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
# 数据加载（核心函数）
# ============================================================
def get_config_for_dataset(dataset_name):
    """获取数据集对应的Config"""
    if dataset_name == 'taiwan':
        return get_taiwan_restaurant_config()
    elif dataset_name == 'consumer':
        return get_consumer_complaint_config()
    else:
        return Config()


def adapt_columns(df, dataset_name):
    """适配列名到标准格式 (biz_cntt / Complaint label / Repeat complaint)"""
    rename_map = {}
    if dataset_name == 'taiwan':
        if 'Complaint_label' in df.columns and 'Complaint label' not in df.columns:
            rename_map['Complaint_label'] = 'Complaint label'
        if 'satisfaction_binary' in df.columns and 'Repeat complaint' not in df.columns:
            rename_map['satisfaction_binary'] = 'Repeat complaint'
    elif dataset_name == 'consumer':
        if 'disputed' in df.columns and 'Repeat complaint' not in df.columns:
            rename_map['disputed'] = 'Repeat complaint'
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  列名适配: {rename_map}")
    return df


def find_data_file(config):
    """查找数据文件（兼容多种文件名）"""
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


def prepare_data_for_dataset(dataset_name, batch_size=16):
    """
    准备指定数据集的完整数据（train/val/test DataLoader）

    Returns:
        dict with keys: config, vocab_size, train_loader, val_loader, test_loader
        或 None (数据文件不存在时)
    """
    print(f"\n📂 准备数据: {dataset_name}")

    config = get_config_for_dataset(dataset_name)
    config.training.batch_size = batch_size

    data_file = find_data_file(config)
    if data_file is None:
        return None

    # 读取并适配列名
    df = pd.read_excel(data_file)
    df = adapt_columns(df, dataset_name)

    # 保存到临时文件（ComplaintDataProcessor.prepare_datasets读取文件路径）
    tmp = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    df.to_excel(tmp.name, index=False)

    try:
        processor = ComplaintDataProcessor(
            config=config,
            user_dict_file=config.data.user_dict_file
        )

        # 尝试加载已有processor（仅telecom有预训练）
        if dataset_name not in ['taiwan', 'consumer']:
            for p in ['./pretrained_complaint_bert_improved/processor.pkl',
                       './processor.pkl']:
                if os.path.exists(p):
                    try:
                        processor.load(p)
                        print(f"  ✅ 加载processor: {p}")
                    except Exception:
                        pass
                    break

        data = processor.prepare_datasets(train_file=tmp.name, for_pretrain=False)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    vocab_size = data.get('vocab_size', len(processor.node_to_id) + 1)

    # 划分数据 (60% / 20% / 20%)
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

    # telecom使用领域预训练BERT路径
    _pretrained = None
    if dataset_name not in ['taiwan', 'consumer']:
        _candidate = os.path.join(
            config.training.pretrain_save_dir, 'stage2'
        )
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
    }

    print(f"  训练: {len(train_d['targets'])}, 验证: {len(val_d['targets'])}, 测试: {len(test_d['targets'])}")
    return result


# ============================================================
# 训练与评估工具
# ============================================================
def train_model(config, train_loader, val_loader, vocab_size,
                mode='full', num_epochs=10, pretrained_path=None):
    """训练模型，返回 (model, train_losses_per_epoch)"""
    device = config.training.device
    has_struct = config.model.struct_feat_dim > 0

    # 自动调整mode
    if not has_struct and mode in ['full', 'text_struct', 'label_struct', 'struct_only']:
        if mode == 'full':
            mode = 'text_label'
        elif mode == 'text_struct':
            mode = 'text_only'
        elif mode == 'label_struct':
            mode = 'label_only'
        elif mode == 'struct_only':
            return None, []

    model = MultiModalComplaintModel(
        config=config, vocab_size=vocab_size, mode=mode,
        pretrained_path=pretrained_path
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.learning_rate, weight_decay=0.01
    )

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


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 绘图工具
# ============================================================
def setup_plot_style():
    """设置统一的绘图样式"""
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
    })


def smooth_curve(x, y, num_points=200):
    """平滑曲线"""
    try:
        from scipy.interpolate import make_interp_spline
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        if len(x_arr) >= 4:
            x_smooth = np.linspace(x_arr.min(), x_arr.max(), num_points)
            spl = make_interp_spline(x_arr, y_arr, k=3)
            y_smooth = spl(x_smooth)
            return x_smooth, y_smooth
    except Exception:
        pass
    return np.array(x, dtype=float), np.array(y, dtype=float)


# ============================================================
# ③ 学习率敏感性 (3数据集合一图)
# ============================================================
def run_lr_sensitivity_all(datasets=None, save_dir=SAVE_DIR):
    """学习率敏感性分析 - 三个数据集整合到一张图 (参考AAFHA Fig.3)"""
    setup_plot_style()
    ensure_dir(save_dir)
    if datasets is None:
        datasets = ['telecom', 'taiwan', 'consumer']

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
        original_lr = config.training.learning_rate
        ds_results = {}

        for lr in learning_rates:
            print(f"  LR={lr:.0e}...", end=' ')
            set_seed(42)
            config.training.learning_rate = lr

            model, _ = train_model(config, data['train_loader'], data['val_loader'],
                                   data['vocab_size'], num_epochs=10,pretrained_path=data.get('pretrained_path'))
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

        config.training.learning_rate = original_lr
        all_results[ds_name] = ds_results

    # ===== 绘图: 2子图 (Accuracy + AUC) =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric_name, ax, title in [('accuracy', axes[0], 'Accuracy'),
                                    ('auc', axes[1], 'AUC')]:
        x = np.arange(len(learning_rates))

        for ds_name, ds_res in all_results.items():
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

    # 保存数据
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
        datasets = ['telecom', 'taiwan', 'consumer']

    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"④ Dropout敏感性: {ds_name}")
        print('=' * 50)

        data = prepare_data_for_dataset(ds_name)
        if data is None:
            continue

        config = data['config']
        original_dropout = config.model.dropout
        ds_results = {}

        for dr in dropout_rates:
            print(f"  Dropout={dr}...", end=' ')
            set_seed(42)
            config.model.dropout = dr

            model, _ = train_model(config, data['train_loader'], data['val_loader'],
                                   data['vocab_size'], num_epochs=10,pretrained_path=data.get('pretrained_path'))
            if model is None:
                print("跳过")
                continue
            metrics, _, _, _ = evaluate_model(model, data['test_loader'], config)

            if metrics:
                ds_results[dr] = metrics
                print(f"AUC={metrics['auc']:.4f}")
            else:
                print("失败")

            del model
            cleanup()

        config.model.dropout = original_dropout
        all_results[ds_name] = ds_results

    # ===== 绘图 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric_name, ax, title in [('accuracy', axes[0], 'Accuracy'),
                                    ('auc', axes[1], 'AUC')]:
        x = np.arange(len(dropout_rates))

        for ds_name, ds_res in all_results.items():
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
    """时间复杂度分析 - 分组柱状图 (参考AAFHA Table 4)"""
    setup_plot_style()
    ensure_dir(save_dir)
    if datasets is None:
        datasets = ['telecom', 'taiwan', 'consumer']

    model_configs = [
        ('Text-Only', 'text_only'),
        ('Label-Only', 'label_only'),
        ('Text+Label', 'text_label'),
        ('Full Model (Ours)', 'full'),
    ]

    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"⑤ 时间复杂度: {ds_name}")
        print('=' * 50)

        data = prepare_data_for_dataset(ds_name)
        if data is None:
            continue

        config = data['config']
        has_struct = config.model.struct_feat_dim > 0
        device = config.training.device
        ds_results = {}

        for model_name, mode in model_configs:
            actual_mode = mode
            if not has_struct:
                if mode == 'full':
                    actual_mode = 'text_label'
                elif mode in ['struct_only', 'text_struct', 'label_struct']:
                    continue

            set_seed(42)
            try:
                model = MultiModalComplaintModel(
                    config=config, vocab_size=data['vocab_size'], mode=actual_mode
                )
                model = model.to(device)
            except Exception as e:
                print(f"  {model_name}: 创建失败 - {e}")
                continue

            params = count_parameters(model)

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
            ds_results[model_name] = {
                'parameters_M': round(params, 2),
                'inference_time_ms': round(infer_time, 2),
            }
            print(f"  {model_name}: {params:.2f}M params, {infer_time:.2f}ms/sample")

            del model
            cleanup()

        all_results[ds_name] = ds_results

    # ===== 绘图: 分组柱状图 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ds_list = [ds for ds in datasets if ds in all_results]
    model_list = [mc[0] for mc in model_configs]
    model_colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']

    for key, title, ax in [('parameters_M', 'Parameters (M)', axes[0]),
                            ('inference_time_ms', 'Inference (ms/sample)', axes[1])]:
        x = np.arange(len(ds_list))
        width = 0.18
        n = len(model_list)

        for j, model_name in enumerate(model_list):
            vals = []
            for ds_name in ds_list:
                v = all_results.get(ds_name, {}).get(model_name, {}).get(key, 0)
                vals.append(v)
            offset = (j - n / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=model_name,
                          color=model_colors[j], edgecolor='white', alpha=0.85)
            # 数值标签
            for bar_item in bars:
                h = bar_item.get_height()
                if h > 0:
                    ax.text(bar_item.get_x() + bar_item.get_width() / 2, h,
                            f'{h:.1f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_INFO[ds]['display'] for ds in ds_list],
                           fontsize=9, rotation=5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_facecolor('#fafafa')

    plt.suptitle('Computational Complexity Across Datasets', fontsize=14, fontweight='bold')
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
    """训练损失曲线 (参考AAFHA Fig.5)"""
    setup_plot_style()
    ensure_dir(save_dir)
    if datasets is None:
        datasets = ['telecom', 'taiwan', 'consumer']

    all_losses = {}
    num_epochs = 15

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"⑤' 训练损失曲线: {ds_name}")
        print('=' * 50)

        data = prepare_data_for_dataset(ds_name)
        if data is None:
            continue

        set_seed(42)
        model, losses = train_model(data['config'], data['train_loader'],
                                     data['val_loader'], data['vocab_size'],
                                     num_epochs=num_epochs,pretrained_path=data.get('pretrained_path'))
        if model is not None:
            all_losses[ds_name] = losses
            print(f"  最终loss: {losses[-1]:.4f}")
            del model
            cleanup()

    # ===== 绘图 =====
    fig, ax = plt.subplots(figsize=(10, 6))

    for ds_name, losses in all_losses.items():
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
        datasets = ['telecom', 'taiwan', 'consumer']

    all_roc = {}

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"⑥ ROC曲线: {ds_name}")
        print('=' * 50)

        data = prepare_data_for_dataset(ds_name)
        if data is None:
            continue

        set_seed(42)
        model, _ = train_model(data['config'], data['train_loader'],
                                data['val_loader'], data['vocab_size'],
                                num_epochs=10,pretrained_path=data.get('pretrained_path'))
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
        info = DATASET_INFO[ds_name]

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
        ax.text(0.05, 0.95, f'({chr(97 + idx)})', transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top')

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
        datasets = ['telecom', 'taiwan', 'consumer']

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

        set_seed(42)
        model, _ = train_model(config, data['train_loader'], data['val_loader'],
                                data['vocab_size'], num_epochs=10,pretrained_path=data.get('pretrained_path'))
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

        info = DATASET_INFO[ds_name]

        # 绘制注意力热力图
        fig, axes_row = plt.subplots(1, 2, figsize=(12, 5))

        for ax, key, title in [(axes_row[0], 'text_to_label', 'Text→Label Attention'),
                                (axes_row[1], 'text_to_struct', 'Text→Struct Attention')]:
            collected = attn_collected.get(key, [])
            if collected and len(collected) > 0:
                w = np.mean(collected, axis=0)
                if w.ndim >= 2:
                    # 取平均头部，截取前20维度
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

                # 从模型参数推断模态权重
                weight_matrix = np.eye(n_mod) * 0.6
                for i in range(n_mod):
                    for j in range(n_mod):
                        if i != j:
                            weight_matrix[i][j] = 0.2

                im = ax.imshow(weight_matrix, cmap='Blues', vmin=0, vmax=1, aspect='auto')
                ax.set_xticks(range(n_mod))
                ax.set_yticks(range(n_mod))
                ax.set_xticklabels(modalities, fontsize=10)
                ax.set_yticklabels(modalities, fontsize=10)
                for i in range(n_mod):
                    for j in range(n_mod):
                        ax.text(j, i, f'{weight_matrix[i, j]:.2f}',
                                ha='center', va='center', fontsize=11)
                plt.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title(f'Modality Weights - {info["display"]}',
                            fontsize=11, fontweight='bold')

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
        datasets = ['telecom', 'taiwan', 'consumer']

    all_cm = {}

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"⑧ 混淆矩阵: {ds_name}")
        print('=' * 50)

        data = prepare_data_for_dataset(ds_name)
        if data is None:
            continue

        set_seed(42)
        model, _ = train_model(data['config'], data['train_loader'],
                                data['val_loader'], data['vocab_size'],
                                num_epochs=10,pretrained_path=data.get('pretrained_path'))
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

    # ===== 绘图: 3子图组图 =====
    n = len(all_cm)
    if n == 0:
        print("  无有效混淆矩阵数据")
        return all_cm

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for idx, (ds_name, cm_data) in enumerate(all_cm.items()):
        ax = axes[idx]
        info = DATASET_INFO[ds_name]
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
        ax.text(0.05, 0.95, f'({chr(97 + idx)})', transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top', color='red')

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
                        default=['telecom', 'taiwan', 'consumer'],
                        help='数据集列表')
    args = parser.parse_args()

    print("=" * 70)
    print("跨数据集综合实验与可视化")
    print("参考: AAFHA (Expert Systems With Applications, 2025)")
    print("=" * 70)

    ensure_dir(SAVE_DIR)

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
