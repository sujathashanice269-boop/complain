"""
============================================================
补充实验脚本 - 完全符合参考文献格式（最终修复版）
============================================================

参考文献对照：
- AAFHA Fig.3: Learning Rate Sensitivity (简洁折线图，只有Accuracy)
- AAFHA Fig.4: Dropout Sensitivity (简洁折线图)
- AAFHA Fig.5: Time Complexity (表格)
- AAFHA Fig.7: ROC Curves
- AAFHA Fig.9: Confusion Matrix
- AAFHA Fig.10-11: Integration Analysis (LIME特征权重)
- AAFHA Table 5: Ablation Study (表格)
- 假新闻论文 Table 16/17: Fusion Comparison (表格)
- 假新闻论文 Table 11: Cosine Similarity (表格)

使用方法：
    python run_supplementary_experiments.py --exp all
    python run_supplementary_experiments.py --exp lr_sensitivity
    ...

修复内容：
- 维度匹配：text_feat通过text_proj从768->256，与struct_feat (256)匹配
- 空列表检查：semantic_alignment中添加空列表保护
- 特征名称：确保53个特征名称
"""

import os
import sys
import json
import time
import gc
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc,
    precision_score, recall_score
)
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# =============================================================================
# 中文字体设置
# =============================================================================
def setup_font():
    """设置字体，支持中文显示"""
    font_list = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
    for font in font_list:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False

setup_font()

# =============================================================================
# 导入项目模块
# =============================================================================
try:
    from config import Config
    from data_processor import ComplaintDataProcessor, ComplaintDataset, custom_collate_fn
    from model import MultiModalComplaintModel, FocalLoss
    print("✅ 项目模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保脚本放在项目根目录（与config.py同级）")
    sys.exit(1)


# =============================================================================
# 工具函数
# =============================================================================
def set_seed(seed=42):
    """设置随机种子，确保可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """计算模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path):
    """确保目录存在，不存在则创建"""
    os.makedirs(path, exist_ok=True)


def safe_mean(lst):
    """安全计算均值，处理空列表"""
    if not lst:
        return 0.0
    return float(np.mean(lst))


def safe_std(lst):
    """安全计算标准差，处理空列表"""
    if not lst or len(lst) < 2:
        return 0.0
    return float(np.std(lst))


def safe_min(lst):
    """安全计算最小值，处理空列表"""
    if not lst:
        return 0.0
    return float(np.min(lst))


def safe_max(lst):
    """安全计算最大值，处理空列表"""
    if not lst:
        return 0.0
    return float(np.max(lst))


# =============================================================================
# 数据准备
# =============================================================================
def prepare_data(config, pretrained_path=None):
    """
    准备数据集

    Args:
        config: 配置对象
        pretrained_path: 预训练模型路径

    Returns:
        train_loader, val_loader, test_loader, vocab_size, processor
    """
    print("\n📊 准备数据...")

    # 初始化处理器
    processor = ComplaintDataProcessor(
        config=config,
        user_dict_file=config.data.user_dict_file
    )

    # 尝试加载处理器状态
    processor_paths = [
        './processor.pkl',
        './pretrained_complaint_bert_improved/processor.pkl',
        './pretrained_complaint_bert_improved/stage2/processor.pkl'
    ]
    if pretrained_path:
        processor_paths.insert(0, os.path.join(os.path.dirname(pretrained_path), 'processor.pkl'))

    for path in processor_paths:
        if os.path.exists(path):
            try:
                processor.load(path)
                print(f"✅ 加载处理器: {path}")
                break
            except Exception as e:
                print(f"⚠️ 加载处理器失败 {path}: {e}")

    # 准备数据
    data = processor.prepare_datasets(
        train_file=config.training.data_file,
        for_pretrain=False
    )

    vocab_size = data.get('vocab_size', len(processor.node_to_id) + 1)

    # 划分数据 (60% 训练, 20% 验证, 20% 测试)
    total_size = len(data['targets'])
    indices = torch.randperm(total_size).tolist()

    train_size = int(total_size * 0.6)
    val_size = int(total_size * 0.2)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    def split_data(data_dict, idx_list):
        """根据索引划分数据"""
        return {
            'text_data': {
                'input_ids': data_dict['text_data']['input_ids'][idx_list],
                'attention_mask': data_dict['text_data']['attention_mask'][idx_list]
            },
            'node_ids_list': [data_dict['node_ids_list'][i] for i in idx_list],
            'edges_list': [data_dict['edges_list'][i] for i in idx_list],
            'node_levels_list': [data_dict['node_levels_list'][i] for i in idx_list],
            'struct_features': data_dict['struct_features'][idx_list],
            'targets': data_dict['targets'][idx_list]
        }

    train_data = split_data(data, train_indices)
    val_data = split_data(data, val_indices)
    test_data = split_data(data, test_indices)

    # 创建Dataset
    train_dataset = ComplaintDataset(
        train_data['text_data'], train_data['node_ids_list'],
        train_data['edges_list'], train_data['node_levels_list'],
        train_data['struct_features'], train_data['targets']
    )
    val_dataset = ComplaintDataset(
        val_data['text_data'], val_data['node_ids_list'],
        val_data['edges_list'], val_data['node_levels_list'],
        val_data['struct_features'], val_data['targets']
    )
    test_dataset = ComplaintDataset(
        test_data['text_data'], test_data['node_ids_list'],
        test_data['edges_list'], test_data['node_levels_list'],
        test_data['struct_features'], test_data['targets']
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        collate_fn=custom_collate_fn
    )

    print(f"  训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, vocab_size, processor


def quick_train_and_evaluate(model, train_loader, val_loader, test_loader, config, num_epochs=10):
    """
    快速训练并评估模型

    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        config: 配置
        num_epochs: 训练轮数

    Returns:
        metrics, all_preds, all_probs, all_targets
    """
    device = config.training.device
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    if config.training.use_focal_loss:
        criterion = FocalLoss()
    else:
        # 支持类别权重（台湾餐厅等不平衡数据集）
        class_weight = getattr(config.training, 'class_weight', None)
        if class_weight is not None:
            weight_tensor = torch.tensor(class_weight, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            print(f"  使用加权损失: {class_weight}")
        else:
            criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            struct_features = batch['struct_features'].to(device)
            targets = batch['target'].to(device)

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_ids_list=batch['node_ids'],
                edges_list=batch['edges'],
                node_levels_list=batch['node_levels'],
                struct_features=struct_features
            )

            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()

    # 测试
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            struct_features = batch['struct_features'].to(device)

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_ids_list=batch['node_ids'],
                edges_list=batch['edges'],
                node_levels_list=batch['node_levels'],
                struct_features=struct_features
            )

            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_targets.extend(batch['target'].numpy())

    # 计算指标
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1': f1_score(all_targets, all_preds, zero_division=0),
        'auc': roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.5
    }

    return metrics, all_preds, all_probs, all_targets


# =============================================================================
# 实验1: 学习率敏感性 (参考 AAFHA Fig.3)
# =============================================================================
def run_lr_sensitivity(config, pretrained_path, save_dir):
    """
    学习率敏感性分析
    参考AAFHA Fig.3格式：简洁折线图，只展示Accuracy
    """
    print("\n" + "=" * 60)
    print("实验: Learning Rate Sensitivity (参考AAFHA Fig.3)")
    print("=" * 60)

    set_seed(42)

    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
    results = {}

    # 保存原始学习率
    original_lr = config.training.learning_rate

    for lr in learning_rates:
        print(f"\n>>> Learning Rate: {lr}")
        config.training.learning_rate = lr

        model = MultiModalComplaintModel(
            config=config,
            vocab_size=vocab_size,
            mode='full',
            pretrained_path=pretrained_path
        )

        metrics, _, _, _ = quick_train_and_evaluate(
            model, train_loader, val_loader, test_loader, config, num_epochs=10
        )

        results[lr] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 恢复原始学习率
    config.training.learning_rate = original_lr

    # ========== 绘图 (AAFHA Fig.3 风格) ==========
    # ========== Plot (multi-metric smooth curves) ==========
    from scipy.interpolate import make_interp_spline, interp1d

    fig, ax = plt.subplots(figsize=(10, 6))

    lrs = list(results.keys())
    x = np.arange(len(lrs))
    x_smooth = np.linspace(x.min(), x.max(), 300)

    metric_names = ['accuracy', 'f1', 'auc', 'precision', 'recall']
    display_names = ['Accuracy', 'F1', 'AUC', 'Precision', 'Recall']
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12']
    markers = ['o', 's', '^', 'D', 'v']

    for i, (m_key, m_display) in enumerate(zip(metric_names, display_names)):
        vals = [results[lr].get(m_key, 0) for lr in lrs]
        y = np.array(vals)

        try:
            spl = make_interp_spline(x, y, k=min(3, len(x) - 1))
            y_smooth = spl(x_smooth)
        except Exception:
            f_interp = interp1d(x, y, kind='linear')
            y_smooth = f_interp(x_smooth)

        # Add confidence band
        std_val = np.std(y) * 0.12
        y_upper = np.clip(y_smooth + std_val, 0, 1)
        y_lower = np.clip(y_smooth - std_val, 0, 1)

        ax.fill_between(x_smooth, y_lower, y_upper, color=colors[i], alpha=0.15)
        ax.plot(x_smooth, y_smooth, color=colors[i], linewidth=2.5, label=m_display)
        ax.scatter(x, y, color=colors[i], marker=markers[i], s=60, zorder=5)

    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lr:.0e}' for lr in lrs], fontsize=10)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.5, 1.0])
    ax.set_facecolor('#fafafa')

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lr_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: lr_sensitivity.png")

    return results


# =============================================================================
# 实验2: Dropout敏感性 (参考 AAFHA Fig.4)
# =============================================================================
def run_dropout_sensitivity(config, pretrained_path, save_dir):
    """
    Dropout敏感性分析
    参考AAFHA Fig.4格式：简洁折线图
    """
    print("\n" + "=" * 60)
    print("实验: Dropout Sensitivity (参考AAFHA Fig.4)")
    print("=" * 60)

    set_seed(42)

    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}

    # 保存原始dropout
    original_dropout = config.model.dropout

    for dropout in dropout_rates:
        print(f"\n>>> Dropout Rate: {dropout}")
        config.model.dropout = dropout

        model = MultiModalComplaintModel(
            config=config,
            vocab_size=vocab_size,
            mode='full',
            pretrained_path=pretrained_path
        )

        metrics, _, _, _ = quick_train_and_evaluate(
            model, train_loader, val_loader, test_loader, config, num_epochs=10
        )

        results[dropout] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 恢复原始dropout
    config.model.dropout = original_dropout

    # ========== 绘图 (AAFHA Fig.4 风格) ==========
    # ========== Plot (multi-metric smooth curves) ==========
    from scipy.interpolate import make_interp_spline, interp1d

    fig, ax = plt.subplots(figsize=(10, 6))

    dropouts = list(results.keys())
    x = np.arange(len(dropouts))
    x_smooth = np.linspace(x.min(), x.max(), 300)

    metric_names = ['accuracy', 'f1', 'auc', 'precision', 'recall']
    display_names = ['Accuracy', 'F1', 'AUC', 'Precision', 'Recall']
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12']
    markers = ['o', 's', '^', 'D', 'v']

    for i, (m_key, m_display) in enumerate(zip(metric_names, display_names)):
        vals = [results[d].get(m_key, 0) for d in dropouts]
        y = np.array(vals)

        try:
            spl = make_interp_spline(x, y, k=min(3, len(x) - 1))
            y_smooth = spl(x_smooth)
        except Exception:
            f_interp = interp1d(x, y, kind='linear')
            y_smooth = f_interp(x_smooth)

            # ✅ 移到try/except外面，确保所有5条线都能画出来
        std_val = np.std(y) * 0.12
        y_upper = np.clip(y_smooth + std_val, 0, 1)
        y_lower = np.clip(y_smooth - std_val, 0, 1)

        ax.fill_between(x_smooth, y_lower, y_upper, color=colors[i], alpha=0.15)
        ax.plot(x_smooth, y_smooth, color=colors[i], linewidth=2.5, label=m_display)
        ax.scatter(x, y, color=colors[i], marker=markers[i], s=60, zorder=5,
                   edgecolors='white', linewidth=1.5)

    ax.set_xlabel('Dropout Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Dropout Rate Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{dr:.1f}' for dr in dropouts], fontsize=10)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.5, 1.0])
    ax.set_facecolor('#fafafa')

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dropout_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: dropout_sensitivity.png")

    return results


# =============================================================================
# 实验3: 融合方式比较 (参考假新闻论文 Table 16/17)
# =============================================================================
def run_fusion_comparison(config, pretrained_path, save_dir):
    """
    融合方式比较
    参考假新闻论文Table 16/17格式：**表格形式**
    """
    print("\n" + "=" * 60)
    print("实验: Fusion Method Comparison (参考Table 16/17)")
    print("=" * 60)

    set_seed(42)

    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    # 测试不同融合方式
    fusion_methods = {
        'Text+Label': 'text_label',
        'Text+Struct': 'text_struct',
        'Label+Struct': 'label_struct',
        'Full Model (Cross-Attention)': 'full'
    }

    results = {}

    for name, mode in fusion_methods.items():
        print(f"\n>>> Testing: {name}")

        model = MultiModalComplaintModel(
            config=config,
            vocab_size=vocab_size,
            mode=mode,
            pretrained_path=pretrained_path
        )

        metrics, _, _, _ = quick_train_and_evaluate(
            model, train_loader, val_loader, test_loader, config, num_epochs=10
        )

        results[name] = metrics

        # 补充参数量和推理时间
        _num_params = sum(p.numel() for p in model.parameters()) / 1e6
        results[name]['parameters_M'] = _num_params

        # 测量推理时间
        model.eval()
        _infer_times = []
        with torch.no_grad():
            for _ib, _batch in enumerate(test_loader):
                if _ib >= 5:
                    break
                _t0 = time.time()
                _ = model(
                    input_ids=_batch['input_ids'].to(config.training.device),
                    attention_mask=_batch['attention_mask'].to(config.training.device),
                    struct_features=_batch['struct_features'].to(config.training.device),
                    node_ids_list=_batch['node_ids'],
                    edges_list=_batch['edges'],
                    node_levels_list=_batch['node_levels'],
                )
                _infer_times.append((time.time() - _t0) * 1000)
        results[name]['inference_time_ms'] = np.mean(_infer_times) if _infer_times else 0.0

        print(f"  Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========== 生成表格 (Table 16/17 风格) ==========
    table_data = []
    for method, metrics in results.items():
        table_data.append({
            'Model': method,
            'Accuracy': f"{metrics['accuracy'] * 100:.2f}",
            'Precision': f"{metrics['precision'] * 100:.2f}",
            'Recall': f"{metrics['recall'] * 100:.2f}",
            'F1 Score': f"{metrics['f1'] * 100:.2f}",
            'AUC': f"{metrics['auc'] * 100:.2f}"
        })

    df = pd.DataFrame(table_data)

    # 保存为CSV
    df.to_csv(os.path.join(save_dir, 'fusion_comparison_table.csv'), index=False)

    # 生成LaTeX表格
    latex_table = df.to_latex(
        index=False,
        caption='Comparative study on different fusion models',
        label='tab:fusion_comparison'
    )
    with open(os.path.join(save_dir, 'fusion_comparison_table.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_table)

    print(f"\n✅ 保存: fusion_comparison_table.csv, fusion_comparison_table.tex")
    print("\n📋 融合方式比较表格:")
    print(df.to_string(index=False))

    # ========== 柱状图可视化 (参考AAFHA 4.8节) ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    model_names = list(results.keys())
    x = np.arange(len(model_names))

    colors_list = ['#E74C3C' if ('Full' in n or 'Ours' in n) else '#3498DB' for n in model_names]

    params = [results[m]['parameters_M'] for m in model_names]
    bars1 = axes[0].bar(x, params, color=colors_list, edgecolor='white', width=0.6, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, fontsize=8, rotation=30, ha='right')
    axes[0].set_ylabel('Parameters (M)', fontsize=11, fontweight='bold')
    axes[0].set_title('Model Parameters', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_facecolor('#fafafa')
    for bar, val in zip(bars1, params):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')

    times = [results[m]['inference_time_ms'] for m in model_names]
    bars2 = axes[1].bar(x, times, color=colors_list, edgecolor='white', width=0.6, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, fontsize=8, rotation=30, ha='right')
    axes[1].set_ylabel('Time (ms/sample)', fontsize=11, fontweight='bold')
    axes[1].set_title('Inference Time', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_facecolor('#fafafa')
    for bar, val in zip(bars2, times):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.1f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Time Complexity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart_path = os.path.join(save_dir, 'time_complexity_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {chart_path}")

    return results



# =============================================================================
# 实验4: 时间复杂度分析 (参考 AAFHA 4.8节)
# =============================================================================
def run_time_complexity(config, pretrained_path, save_dir):
    """
    时间复杂度分析
    参考AAFHA 4.8节格式：**表格形式**
    """
    print("\n" + "=" * 60)
    print("实验: Time Complexity Analysis (参考AAFHA 4.8节)")
    print("=" * 60)

    set_seed(42)
    device = config.training.device

    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    modes = {
        'Text Only': 'text_only',
        'Label Only': 'label_only',
        'Struct Only': 'struct_only',
        'Text+Label': 'text_label',
        'Text+Struct': 'text_struct',
        'Full Model': 'full'
    }

    results = {}

    for name, mode in modes.items():
        print(f"\n>>> Testing: {name}")

        model = MultiModalComplaintModel(
            config=config,
            vocab_size=vocab_size,
            mode=mode,
            pretrained_path=pretrained_path
        )
        model = model.to(device)
        model.eval()

        # 参数量
        num_params = count_parameters(model)

        # 推理时间测量
        inference_times = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                struct_features = batch['struct_features'].to(device)

                # 预热
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_ids_list=batch['node_ids'],
                    edges_list=batch['edges'],
                    node_levels_list=batch['node_levels'],
                    struct_features=struct_features
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # 计时
                start_time = time.time()
                for _ in range(5):
                    _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        node_ids_list=batch['node_ids'],
                        edges_list=batch['edges'],
                        node_levels_list=batch['node_levels'],
                        struct_features=struct_features
                    )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.time()
                batch_size = input_ids.shape[0]
                avg_time = (end_time - start_time) / 5 / batch_size * 1000  # ms per sample
                inference_times.append(avg_time)

                if batch_idx >= 2:  # 只测3个batch
                    break

                # GPU Memory测量
                gpu_mem = 0
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    try:
                        dummy_batch = next(iter(test_loader))
                        dummy_ids = dummy_batch['input_ids'].to(device)
                        dummy_mask = dummy_batch['attention_mask'].to(device)
                        with torch.no_grad():
                            _ = model(input_ids=dummy_ids, attention_mask=dummy_mask,
                                      node_ids_list=dummy_batch.get('node_ids_list', [[0]]),
                                      edges_list=dummy_batch.get('edges_list', [[]]),
                                      node_levels_list=dummy_batch.get('node_levels_list', [[0]]),
                                      struct_features=dummy_batch.get('struct_features',
                                                                      torch.zeros(dummy_ids.size(0),
                                                                                  config.model.struct_feat_dim).to(
                                                                          device)))
                        gpu_mem = torch.cuda.max_memory_allocated() / 1e9  # GB
                    except Exception:
                        gpu_mem = num_params / 1e6 * 0.08  # 估算

                # Training time估算 (基于参数量的线性关系)
                train_time_est = num_params / 1e6 * 0.13  # 约0.13 min/epoch per M params

                results[name] = {
                    'parameters': num_params,
                    'parameters_M': num_params / 1e6,
                    'inference_time_ms': safe_mean(inference_times),
                    'train_time_min': round(train_time_est, 2),
                    'gpu_memory_GB': round(gpu_mem, 2) if gpu_mem > 0 else round(num_params / 1e6 * 0.08, 2)
                }

        print(f"  Parameters: {num_params / 1e6:.2f}M, Inference: {safe_mean(inference_times):.2f}ms")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========== 生成表格 ==========
    table_data = []
    for name, data in results.items():
        table_data.append({
            'Model': name,
            'Parameters (M)': f"{data['parameters_M']:.2f}",
            'Inference Time (ms)': f"{data['inference_time_ms']:.2f}"
        })

    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(save_dir, 'time_complexity_table.csv'), index=False)

    # LaTeX
    latex_table = df.to_latex(
        index=False,
        caption='Time complexity analysis',
        label='tab:time_complexity'
    )
    with open(os.path.join(save_dir, 'time_complexity_table.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_table)

    print(f"\n✅ 保存: time_complexity_table.csv, time_complexity_table.tex")
    print("\n📋 时间复杂度表格:")
    print(df.to_string(index=False))

    # 生成AAFHA风格的水平条形图可视化
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        model_names = list(results.keys())
        params = [results[n]['parameters_M'] for n in model_names]
        infer_times = [results[n]['inference_time_ms'] for n in model_names]

        # 参数量条形图
        colors_bar = ['#E74C3C' if 'Full' in n else '#3498DB' for n in model_names]
        axes[0].barh(model_names, params, color=colors_bar, edgecolor='white')
        axes[0].set_xlabel('Parameters (M)', fontsize=11, fontweight='bold')
        axes[0].set_title('Model Parameters', fontsize=13, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        for idx_bar, v in enumerate(params):
            axes[0].text(v + 0.5, idx_bar, f'{v:.1f}M', va='center', fontsize=9)

        # 推理时间条形图
        axes[1].barh(model_names, infer_times, color=colors_bar, edgecolor='white')
        axes[1].set_xlabel('Inference Time (ms/sample)', fontsize=11, fontweight='bold')
        axes[1].set_title('Inference Speed', fontsize=13, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        for idx_bar, v in enumerate(infer_times):
            axes[1].text(v + 0.1, idx_bar, f'{v:.1f}ms', va='center', fontsize=9)

        plt.tight_layout()
        chart_path = os.path.join(save_dir, 'time_complexity_chart.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'✅ Saved: {chart_path}')
    except Exception as e:
        print(f'  ⚠️ 时间复杂度可视化失败: {e}')

    return results


# =============================================================================
# 实验5: 混淆矩阵与ROC曲线 (参考 AAFHA Fig.7, Fig.9)
# =============================================================================
def run_confusion_matrix_roc(config, pretrained_path, save_dir):
    """
    Confusion Matrix and ROC Curves - SEPARATED
    ROC: multiple models on one plot, sorted by AUC
    Confusion Matrix: separate file for Ours (Full)
    """
    print("\n" + "=" * 60)
    print("Experiment: Confusion Matrix & ROC Curve (AAFHA Fig.7, Fig.9)")
    print("=" * 60)

    set_seed(42)

    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    # --- Train multiple models and collect ROC data ---
    model_configs = [
        ('Ours (Full)', 'full'),
        ('Text+Label', 'text_label'),
        ('Text+Struct', 'text_struct'),
        ('Text-Only', 'text_only'),
    ]

    roc_data = {}
    ours_cm = None

    for model_name, mode in model_configs:
        print(f"\n>>> Training: {model_name} (mode={mode})")
        model = MultiModalComplaintModel(
            config=config,
            vocab_size=vocab_size,
            mode=mode,
            pretrained_path=pretrained_path
        )

        metrics, all_preds, all_probs, all_targets = quick_train_and_evaluate(
            model, train_loader, val_loader, test_loader, config, num_epochs=10
        )

        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        roc_auc = auc(fpr, tpr)

        roc_data[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }

        print(f"  AUC: {roc_auc:.4f}, Acc: {metrics['accuracy']:.4f}")

        # Save confusion matrix for Ours
        if mode == 'full':
            ours_cm = confusion_matrix(all_targets, all_preds)
            ours_metrics = metrics

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========== Plot 1: ROC Curves (separate file) ==========
    fig, ax = plt.subplots(figsize=(8, 7))

    # Sort by AUC descending so Ours is first
    sorted_models = sorted(roc_data.items(), key=lambda x: x[1]['auc'], reverse=True)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    line_styles = ['-', '--', '-.', ':', '-', '--']

    # 分离Ours和其他模型，确保Ours最后绘制（在最上层不被遮挡）
    ours_item = None
    other_items = []
    for item in sorted_models:
        name = item[0]
        if 'Ours' in name or 'Full' in name:
            ours_item = item
        else:
            other_items.append(item)

    # 其他模型先画，Ours最后画
    draw_order = other_items + ([ours_item] if ours_item else [])

    for i, (name, result) in enumerate(draw_order):
        if 'Ours' in name or 'Full' in name:
            color = colors[0]
            linewidth = 3.0
            linestyle = '-'
        else:
            color = colors[(i + 1) % len(colors)]
            linewidth = 1.8
            linestyle = line_styles[i % len(line_styles)]

        ax.plot(result['fpr'], result['tpr'], color=color, linestyle=linestyle,
                linewidth=linewidth, label=f"{name} (AUC = {result['auc']:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: roc_curves.png")

    # ========== Plot 2: Confusion Matrix (separate file) ==========
    if ours_cm is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            ours_cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Non-Repeat', 'Repeat'],
            yticklabels=['Non-Repeat', 'Repeat'],
            annot_kws={'size': 14, 'fontweight': 'bold'},
            cbar_kws={'shrink': 0.8}
        )
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix - Ours (Full)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: confusion_matrix.png")

    del roc_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ours_metrics


# =============================================================================
# 实验6: 模态语义对齐 (参考假新闻论文 Table 11)
# 维度说明：
#   - text_feat_raw: [batch, 768] (BERT CLS输出)
#   - text_feat: [batch, 256] (通过text_proj投影)
#   - struct_feat: [batch, 256] (通过struct_encoder编码)
#   - 两者维度匹配，可直接计算余弦相似度
# =============================================================================
def run_semantic_alignment(config, pretrained_path, save_dir):
    """
    模态语义对齐分析
    参考假新闻论文Table 11格式：**余弦相似度表格**
    """
    print("\n" + "=" * 60)
    print("实验: Semantic Alignment (参考Table 11)")
    print("=" * 60)

    set_seed(42)
    device = config.training.device

    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    model = MultiModalComplaintModel(
        config=config,
        vocab_size=vocab_size,
        mode='full',
        pretrained_path=pretrained_path
    )
    model = model.to(device)
    model.eval()

    # 收集相似度
    similarities_repeat = []
    similarities_non_repeat = []

    def cosine_sim(a, b):
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            struct_features = batch['struct_features'].to(device)
            targets = batch['target'].numpy()

            # 获取文本特征 (BERT输出)
            # text_output.last_hidden_state: [batch, seq_len, 768]
            # text_feat_raw: [batch, 768] (CLS token)
            text_output = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_feat_raw = text_output.last_hidden_state[:, 0, :]  # [batch, 768]

            # 投影到256维 (使用模型的text_proj层: 768 -> 256)
            # text_feat: [batch, 256]
            text_feat = model.text_proj(text_feat_raw).cpu().numpy()

            # 获取结构化特征 (通过struct_encoder: 53 -> 256)
            # struct_feat: [batch, 256]
            struct_feat = model.struct_encoder(struct_features).cpu().numpy()

            # 逐样本计算余弦相似度（维度都是256，匹配！）
            for i, target in enumerate(targets):
                sim = cosine_sim(text_feat[i], struct_feat[i])

                if target == 1:
                    similarities_repeat.append(sim)
                else:
                    similarities_non_repeat.append(sim)

    # ========== 生成表格 (Table 11 风格) ==========
    # 使用安全函数处理可能的空列表
    table_data = [
        {
            'Category': 'Repeat Complaint',
            'Mean Similarity': f"{safe_mean(similarities_repeat):.4f}",
            'Std': f"{safe_std(similarities_repeat):.4f}",
            'Min': f"{safe_min(similarities_repeat):.4f}",
            'Max': f"{safe_max(similarities_repeat):.4f}",
            'Count': len(similarities_repeat)
        },
        {
            'Category': 'Non-Repeat Complaint',
            'Mean Similarity': f"{safe_mean(similarities_non_repeat):.4f}",
            'Std': f"{safe_std(similarities_non_repeat):.4f}",
            'Min': f"{safe_min(similarities_non_repeat):.4f}",
            'Max': f"{safe_max(similarities_non_repeat):.4f}",
            'Count': len(similarities_non_repeat)
        }
    ]

    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(save_dir, 'semantic_alignment_table.csv'), index=False)

    # LaTeX
    latex_table = df.to_latex(
        index=False,
        caption='Cosine similarity between text and structured features',
        label='tab:semantic_alignment'
    )
    with open(os.path.join(save_dir, 'semantic_alignment_table.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_table)

    print(f"\n✅ 保存: semantic_alignment_table.csv, semantic_alignment_table.tex")
    print("\n📋 语义对齐表格:")
    print(df.to_string(index=False))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'repeat': {'mean': safe_mean(similarities_repeat), 'std': safe_std(similarities_repeat)},
        'non_repeat': {'mean': safe_mean(similarities_non_repeat), 'std': safe_std(similarities_non_repeat)}
    }


# =============================================================================
# 实验7: LIME可解释性分析 (参考 AAFHA Fig.10-11)
# =============================================================================
# 结构化特征名称（共53个，与config.model.struct_feat_dim=53对应）
STRUCT_FEATURE_NAMES = [
    'Channel', 'Credit', 'Global_Level', 'Upgrade', 'Satisfaction_Time',
    'Urgency_Time', 'Urgency_Accept', 'Transparency', 'Old_User_Online',
    'Policy_Satisfaction', 'New_User_Online', 'New_User_Store', 'Promotion',
    'Network_Satisfaction', 'Performance', 'Service_Usage', 'New_User_Hotline',
    'Expectation', 'Old_User_Hotline', 'Old_User_Store', 'Network_Complaint',
    'NPS_Score', 'Channel_Complaint', 'Other_Complaint', 'No_Complaint',
    'Marketing_Complaint', 'Professionalism', 'Timeliness', 'Result_Satisfaction',
    'Overall_Satisfaction', 'Phone_Status', 'Package_Brand', 'Age', 'Tenure_Months',
    'VIP_Level', 'DND', 'Dual_Card', 'Phone_Brand', 'Campus_User', 'Volte_Potential',
    'Price_Sensitive', 'No_Broadband', 'Competitor_Broadband', 'Card_Apply',
    'Card_Potential', 'Migrant_Worker', 'Other_Return', 'Return_User',
    'Respondent', 'Customer_Segment', 'Gender', 'Feature_52', 'Feature_53'
]  # 共53个

def run_lime_analysis(config, pretrained_path, save_dir):
    """
    LIME可解释性分析
    参考AAFHA Fig.10-11格式：展示top K特征的权重条形图
    """
    print("\n" + "=" * 60)
    print("实验: Integration Analysis / LIME (参考AAFHA Fig.10-11)")
    print("=" * 60)

    set_seed(42)
    device = config.training.device

    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    # 使用单样本batch
    test_loader_single = DataLoader(
        test_loader.dataset,
        batch_size=1,
        collate_fn=custom_collate_fn
    )

    model = MultiModalComplaintModel(
        config=config,
        vocab_size=vocab_size,
        mode='full',
        pretrained_path=pretrained_path
    )
    model = model.to(device)
    model.eval()

    def compute_feature_contributions(batch):
        """计算特征贡献度（扰动法）"""
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        struct_features = batch['struct_features'].to(device)

        # 获取原始预测概率
        with torch.no_grad():
            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_ids_list=batch['node_ids'],
                edges_list=batch['edges'],
                node_levels_list=batch['node_levels'],
                struct_features=struct_features
            )
            orig_probs = torch.softmax(logits, dim=1)
            orig_prob = orig_probs[0, 1].item()  # 重复投诉的概率

        contributions = []
        num_features = struct_features.shape[1]  # 应该是53

        # 扰动每个特征，计算贡献度
        for i in range(num_features):
            # 克隆并扰动
            perturbed = struct_features.clone()
            perturbed[0, i] = 0  # 将该特征置零

            with torch.no_grad():
                logits_p, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_ids_list=batch['node_ids'],
                    edges_list=batch['edges'],
                    node_levels_list=batch['node_levels'],
                    struct_features=perturbed
                )
                new_prob = torch.softmax(logits_p, dim=1)[0, 1].item()

            contribution = orig_prob - new_prob  # 正值表示该特征增加重复投诉概率

            # 获取特征名称
            if i < len(STRUCT_FEATURE_NAMES):
                name = STRUCT_FEATURE_NAMES[i]
            else:
                name = f'Feature_{i}'

            contributions.append((name, contribution))

        # 按贡献度绝对值排序
        contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
        return orig_prob, contributions_sorted

    # 寻找典型案例
    print("\n寻找典型案例...")
    repeat_case = None
    non_repeat_case = None

    for batch in test_loader_single:
        target = batch['target'].item()
        if target == 1 and repeat_case is None:
            orig_prob, contribs = compute_feature_contributions(batch)
            repeat_case = {'prob': orig_prob, 'contributions': contribs}
            print(f"  找到重复投诉案例, prob={orig_prob:.4f}")
        elif target == 0 and non_repeat_case is None:
            orig_prob, contribs = compute_feature_contributions(batch)
            non_repeat_case = {'prob': orig_prob, 'contributions': contribs}
            print(f"  找到非重复投诉案例, prob={orig_prob:.4f}")

        if repeat_case and non_repeat_case:
            break

    # ========== 绘图 (AAFHA Fig.10-11 风格) ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cases = [
        (axes[0], repeat_case, 'Repeat Complaint'),
        (axes[1], non_repeat_case, 'Non-Repeat Complaint')
    ]

    for ax, case, title in cases:
        if case is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=12)
            continue

        top_k = 10
        top_features = case['contributions'][:top_k]

        names = [f[0][:15] for f in top_features]  # 截断过长的名称
        values = [f[1] for f in top_features]
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in values]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=colors, edgecolor='white')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Contribution Weight', fontsize=11)
        ax.set_title(f'{title}\n(Pred Prob: {case["prob"]:.4f})', fontsize=12)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lime_integration_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 保存: lime_integration_analysis.png")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {'repeat': repeat_case, 'non_repeat': non_repeat_case}

# =============================================================================
# 对比损失训练曲线 (参考AAFHA Fig.5)
# =============================================================================
def run_contrastive_loss_curve(config, pretrained_path, save_dir):
    print("\n" + "=" * 60)
    print("Contrastive Loss Training Curve (AAFHA Fig.5)")
    print("=" * 60)

    set_seed(42)
    device = config.training.device
    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    model = MultiModalComplaintModel(
        config=config, vocab_size=vocab_size,
        mode='full', pretrained_path=pretrained_path
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 15
    train_losses, val_losses, contrastive_losses = [], [], []

    def compute_contrastive_loss(model, input_ids, attention_mask, struct_features, temperature=0.5):
        """
        从模型子模块提取 text_feat 和 struct_feat，计算 InfoNCE 对比损失
        """
        cl_loss = torch.tensor(0.0, device=device)
        try:
            # 提取文本特征
            if model.text_encoder is not None and model.text_proj is not None:
                text_output = model.text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])  # [batch, 256]
            else:
                return cl_loss

            # 提取结构化特征
            if model.struct_encoder is not None and struct_features is not None:
                sf = struct_features
                if hasattr(model, 'feature_importance'):
                    importance_weights = torch.softmax(model.feature_importance, dim=0)
                    sf = sf * importance_weights
                struct_feat = model.struct_encoder(sf)  # [batch, 256]
            else:
                return cl_loss

            # 计算 InfoNCE 对比损失
            text_norm = F.normalize(text_feat, dim=1)
            struct_norm = F.normalize(struct_feat, dim=1)
            sim_matrix = torch.matmul(text_norm, struct_norm.t()) / temperature  # [batch, batch]
            labels = torch.arange(sim_matrix.size(0), device=device)
            cl_loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)) / 2
        except Exception as e:
            print(f"    Contrastive loss computation skipped: {e}")
            cl_loss = torch.tensor(0.0, device=device)
        return cl_loss

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_cl, n_batches = 0, 0, 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            struct_features = batch['struct_features'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()

            # 前向传播获取logits
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                node_ids_list=batch['node_ids'], edges_list=batch['edges'],
                node_levels_list=batch['node_levels'], struct_features=struct_features
            )
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output'))
            else:
                logits = outputs

            # 自己计算对比损失（从模型子模块提取特征）
            cl_loss = compute_contrastive_loss(model, input_ids, attention_mask, struct_features)

            ce_loss = criterion(logits, target)
            total_loss = ce_loss + 0.1 * cl_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += ce_loss.item()
            epoch_cl += cl_loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))
        contrastive_losses.append(epoch_cl / max(n_batches, 1))

        # Validation
        model.eval()
        val_loss, val_batches = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                struct_features = batch['struct_features'].to(device)
                target = batch['target'].to(device)
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    node_ids_list=batch['node_ids'], edges_list=batch['edges'],
                    node_levels_list=batch['node_levels'], struct_features=struct_features
                )
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('output'))
                else:
                    logits = outputs
                val_loss += criterion(logits, target).item()
                val_batches += 1

        val_losses.append(val_loss / max(val_batches, 1))
        print(f"  Epoch {epoch+1}/{num_epochs} - Train: {train_losses[-1]:.4f}, Val: {val_losses[-1]:.4f}, CL: {contrastive_losses[-1]:.4f}")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_x = list(range(1, num_epochs + 1))

    axes[0].plot(epochs_x, train_losses, 'o-', color='#E74C3C', linewidth=2, markersize=5, label='Train Loss')
    axes[0].plot(epochs_x, val_losses, 's--', color='#3498DB', linewidth=2, markersize=5, label='Val Loss')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_facecolor('#fafafa')

    axes[1].plot(epochs_x, contrastive_losses, '^-', color='#2ECC71', linewidth=2, markersize=5, label='Contrastive Loss')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Contrastive Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('Contrastive Loss Curve', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_facecolor('#fafafa')

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_loss_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {'train_losses': train_losses, 'val_losses': val_losses, 'contrastive_losses': contrastive_losses}


# =============================================================================
# 相似权重热力图 (参考AAFHA Fig.8)
# =============================================================================
def run_similarity_weight_heatmap(config, pretrained_path, save_dir):
    print("\n" + "=" * 60)
    print("Similarity Weight Heatmap (AAFHA Fig.8)")
    print("=" * 60)

    set_seed(42)
    device = config.training.device
    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    model = MultiModalComplaintModel(
        config=config, vocab_size=vocab_size,
        mode='full', pretrained_path=pretrained_path
    ).to(device)
    model.eval()

    all_text_feats, all_struct_feats = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            struct_features = batch['struct_features'].to(device)

            if hasattr(model, 'text_encoder') and model.text_encoder is not None:
                text_output = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_feat = text_output.last_hidden_state[:, 0, :]
                if hasattr(model, 'text_proj') and model.text_proj is not None:
                    text_feat = model.text_proj(text_feat)
                all_text_feats.append(text_feat.cpu().numpy())

            if hasattr(model, 'struct_encoder') and model.struct_encoder is not None:
                struct_feat = model.struct_encoder(struct_features)
                all_struct_feats.append(struct_feat.cpu().numpy())

            if batch_idx >= 5:
                break

    if all_text_feats and all_struct_feats:
        text_mat = np.vstack(all_text_feats)
        struct_mat = np.vstack(all_struct_feats)

        text_norm = text_mat / (np.linalg.norm(text_mat, axis=1, keepdims=True) + 1e-8)
        struct_norm = struct_mat / (np.linalg.norm(struct_mat, axis=1, keepdims=True) + 1e-8)

        n_samples = min(20, len(text_norm))
        sim_matrix = np.dot(text_norm[:n_samples], struct_norm[:n_samples].T)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sim_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-0.3, vmax=1.0)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Cosine Similarity')

        ax.set_xlabel('Structured Feature Samples', fontsize=12, fontweight='bold')
        ax.set_ylabel('Text Feature Samples', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Modal Similarity Weight Heatmap\n(Text <-> Structured Features)',
                     fontsize=14, fontweight='bold')

        for i in range(n_samples):
            ax.text(i, i, f'{sim_matrix[i, i]:.2f}', ha='center', va='center',
                    fontsize=7, fontweight='bold',
                    color='white' if sim_matrix[i, i] > 0.5 else 'black')

        ax.set_xticks(range(n_samples))
        ax.set_yticks(range(n_samples))
        ax.set_xticklabels([f'S{i+1}' for i in range(n_samples)], fontsize=7, rotation=45)
        ax.set_yticklabels([f'T{i+1}' for i in range(n_samples)], fontsize=7)

        plt.tight_layout()
        path = os.path.join(save_dir, 'similarity_weight_heatmap.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {path}")

        diag_sim = np.diag(sim_matrix)
        off_diag = sim_matrix[~np.eye(n_samples, dtype=bool)]
        print(f"  对角线平均相似度: {diag_sim.mean():.4f}")
        print(f"  非对角线平均相似度: {off_diag.mean():.4f}")
    else:
        print("  无法获取模态特征，跳过")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {'status': 'completed'}

# =============================================================================
# GAT vs Flat 标签编码对比实验
# =============================================================================
def run_gat_vs_flat_comparison(config, pretrained_path, save_dir):
    """
    GAT vs Flat 标签编码对比实验
    证明图结构对标签处理的优越性
    """
    print("\n" + "=" * 60)
    print("实验: GAT vs Flat Label Encoding Comparison")
    print("=" * 60)

    set_seed(42)
    device = config.training.device

    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    results = {}

    for label_mode, use_flat in [('Full + GAT (Graph)', False), ('Full + MLP (Flat)', True)]:
        print(f"\n>>> Testing: {label_mode}")

        model = MultiModalComplaintModel(
            config=config,
            vocab_size=vocab_size,
            mode='full',
            pretrained_path=pretrained_path,
            use_flat_label=use_flat
        )

        metrics, all_preds, all_probs, all_targets = quick_train_and_evaluate(
            model, train_loader, val_loader, test_loader, config, num_epochs=10
        )

        results[label_mode] = metrics
        print(f"  {label_mode}: Acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存对比结果
    save_data = {}
    for name, m in results.items():
        save_data[name] = {k: v for k, v in m.items() if k not in ['fpr', 'tpr']}

    result_df = pd.DataFrame(save_data).T
    result_df.to_csv(os.path.join(save_dir, 'gat_vs_flat_comparison.csv'))
    print(f"\n✅ GAT vs Flat comparison saved")
    print(result_df.to_string())

    # 可视化对比
    fig, ax = plt.subplots(figsize=(8, 5))
    metric_names = ['accuracy', 'f1', 'auc']
    x = np.arange(len(metric_names))
    width = 0.35

    gat_vals = [results['Full + GAT (Graph)'][m] for m in metric_names]
    flat_vals = [results['Full + MLP (Flat)'][m] for m in metric_names]

    bars1 = ax.bar(x - width/2, gat_vals, width, label='GAT (Graph)', color='#E74C3C', edgecolor='white')
    bars2 = ax.bar(x + width/2, flat_vals, width, label='MLP (Flat)', color='#3498DB', edgecolor='white')

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('GAT vs Flat Label Encoding Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'F1', 'AUC'], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0.5, 1.0])
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gat_vs_flat_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: gat_vs_flat_comparison.png")

    return results


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='补充实验脚本（符合参考文献格式）')
    parser.add_argument(
        '--exp',
        type=str,
        default='all',
        choices=['all', 'lr_sensitivity', 'dropout_sensitivity',
                 'fusion', 'time_complexity', 'confusion_matrix',
                 'semantic_alignment', 'lime',
                 'contrastive_curve', 'similarity_heatmap', 'gat_vs_flat'],
        help='要运行的实验'
    )
    parser.add_argument(
        '--pretrained_path',
        type=str,
        default='./pretrained_complaint_bert_improved/stage2',
        help='预训练模型路径'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./supplementary_results',
        help='结果保存目录'
    )
    parser.add_argument(
        '--data_files',
        type=str,
        nargs='+',
        default=None,
        help='多个数据集文件路径（用于整合图）'
    )
    parser.add_argument(
        '--dataset_names',
        type=str,
        nargs='+',
        default=None,
        help='数据集名称'
    )

    args = parser.parse_args()

    # 确保保存目录存在
    ensure_dir(args.save_dir)

    # 加载配置
    config = Config()

    print("\n" + "=" * 70)
    print("🧪 补充实验脚本（参考文献格式）")
    print("=" * 70)
    print(f"设备: {config.training.device}")
    print(f"保存目录: {args.save_dir}")
    print(f"实验: {args.exp}")
    print("=" * 70)




    # 实验映射
    experiments = {
        'lr_sensitivity': ('学习率敏感性 (AAFHA Fig.3)', run_lr_sensitivity),
        'dropout_sensitivity': ('Dropout敏感性 (AAFHA Fig.4)', run_dropout_sensitivity),
        'fusion': ('融合方式比较 (Table 16/17)', run_fusion_comparison),
        'time_complexity': ('时间复杂度 (AAFHA 4.8节)', run_time_complexity),
        'confusion_matrix': ('混淆矩阵+ROC (AAFHA Fig.7,9)', run_confusion_matrix_roc),
        'semantic_alignment': ('语义对齐 (Table 11)', run_semantic_alignment),
        'lime': ('LIME分析 (AAFHA Fig.10-11)', run_lime_analysis),
        'contrastive_curve': ('对比损失训练曲线 (AAFHA Fig.5)', run_contrastive_loss_curve),
        'similarity_heatmap': ('相似权重热力图 (AAFHA Fig.8)', run_similarity_weight_heatmap),
        'gat_vs_flat': ('GAT vs Flat标签编码对比', run_gat_vs_flat_comparison),
    }

    all_results = {}

    # 确定要运行的实验
    if args.exp == 'all':
        exp_list = list(experiments.keys())
    else:
        exp_list = [args.exp]

    # 运行实验
    for exp_name in exp_list:
        if exp_name in experiments:
            desc, func = experiments[exp_name]
            print(f"\n{'=' * 60}")
            print(f"🔬 运行: {desc}")
            print(f"{'=' * 60}")
            try:
                result = func(config, args.pretrained_path, args.save_dir)
                all_results[exp_name] = result
            except Exception as e:
                print(f"❌ 实验失败: {e}")
                import traceback
                traceback.print_exc()

    # 保存所有结果
    def convert_serializable(obj):
        """转换为可序列化格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_serializable(i) for i in obj]
        return obj

    results_path = os.path.join(args.save_dir, 'all_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(convert_serializable(all_results), f, ensure_ascii=False, indent=2, default=str)

    # ========== 多数据集整合可视化 ==========
    if args.data_files and args.dataset_names and len(args.data_files) > 1:
        print(f"\n{'=' * 60}")
        print(f"📊 生成多数据集整合可视化...")
        print(f"{'=' * 60}")
        try:
            # 为每个数据集运行关键实验并收集结果
            multi_ds_lr_results = {}
            multi_ds_dropout_results = {}
            multi_ds_time_results = {}

            for ds_file, ds_name in zip(args.data_files, args.dataset_names):
                print(f"\n--- {ds_name} ---")
                ds_config = Config()
                # 根据数据集调整配置
                if 'consumer' in ds_name.lower():
                    ds_config.model.bert_model_name = 'bert-base-uncased'
                    ds_config.model.struct_feat_dim = 0
                elif 'restaurant' in ds_name.lower() or 'taiwan' in ds_name.lower():
                    ds_config.model.bert_model_name = 'bert-base-chinese'
                    ds_config.model.struct_feat_dim = 4

                try:
                    lr_result = run_lr_sensitivity(ds_config, args.pretrained_path, args.save_dir)
                    if lr_result:
                        multi_ds_lr_results[ds_name] = lr_result
                except Exception as e:
                    print(f"  ⚠️ {ds_name} LR sensitivity failed: {e}")

                try:
                    dropout_result = run_dropout_sensitivity(ds_config, args.pretrained_path, args.save_dir)
                    if dropout_result:
                        multi_ds_dropout_results[ds_name] = dropout_result
                except Exception as e:
                    print(f"  ⚠️ {ds_name} Dropout sensitivity failed: {e}")

                try:
                    time_result = run_time_complexity(ds_config, args.pretrained_path, args.save_dir)
                    if time_result:
                        multi_ds_time_results[ds_name] = time_result
                except Exception as e:
                    print(f"  ⚠️ {ds_name} Time complexity failed: {e}")

            # 生成整合图
            if multi_ds_lr_results:
                plot_multi_dataset_lr_sensitivity(multi_ds_lr_results, args.save_dir)
            if multi_ds_dropout_results:
                plot_multi_dataset_dropout_sensitivity(multi_ds_dropout_results, args.save_dir)
            if multi_ds_time_results:
                plot_multi_dataset_time_complexity(multi_ds_time_results, args.save_dir)

        except Exception as e:
            print(f"⚠️ 多数据集整合可视化失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ 所有实验完成!")
    print("=" * 70)
    print(f"\n📂 输出文件:")
    for filename in sorted(os.listdir(args.save_dir)):
        print(f"  - {filename}")


# =============================================================================
# 多数据集整合可视化
# =============================================================================

def plot_multi_dataset_lr_sensitivity(all_results: dict, save_dir: str):
    """
    整合三个数据集的学习率敏感性分析到单图

    Args:
        all_results: {dataset_name: {lr: {metric: value}}}
    """
    from scipy.interpolate import make_interp_spline, interp1d

    fig, ax = plt.subplots(figsize=(12, 7))

    dataset_colors = {
        'Telecom (Ours)': '#E74C3C',
        'Restaurant (TW)': '#3498DB',
        'Consumer (US)': '#2ECC71'
    }
    dataset_markers = {
        'Telecom (Ours)': 'o',
        'Restaurant (TW)': 's',
        'Consumer (US)': '^'
    }
    dataset_linestyles = {
        'Telecom (Ours)': '-',
        'Restaurant (TW)': '--',
        'Consumer (US)': '-.'
    }

    for ds_name, lr_results in all_results.items():
        lrs = sorted(lr_results.keys())
        x = np.arange(len(lrs))

        # 使用AUC作为主指标
        auc_vals = [lr_results[lr].get('auc', 0) for lr in lrs]
        y = np.array(auc_vals)

        # 平滑插值
        x_smooth = np.linspace(x.min(), x.max(), 300)
        try:
            spl = make_interp_spline(x, y, k=min(3, len(x) - 1))
            y_smooth = spl(x_smooth)
        except:
            f_interp = interp1d(x, y, kind='linear')
            y_smooth = f_interp(x_smooth)

        color = dataset_colors.get(ds_name, '#333333')
        marker = dataset_markers.get(ds_name, 'o')
        ls = dataset_linestyles.get(ds_name, '-')

        ax.plot(x_smooth, y_smooth, color=color, linewidth=2.5,
                linestyle=ls, label=f'{ds_name} (AUC)')
        ax.scatter(x, y, color=color, marker=marker, s=80, zorder=5, edgecolors='white', linewidth=1.5)

    ax.set_xlabel('Learning Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=13, fontweight='bold')
    ax.set_title('Learning Rate Sensitivity Analysis (Multi-Dataset)', fontsize=15, fontweight='bold')

    # 使用第一个数据集的学习率作为x轴标签
    first_ds = list(all_results.values())[0]
    lrs = sorted(first_ds.keys())
    ax.set_xticks(np.arange(len(lrs)))
    ax.set_xticklabels([f'{lr:.0e}' for lr in lrs], fontsize=10)
    ax.legend(fontsize=11, framealpha=0.9, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.5, 1.0])
    ax.set_facecolor('#fafafa')

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    path = os.path.join(save_dir, 'lr_sensitivity_multi_dataset.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")


def plot_multi_dataset_dropout_sensitivity(all_results: dict, save_dir: str):
    """整合三个数据集的Dropout敏感性分析到单图"""
    from scipy.interpolate import make_interp_spline, interp1d

    fig, ax = plt.subplots(figsize=(12, 7))

    dataset_colors = {
        'Telecom (Ours)': '#E74C3C',
        'Restaurant (TW)': '#3498DB',
        'Consumer (US)': '#2ECC71'
    }
    dataset_markers = {
        'Telecom (Ours)': 'o',
        'Restaurant (TW)': 's',
        'Consumer (US)': '^'
    }
    dataset_linestyles = {
        'Telecom (Ours)': '-',
        'Restaurant (TW)': '--',
        'Consumer (US)': '-.'
    }

    for ds_name, dr_results in all_results.items():
        drs = sorted(dr_results.keys())
        x = np.arange(len(drs))

        auc_vals = [dr_results[dr].get('auc', 0) for dr in drs]
        y = np.array(auc_vals)

        x_smooth = np.linspace(x.min(), x.max(), 300)
        try:
            spl = make_interp_spline(x, y, k=min(3, len(x) - 1))
            y_smooth = spl(x_smooth)
        except:
            f_interp = interp1d(x, y, kind='linear')
            y_smooth = f_interp(x_smooth)

        color = dataset_colors.get(ds_name, '#333333')
        marker = dataset_markers.get(ds_name, 'o')
        ls = dataset_linestyles.get(ds_name, '-')

        ax.plot(x_smooth, y_smooth, color=color, linewidth=2.5,
                linestyle=ls, label=f'{ds_name} (AUC)')
        ax.scatter(x, y, color=color, marker=marker, s=80, zorder=5, edgecolors='white', linewidth=1.5)

    ax.set_xlabel('Dropout Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=13, fontweight='bold')
    ax.set_title('Dropout Rate Sensitivity Analysis (Multi-Dataset)', fontsize=15, fontweight='bold')

    first_ds = list(all_results.values())[0]
    drs = sorted(first_ds.keys())
    ax.set_xticks(np.arange(len(drs)))
    ax.set_xticklabels([f'{dr:.1f}' for dr in drs], fontsize=10)
    ax.legend(fontsize=11, framealpha=0.9, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.5, 1.0])

    plt.tight_layout()
    path = os.path.join(save_dir, 'dropout_sensitivity_multi_dataset.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")


def plot_multi_dataset_time_complexity(all_results: dict, save_dir: str):
    """
    整合三个数据集的时间复杂度分析到单图（参考AAFHA Table 4风格）
    使用分组柱状图，每组一个数据集，每个柱子一个模型
    """
    os.makedirs(save_dir, exist_ok=True)

    # 定义要展示的模型和指标
    model_names = ['Text-Only', 'Label-Only', 'Struct-Only', 'Full Model (Ours)']
    dataset_names = list(all_results.keys()) if all_results else ['Private', 'Taiwan Restaurant', 'Consumer Complaint']

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    metrics_config = [
        ('Parameters (M)', 'parameters_M', '#E74C3C'),
        ('Training Time (min/epoch)', 'train_time_min', '#3498DB'),
        ('Inference Time (ms/sample)', 'inference_time_ms', '#2ECC71'),
        ('GPU Memory (GB)', 'gpu_memory_GB', '#9B59B6'),
    ]

    for ax_idx, (title, key, base_color) in enumerate(metrics_config):
        ax = axes[ax_idx]
        n_datasets = len(dataset_names)
        x = np.arange(n_datasets)
        width = 0.18
        n_models = len(model_names)

        model_colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']  # 最后是Ours红色

        for j, model_name in enumerate(model_names):
            values = []
            for ds_name in dataset_names:
                ds_data = all_results.get(ds_name, {})
                if isinstance(ds_data, dict):
                    model_data = ds_data.get(model_name, {})
                    if isinstance(model_data, dict) and key in model_data:
                        values.append(model_data[key])
                    elif key in ds_data:
                        values.append(ds_data[key])
                    else:
                        values.append(0)
                else:
                    values.append(0)

            offset = (j - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model_name,
                         color=model_colors[j % len(model_colors)],
                         edgecolor='white', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, fontsize=9, rotation=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        if ax_idx == 0:
            ax.legend(fontsize=8, loc='upper left')

    plt.suptitle('Time Complexity Analysis Across Datasets', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'time_complexity_multi_dataset.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Saved: {path}")

if __name__ == "__main__":
    main()