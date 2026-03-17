"""
============================================================
run_fusion_ablation.py - 融合机制消融实验 (补齐缺失数据)
============================================================
补齐论文消融表中缺失的实验数据:
  1. w/o GAT encoding: 三模态 + Flat MLP编码 (三个数据集各1次)
  2. w/o Cross-modal attention: concat融合 (台湾/NHTSA各1次)
  3. w/o Gated fusion: simple_attention融合 (台湾/NHTSA各1次)
  4. Full Model (gated + GAT): 作为参考基准 (台湾/NHTSA各1次)

共 7 + 2(参考) = 9 次训练 (移动客户concat/simple_attention已有)

原理:
  三个数据集的模型在 mode='full' 时都有 self.text_led_cross_modal
  (TextLedCrossModalAttention), 调用接口完全一致:
    (text_tokens, label_nodes, struct_tokens, label_mask) → (t, l, s, attn)
  替换该属性为 ConcatFusion / SimpleAttentionFusion 即可实现融合消融

用法:
    python run_fusion_ablation.py
    python run_fusion_ablation.py --dataset telecom
    python run_fusion_ablation.py --dataset taiwan
    python run_fusion_ablation.py --dataset nhtsa
    python run_fusion_ablation.py --save_dir ./outputs/fusion_ablation
============================================================
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONUNBUFFERED'] = '1'
import sys
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# ============================================================
# 路径设置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

DEFAULT_SAVE_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'fusion_ablation')

# ============================================================
# 从主文件导入
# ============================================================
from cross_dataset_experiments import (
    prepare_data_for_dataset,
    evaluate_model,
    set_seed,
    cleanup,
    ensure_dir,
    _load_standalone_modules,
)
from model import MultiModalComplaintModel

# 延迟加载 standalone 模块
taiwan_mod = None
nhtsa_mod = None


def _ensure_standalone():
    """确保 standalone 模块已加载"""
    global taiwan_mod, nhtsa_mod
    if taiwan_mod is None or nhtsa_mod is None:
        _load_standalone_modules()
        import cross_dataset_experiments as cde
        taiwan_mod = getattr(cde, 'taiwan_mod', None)
        nhtsa_mod = getattr(cde, 'nhtsa_mod', None)
        if taiwan_mod is None:
            try:
                import importlib
                taiwan_mod = importlib.import_module('run_taiwan_restaurant_standalone')
            except Exception as e:
                print(f"  ⚠️ 台湾模块加载失败: {e}")
        if nhtsa_mod is None:
            try:
                import importlib
                nhtsa_mod = importlib.import_module('run_nhtsa_standalone')
            except Exception as e:
                print(f"  ⚠️ NHTSA模块加载失败: {e}")


# ============================================================
# 融合模块 (从 fusion_models.py 提取, 避免复杂导入链)
# ============================================================
class ConcatFusion(nn.Module):
    """纯拼接融合 - 无跨模态交互 (对应 w/o Cross-modal attention)"""

    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim

    def forward(self, text_feat, label_feat, struct_feat,
                label_mask=None, return_attention=False):
        # Pool text
        if text_feat.dim() == 3:
            text_pooled = text_feat.mean(dim=1)
        else:
            text_pooled = text_feat

        # Pool label (masked)
        if label_feat.dim() == 3:
            if label_mask is not None:
                valid_mask = ~label_mask
                mask_expanded = valid_mask.unsqueeze(-1).float()
                label_pooled = (label_feat * mask_expanded).sum(dim=1) / (
                    mask_expanded.sum(dim=1) + 1e-8)
            else:
                label_pooled = label_feat.mean(dim=1)
        else:
            label_pooled = label_feat

        # Pool struct
        if struct_feat.dim() == 3:
            struct_pooled = struct_feat.mean(dim=1)
        else:
            struct_pooled = struct_feat

        attention_weights = {} if return_attention else None
        return text_pooled, label_pooled, struct_pooled, attention_weights


class SimpleAttentionFusion(nn.Module):
    """简单注意力融合 - 无门控机制 (对应 w/o Gated fusion)"""

    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.text_to_label_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.text_to_struct_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.text_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.label_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.struct_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.text_norm = nn.LayerNorm(dim)
        self.label_norm = nn.LayerNorm(dim)
        self.struct_norm = nn.LayerNorm(dim)

    def forward(self, text_tokens, label_tokens, struct_tokens,
                label_mask=None, return_attention=False):
        attention_weights = {}
        # Self-attention
        text_self, _ = self.text_self_attn(
            text_tokens, text_tokens, text_tokens)
        text_tokens = self.text_norm(text_tokens + text_self)

        label_self, _ = self.label_self_attn(
            label_tokens, label_tokens, label_tokens,
            key_padding_mask=label_mask)
        label_tokens = self.label_norm(label_tokens + label_self)

        struct_self, _ = self.struct_self_attn(
            struct_tokens, struct_tokens, struct_tokens)
        struct_tokens = self.struct_norm(struct_tokens + struct_self)

        # Cross-modal (text queries others, no gating)
        text_to_label, attn_t2l = self.text_to_label_attn(
            text_tokens, label_tokens, label_tokens,
            key_padding_mask=label_mask,
            need_weights=return_attention,
            average_attn_weights=False)
        text_to_struct, attn_t2s = self.text_to_struct_attn(
            text_tokens, struct_tokens, struct_tokens,
            need_weights=return_attention,
            average_attn_weights=False)

        # Simple addition (no learnable gating weights)
        text_enhanced = text_tokens + 0.5 * text_to_label + 0.5 * text_to_struct
        label_enhanced = label_tokens
        struct_enhanced = struct_tokens

        # Pooling
        text_pooled = text_enhanced.mean(dim=1)
        if label_mask is not None:
            valid_mask = ~label_mask
            mask_expanded = valid_mask.unsqueeze(-1).float()
            label_pooled = (label_enhanced * mask_expanded).sum(dim=1) / (
                mask_expanded.sum(dim=1) + 1e-8)
        else:
            label_pooled = label_enhanced.mean(dim=1)
        struct_pooled = struct_enhanced.mean(dim=1)

        if return_attention:
            attention_weights = {
                'text_to_label': attn_t2l,
                'text_to_struct': attn_t2s,
            }
        return text_pooled, label_pooled, struct_pooled, attention_weights


# ============================================================
# 模型创建 (支持 use_flat_label + 融合类型替换)
# ============================================================
def create_model_extended(config, vocab_size, pretrained_path=None,
                          model_source='main', use_flat_label=False,
                          fusion_type='gated'):
    """
    创建模型并根据需要替换融合模块

    Args:
        fusion_type: 'gated' (Full Model)
                     'concat' (w/o Cross-modal attention)
                     'simple_attention' (w/o Gated fusion)
        use_flat_label: True → w/o GAT encoding (Flat MLP替代GAT)

    Returns:
        model (nn.Module)
    """
    _ensure_standalone()

    mode = 'full'

    # 创建基础模型 (所有都用 mode='full')
    if model_source == 'taiwan':
        model = taiwan_mod.MultiModalComplaintModel(
            config=config, vocab_size=vocab_size, mode=mode,
            pretrained_path=pretrained_path, use_flat_label=use_flat_label)
    elif model_source == 'nhtsa':
        model = nhtsa_mod.MultiModalComplaintModel(
            config=config, vocab_size=vocab_size, mode=mode,
            pretrained_path=pretrained_path, use_flat_label=use_flat_label)
    else:
        model = MultiModalComplaintModel(
            config=config, vocab_size=vocab_size, mode=mode,
            pretrained_path=pretrained_path, use_flat_label=use_flat_label)

    # 替换融合模块 (monkey-patch)
    if fusion_type == 'concat' and hasattr(model, 'text_led_cross_modal'):
        print("    🔄 替换融合: TextLedCrossModalAttention → ConcatFusion")
        model.text_led_cross_modal = ConcatFusion(dim=256)
    elif fusion_type == 'simple_attention' and hasattr(model, 'text_led_cross_modal'):
        print("    🔄 替换融合: TextLedCrossModalAttention → SimpleAttentionFusion")
        model.text_led_cross_modal = SimpleAttentionFusion(
            dim=256, num_heads=4, dropout=0.1)

    return model


# ============================================================
# 稳定训练函数 (与 run_three_charts.py 一致的策略)
# ============================================================
def train_model_for_fusion(config, train_loader, val_loader, vocab_size,
                           pretrained_path=None, model_source='main',
                           use_flat_label=False, fusion_type='gated',
                           num_epochs=15):
    """
    训练融合消融模型

    策略: 冻结BERT底层 + 分层LR + warmup + 梯度裁剪
    """
    device = config.training.device

    # 创建模型
    model = create_model_extended(
        config, vocab_size,
        pretrained_path=pretrained_path,
        model_source=model_source,
        use_flat_label=use_flat_label,
        fusion_type=fusion_type)
    model = model.to(device)

    # ====== 冻结BERT底层 ======
    freeze_layers = 0
    if hasattr(model, 'text_encoder') and model.text_encoder is not None:
        _train_size = len(train_loader.dataset) if train_loader else 0
        if _train_size < 20000:
            freeze_layers = 8  # 台湾/NHTSA: 冻结0-8

        if freeze_layers > 0:
            for name, param in model.text_encoder.named_parameters():
                if 'embeddings' in name:
                    param.requires_grad = False
                elif 'encoder.layer.' in name:
                    layer_num = int(
                        name.split('encoder.layer.')[1].split('.')[0])
                    if layer_num <= freeze_layers:
                        param.requires_grad = False
            frozen = sum(p.numel() for p in model.text_encoder.parameters()
                         if not p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"    ❄️ 冻结BERT层0-{freeze_layers}: "
                  f"{frozen / 1e6:.1f}M / {total / 1e6:.1f}M")

    # ====== 分层学习率 ======
    base_lr = config.training.learning_rate
    if hasattr(model, 'text_encoder') and model.text_encoder is not None:
        bert_trainable = [p for n, p in model.text_encoder.named_parameters()
                          if p.requires_grad]
        other_params = [p for n, p in model.named_parameters()
                        if 'text_encoder' not in n and p.requires_grad]
        bert_lr = base_lr * 0.2
        optimizer = torch.optim.AdamW([
            {'params': bert_trainable, 'lr': bert_lr, 'weight_decay': 0.01},
            {'params': other_params, 'lr': base_lr, 'weight_decay': 0.01},
        ])
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=base_lr, weight_decay=0.01)

    # ====== Warmup + Cosine ======
    total_steps = num_epochs * len(train_loader)
    warmup_steps = max(total_steps // 10, 3)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ====== 损失函数 ======
    _cw = getattr(config.training, 'class_weight', None)
    if _cw is not None:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(_cw, dtype=torch.float32).to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # ====== 训练循环 ======
    has_struct = config.model.struct_feat_dim > 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batch = 0

        for batch in train_loader:
            optimizer.zero_grad()
            try:
                logits, _ = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    struct_features=(batch['struct_features'].to(device)
                                     if has_struct else None),
                    node_ids_list=batch['node_ids'],
                    edges_list=batch['edges'],
                    node_levels_list=batch['node_levels'],
                )
                loss = criterion(logits, batch['target'].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                n_batch += 1
            except Exception as e:
                if n_batch == 0:
                    print(f"    [训练警告] {e}")
                continue

        avg_loss = epoch_loss / max(n_batch, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f}")

    # 解冻
    for param in model.parameters():
        param.requires_grad = True

    return model


# ============================================================
# 主实验逻辑
# ============================================================
def run_fusion_ablation(datasets=None, save_dir=DEFAULT_SAVE_DIR):
    """
    运行融合机制消融实验

    对每个数据集运行4种配置:
      gated (Full Model)  → 参考基准
      concat              → w/o Cross-modal attention
      simple_attention    → w/o Gated fusion
      full_flat           → w/o GAT encoding
    """
    ensure_dir(save_dir)

    if datasets is None:
        datasets = ['telecom', 'taiwan', 'nhtsa']

    # 实验配置: (实验名, fusion_type, use_flat_label, 论文中的名称)
    experiments = [
        ('gated', 'gated', False, 'Full Model'),
        ('concat', 'concat', False, 'w/o Cross-modal attention'),
        ('simple_attention', 'simple_attention', False, 'w/o Gated fusion'),
        ('full_flat', 'gated', True, 'w/o GAT encoding'),
    ]

    num_epochs_map = {
        'telecom': 15,
        'taiwan': 20,
        'nhtsa': 20,
    }

    all_results = {}

    for ds_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"📊 数据集: {ds_name}")
        print(f"{'=' * 60}")

        data = prepare_data_for_dataset(ds_name)
        if data is None:
            print(f"  ⚠️ 数据准备失败，跳过 {ds_name}")
            continue

        config = data['config']
        vocab_size = data['vocab_size']
        ms = data.get('model_source', 'main')
        pp = data.get('pretrained_path')
        num_ep = num_epochs_map.get(ds_name, 15)

        ds_results = {}

        for exp_name, fusion_type, use_flat, paper_name in experiments:
            print(f"\n  --- {paper_name} ({exp_name}) ---")
            set_seed(42)
            model = None

            try:
                model = train_model_for_fusion(
                    config, data['train_loader'], data['val_loader'],
                    vocab_size, pretrained_path=pp, model_source=ms,
                    use_flat_label=use_flat, fusion_type=fusion_type,
                    num_epochs=num_ep)

                metrics, _, _, _ = evaluate_model(
                    model, data['test_loader'], config)

                if metrics:
                    ds_results[exp_name] = {
                        'paper_name': paper_name,
                        'accuracy': round(metrics['accuracy'], 4),
                        'precision': round(metrics['precision'], 4),
                        'recall': round(metrics['recall'], 4),
                        'f1': round(metrics['f1'], 4),
                        'auc': round(metrics['auc'], 4),
                    }
                    print(f"    ✅ Acc={metrics['accuracy']:.4f}, "
                          f"F1={metrics['f1']:.4f}, "
                          f"AUC={metrics['auc']:.4f}")
                else:
                    print(f"    ❌ 评估失败")

            except Exception as e:
                print(f"    ❌ 实验失败: {e}")
                import traceback
                traceback.print_exc()

            finally:
                if model is not None:
                    del model
                cleanup()

        all_results[ds_name] = ds_results

    # ============================================================
    # 保存结果
    # ============================================================
    print(f"\n{'=' * 60}")
    print("📋 保存结果")
    print(f"{'=' * 60}")

    # 1. 按数据集分别保存
    for ds_name, ds_res in all_results.items():
        if not ds_res:
            continue
        rows = []
        for exp_name, metrics in ds_res.items():
            rows.append({
                'Experiment': exp_name,
                'Paper Name': metrics['paper_name'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'AUC': metrics['auc'],
            })
        df = pd.DataFrame(rows)
        path = os.path.join(save_dir, f'fusion_ablation_{ds_name}.xlsx')
        df.to_excel(path, index=False)
        print(f"  ✅ {path}")

    # 2. 汇总表 (论文用)
    summary_rows = []
    ds_display = {
        'telecom': 'Telecom (Private)',
        'taiwan': 'Taiwan Restaurant',
        'nhtsa': 'NHTSA Vehicle',
    }
    for ds_name, ds_res in all_results.items():
        for exp_name, metrics in ds_res.items():
            summary_rows.append({
                'Dataset': ds_display.get(ds_name, ds_name),
                'Experiment': metrics['paper_name'],
                'Acc': metrics['accuracy'],
                'Prec': metrics['precision'],
                'Rec': metrics['recall'],
                'F1': metrics['f1'],
                'AUC': metrics['auc'],
            })
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(save_dir, 'fusion_ablation_summary.xlsx')
        summary_df.to_excel(summary_path, index=False)
        print(f"  ✅ {summary_path}")

        # 打印汇总
        print(f"\n{'=' * 80}")
        print("📊 融合机制消融实验汇总 (直接用于论文)")
        print(f"{'=' * 80}")
        print(summary_df.to_string(index=False))

    # 3. JSON
    json_path = os.path.join(save_dir, 'fusion_ablation_all.json')
    try:
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable = {}
        for ds, res in all_results.items():
            serializable[ds] = {}
            for exp, m in res.items():
                serializable[ds][exp] = {
                    k: convert(v) for k, v in m.items()}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"  ✅ {json_path}")
    except Exception as e:
        print(f"  ⚠️ JSON保存失败: {e}")

    return all_results


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='融合机制消融实验 - 补齐缺失数据')
    parser.add_argument(
        '--dataset', type=str, default='all',
        choices=['all', 'telecom', 'taiwan', 'nhtsa'],
        help='要运行的数据集 (default: all)')
    parser.add_argument(
        '--save_dir', type=str, default=DEFAULT_SAVE_DIR,
        help=f'保存目录 (default: {DEFAULT_SAVE_DIR})')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  run_fusion_ablation.py - 融合机制消融实验")
    print("=" * 60)
    print(f"  保存目录: {args.save_dir}")
    print(f"  数据集: {args.dataset}")
    print(f"  设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)

    if args.dataset == 'all':
        datasets = ['telecom', 'taiwan', 'nhtsa']
    else:
        datasets = [args.dataset]

    run_fusion_ablation(datasets=datasets, save_dir=args.save_dir)

    print("\n" + "=" * 60)
    print("  ✅ 融合机制消融实验完成!")
    print(f"  输出目录: {args.save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
