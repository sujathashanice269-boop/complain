"""
客户重复投诉预测 - 训练脚本(修复版)
✅ 修复：模型初始化参数匹配问题
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime
import gc
import json
from transformers import BertTokenizer
import warnings
warnings.filterwarnings('ignore')

# 使用正确的引用
from data_processor import ComplaintDataProcessor, custom_collate_fn
from model import MultiModalComplaintModel
from config import Config


class ComplaintDataset(Dataset):
    """投诉数据集"""

    def __init__(self, text_data, node_ids_list, edges_list, node_levels_list,
                 struct_features, targets=None):
        self.input_ids = text_data['input_ids']
        self.attention_mask = text_data['attention_mask']
        self.node_ids_list = node_ids_list
        self.edges_list = edges_list
        self.node_levels_list = node_levels_list
        self.struct_features = struct_features
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
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


def create_data_loaders(config, pretrained_path=None):
    """
    创建数据加载器
    """
    print(f"加载数据: {config.training.data_file}")

    # 初始化处理器
    processor = ComplaintDataProcessor(
        config=config,
        user_dict_file=config.data.user_dict_file
    )

    # 尝试加载预训练处理器
    if pretrained_path:
        processor_path = os.path.join(pretrained_path, 'processor.pkl')
        if os.path.exists(processor_path):
            print(f"加载预训练处理器: {processor_path}")
            processor.load(processor_path)

    # 准备数据集
    print("准备数据集...")
    data = processor.prepare_datasets(
        train_file=config.training.data_file,
        for_pretrain=False
    )

    print(f"数据集大小: {len(data['targets'])}")
    print(f"标签词汇表大小: {data['vocab_size']}")

    # 划分数据集
    total_size = len(data['targets'])
    indices = torch.randperm(total_size).tolist()

    train_size = int(total_size * 0.6)
    val_size = int(total_size * 0.2)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    print(f"训练集: {len(train_indices)}, 验证集: {len(val_indices)}, 测试集: {len(test_indices)}")

    # 分割数据
    def split_data(data, indices):
        return {
            'text_data': {
                'input_ids': data['text_data']['input_ids'][indices],
                'attention_mask': data['text_data']['attention_mask'][indices]
            },
            'node_ids_list': [data['node_ids_list'][i] for i in indices],
            'edges_list': [data['edges_list'][i] for i in indices],
            'node_levels_list': [data['node_levels_list'][i] for i in indices],
            'struct_features': data['struct_features'][indices],
            'targets': data['targets'][indices]
        }

    train_data = split_data(data, train_indices)
    val_data = split_data(data, val_indices)
    test_data = split_data(data, test_indices)

    # 创建数据集
    train_dataset = ComplaintDataset(
        train_data['text_data'],
        train_data['node_ids_list'],
        train_data['edges_list'],
        train_data['node_levels_list'],
        train_data['struct_features'],
        train_data['targets']
    )

    val_dataset = ComplaintDataset(
        val_data['text_data'],
        val_data['node_ids_list'],
        val_data['edges_list'],
        val_data['node_levels_list'],
        val_data['struct_features'],
        val_data['targets']
    )

    test_dataset = ComplaintDataset(
        test_data['text_data'],
        test_data['node_ids_list'],
        test_data['edges_list'],
        test_data['node_levels_list'],
        test_data['struct_features'],
        test_data['targets']
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size,
                              shuffle=True, collate_fn=custom_collate_fn,drop_last=True)  # ✅ 新增
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size,
                            collate_fn=custom_collate_fn)  # ✅ 新增
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size,
                             collate_fn=custom_collate_fn)  # ✅ 新增

    # 元数据
    metadata = {
        'vocab_size': data['vocab_size'],
        'processor': processor
    }

    return train_loader, val_loader, test_loader, metadata
def train_model(model, train_loader, val_loader, config, device='cuda'):
    """训练模型"""
    model = model.to(device)

    # 损失函数和优化器
    # 支持类别权重（用于不平衡数据集，如台湾餐厅数据集）
    class_weight = getattr(config.training, 'class_weight', None)
    if class_weight is not None:
        weight_tensor = torch.tensor(class_weight, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print(f"  使用加权损失: {class_weight}")
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0

    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_f1': [],
        'val_acc': []
    }

    print(f"\n开始训练,共 {config.training.num_epochs} 个epoch...")

    for epoch in range(config.training.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            node_ids_list = batch['node_ids']
            edges_list = batch['edges']
            node_levels_list = batch['node_levels']
            struct_features = batch['struct_features'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_ids_list=node_ids_list,
                edges_list=edges_list,
                node_levels_list=node_levels_list,
                struct_features=struct_features
            )

            loss = criterion(logits, target)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_targets.extend(target.cpu().numpy())

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 验证阶段
        model.eval()
        val_loss = 0
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
                target = batch['target'].to(device)

                logits, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_ids_list=node_ids_list,
                    edges_list=edges_list,
                    node_levels_list=node_levels_list,
                    struct_features=struct_features
                )

                loss = criterion(logits, target)

                val_loss += loss.item()
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(target.cpu().numpy())
                val_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

        # 计算指标
        train_acc = accuracy_score(train_targets, train_preds)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        val_auc = roc_auc_score(val_targets, val_probs)

        # 记录历史
        training_history['train_loss'].append(train_loss/len(train_loader))
        training_history['val_loss'].append(val_loss/len(val_loader))
        training_history['val_auc'].append(val_auc)
        training_history['val_f1'].append(val_f1)
        training_history['val_acc'].append(val_acc)

        # 调整学习率
        scheduler.step(val_auc)

        # 打印结果
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train - Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  新的最佳模型! AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= config.training.early_stopping_patience:
            print("早停触发,停止训练")
            break

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_auc, training_history
def evaluate_model(model, test_loader, device='cuda'):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    print("\n评估模型...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            node_ids_list = batch['node_ids']
            edges_list = batch['edges']
            node_levels_list = batch['node_levels']
            struct_features = batch['struct_features'].to(device)
            target = batch['target'].to(device)

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_ids_list=node_ids_list,
                edges_list=edges_list,
                node_levels_list=node_levels_list,
                struct_features=struct_features
            )

            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    # 计算指标
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1': f1_score(all_targets, all_preds, average='weighted'),
        'auc': roc_auc_score(all_targets, all_probs)
    }

    return metrics


def main():
    """主训练流程"""

    # 使用Config对象
    config = Config()

    device = config.training.device
    print(f"使用设备: {device}")

    # 创建保存目录
    os.makedirs(config.training.save_dir, exist_ok=True)

    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader, metadata = create_data_loaders(
        config,
        pretrained_path=config.training.pretrain_save_dir
    )

    print(f"词汇表大小: {metadata['vocab_size']}")

    # ✅ 修复：使用新的模型初始化方式
    print("\n创建模型...")
    model = MultiModalComplaintModel(
        config=config,
        vocab_size=metadata['vocab_size'],
        mode='full',
        pretrained_path=config.training.pretrain_save_dir
    )

    # 训练模型
    print("\n开始训练...")
    model, best_auc = train_model(
        model, train_loader, val_loader, config, device
    )

    # 在测试集上评估
    print("\n在测试集上评估...")
    test_metrics = evaluate_model(model, test_loader, device)

    print("\n测试集性能:")
    print(f"  准确率: {test_metrics['accuracy']:.4f}")
    print(f"  精确率: {test_metrics['precision']:.4f}")
    print(f"  召回率: {test_metrics['recall']:.4f}")
    print(f"  F1分数: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")

    # 保存模型
    model_path = os.path.join(config.training.save_dir, 'best_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_metadata': {
            'vocab_size': metadata['vocab_size']
        },
        'best_auc': best_auc,
        'test_metrics': test_metrics
    }, model_path)

    print(f"\n模型已保存: {model_path}")
    print(f"最佳验证AUC: {best_auc:.4f}")

    # 保存结果到JSON
    results_path = os.path.join(config.training.save_dir, 'training_results.json')
    results = {
        'validation': {
            'best_auc': float(best_auc)
        },
        'test': {k: float(v) for k, v in test_metrics.items()}
    }

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"训练结果已保存: {results_path}")

    # 清理内存
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return test_metrics


if __name__ == "__main__":
    main()