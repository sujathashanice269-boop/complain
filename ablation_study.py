"""
消融实验 - 最终修复版本
✅ 修复: 正确加载预训练的全局标签词汇表
✅ 最小改动,保留所有功能
"""

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import gc
import json
import os
import pandas as pd
from tqdm import tqdm

from config import Config
from data_processor import ComplaintDataProcessor, ComplaintDataset, custom_collate_fn
from model import MultiModalComplaintModel


def train_and_evaluate(model, train_loader, val_loader, test_loader, config, device, exp_name):
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

        num_epochs = 3

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

        num_epochs = 3

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

        num_epochs = 5

        print(f"✅ {exp_name}: 冻结BERT层0-8, 保留9-11, BERT_lr=5e-6, {num_epochs}轮")
    elif exp_name == 'text_struct':
        # text_struct: 文本+结构化 - 独立调参
        for param in model.text_encoder.parameters():
            param.requires_grad = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=3e-5)
        num_epochs = 5
        print(f"✅ {exp_name}: 冻结BERT, 分类头lr=3e-5, {num_epochs}轮")

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


def run_ablation_study(pretrained_path=None, config=None, dataset_name='default'):
    """运行消融实验

    Args:
        pretrained_path: 预训练模型路径
        config: 配置对象 (None则使用默认Config)
        dataset_name: 数据集名称 ('default', 'taiwan', 'nhtsa')
    """
    import pandas as pd
    import tempfile

    print("\n" + "="*60)
    print(f"消融实验开始 (数据集: {dataset_name})")
    print("="*60)

    if config is None:
        config = Config()

    # 消融实验用较少轮数
    config.training.num_epochs = 10
    config.training.batch_size = 16

    # 根据数据集决定实验列表
    has_struct = (config.model.struct_feat_dim > 0)

    if has_struct:
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
    else:
        # 双模态数据集: 无结构化特征, 只有text+label
        experiments = [
            ('full_model', 'text_label'),
            ('text_only', 'text_only'),
            ('label_only', 'label_only'),
            ('text_label', 'text_label'),
            ('No_pretrain', 'text_label'),
            ('label_gat', 'label_only'),
            ('label_flat', 'label_only'),
        ]

    results = {}
    experiment_seeds = {
        'full_model': 42, 'text_only': 43, 'label_only': 44,
        'struct_only': 45, 'text_label': 46, 'text_struct': 47,
        'label_struct': 48, 'No_pretrain': 50, 'label_gat': 51, 'label_flat': 52,
    }

    # ===== 数据预处理: 列名适配 =====
    data_file = config.training.data_file
    temp_file = None

    if dataset_name == 'taiwan':
        if os.path.exists(data_file):
            df = pd.read_excel(data_file)
            rename_map = {}
            if 'Complaint_label' in df.columns and 'Complaint label' not in df.columns:
                rename_map['Complaint_label'] = 'Complaint label'
            if 'satisfaction_binary' in df.columns and 'Repeat complaint' not in df.columns:
                rename_map['satisfaction_binary'] = 'Repeat complaint'
            if rename_map:
                df = df.rename(columns=rename_map)
                temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
                df.to_excel(temp_file.name, index=False)
                data_file = temp_file.name
                print(f"  台湾数据集列名适配: {rename_map}")

    elif dataset_name == 'nhtsa':
        if os.path.exists(data_file):
            df = pd.read_excel(data_file)
            rename_map = {}
            if 'disputed' in df.columns and 'Repeat complaint' not in df.columns:
                rename_map['disputed'] = 'Repeat complaint'
            if rename_map:
                df = df.rename(columns=rename_map)
                temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
                df.to_excel(temp_file.name, index=False)
                data_file = temp_file.name
                print(f"  NHTSA数据集列名适配: {rename_map}")

    for exp_name, mode in experiments:
        print(f"\n运行实验: {exp_name}")
        print("-" * 40)
        seed = experiment_seeds.get(exp_name, 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"随机种子: {seed}")

        use_No_pretrain = False

        if exp_name == 'No_pretrain':
            current_pretrained_path = None
            use_No_pretrain = True
            print("[对照组] 完全随机初始化BERT")
        else:
            current_pretrained_path = pretrained_path
            if current_pretrained_path and os.path.exists(current_pretrained_path):
                print(f"[实验组] 使用预训练模型: {current_pretrained_path}")
            else:
                print("[警告] 预训练模型路径不存在,将使用原始BERT")

        # 创建processor并加载预训练的词汇表
        processor = ComplaintDataProcessor(
            config=config,
            user_dict_file=config.data.user_dict_file
        )

        processor_loaded = False
        if current_pretrained_path:
            parent_dir = os.path.dirname(current_pretrained_path)
            processor_path = os.path.join(parent_dir, 'processor.pkl')
            if os.path.exists(processor_path):
                print(f"从processor.pkl加载词汇表: {processor_path}")
                processor.load(processor_path)
                processor_loaded = True

        if not processor_loaded and current_pretrained_path:
            vocab_path = os.path.join(current_pretrained_path, 'global_vocab.pkl')
            if os.path.exists(vocab_path):
                print(f"从global_vocab.pkl加载词汇表: {vocab_path}")
                processor.load_global_vocab(vocab_path)
                processor_loaded = True

        if not processor_loaded:
            print("未找到预训练词汇表,将使用当前数据构建")

        print(f"\n加载数据: {data_file}")

        data = processor.prepare_datasets(
            train_file=data_file,
            for_pretrain=False
        )

        print(f"数据集大小: {len(data['targets'])}")
        print(f"标签词汇表大小: {data['vocab_size']}")

        # 划分数据集
        total_size = len(data['targets'])
        torch.manual_seed(seed)
        indices = torch.randperm(total_size).tolist()
        train_size = int(total_size * 0.6)
        val_size = int(total_size * 0.2)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]

        print(f"训练集: {len(train_indices)}, 验证集: {len(val_indices)}, 测试集: {len(test_indices)}")

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

        train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size,
                                  shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size,
                                collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size,
                                 collate_fn=custom_collate_fn)

        use_flat = (exp_name == 'label_flat')
        model = MultiModalComplaintModel(
            config=config, vocab_size=data['vocab_size'],
            mode=mode, pretrained_path=current_pretrained_path,
            No_pretrain_bert=use_No_pretrain, use_flat_label=use_flat
        )
        model = model.to(config.training.device)
        print(f"模型已移至设备: {config.training.device}")

        accuracy, precision, recall, f1, auc = train_and_evaluate(
            model, train_loader, val_loader, test_loader,
            config, config.training.device, exp_name
        )

        results[exp_name] = {
            'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'f1': f1, 'auc': auc
        }

        print(f"\n{exp_name} 结果:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")

        # 保存消融实验模型 (供error_analysis.py等后续脚本调用)
        if dataset_name == 'default':
            os.makedirs('./models', exist_ok=True)
            _save_path = f'./models/best_{exp_name}_model.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'vocab_size': data['vocab_size'],
                'exp_name': exp_name,
            }, _save_path)
            print(f"  💾 模型已保存: {_save_path}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 清理临时文件
    if temp_file is not None:
        try:
            os.unlink(temp_file.name)
        except Exception:
            pass

    # 打印汇总结果
    print("\n" + "="*60)
    print(f"消融实验结果汇总 ({dataset_name})")
    print("="*60)
    print(f"\n{'实验名称':<15} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 70)

    for exp_name in [e[0] for e in experiments]:
        if exp_name in results:
            r = results[exp_name]
            print(
                f"{exp_name:<15} {r['accuracy']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f} {r['auc']:<10.4f}")

    # 保存结果
    save_dir = f'./outputs/baseline_comparison/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)

    result_file = f'{save_dir}/ablation_results_{dataset_name}.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # 保存到Excel和CSV
    df_data = {
        name: {k: round(v, 4) for k, v in metrics.items() if k in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
        for name, metrics in results.items()}
    df = pd.DataFrame(df_data).T
    df.index.name = 'Experiment'
    df = df.reset_index()
    df.to_excel(f'{save_dir}/ablation_results_{dataset_name}.xlsx', index=False)
    df.to_csv(f'{save_dir}/ablation_results_{dataset_name}.csv', index=False)

    print(f"\n结果已保存到: {save_dir}/")
    return results


if __name__ == "__main__":
    import argparse
    import importlib
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str,
                       default='./pretrained_complaint_bert_improved/stage2',
                       help='预训练模型路径')
    parser.add_argument('--dataset', type=str, default='telecom',
                       choices=['telecom', 'taiwan', 'nhtsa', 'all'],
                       help='数据集选择')
    args = parser.parse_args()

    if args.dataset == 'telecom' or args.dataset == 'all':
        print("\n" + "=" * 70)
        print("自有电信数据集消融实验 (三模态)")
        print("=" * 70)
        run_ablation_study(args.pretrained_path, config=None, dataset_name='default')

    if args.dataset == 'taiwan' or args.dataset == 'all':
        print("\n" + "=" * 70)
        print("台湾餐厅数据集消融实验 → 调用 standalone")
        print("=" * 70)
        taiwan_mod = importlib.import_module('run_taiwan_restaurant_standalone')
        taiwan_mod.run_taiwan_ablation()

    if args.dataset == 'nhtsa' or args.dataset == 'all':
        print("\n" + "=" * 70)
        print("NHTSA车辆投诉数据集消融实验 → 调用 standalone")
        print("=" * 70)
        nhtsa_mod = importlib.import_module('run_nhtsa_standalone')
        nhtsa_mod.run_nhtsa_ablation()