"""
预训练任务 - 支持Text和Label预训练
修复版：解决 get_special_tokens_mask 的 TypeError
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import random


class TextPretrainDataset(Dataset):
    """Text预训练数据集 - 阶段1 MLM"""

    def __init__(self, texts, tokenizer, max_length=128,
                 mask_prob=0.15, use_span_masking=True, span_length=3):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.use_span_masking = use_span_masking
        self.span_length = span_length

    def __len__(self):
        return len(self.texts)

    def create_mlm_labels(self, input_ids):
        """创建MLM标签 - 修复版"""
        labels = input_ids.clone()

        # ✅ 修复：直接传入整个序列，而不是遍历单个元素
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            input_ids.tolist(),
            already_has_special_tokens=True
        )
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        # 创建mask概率矩阵
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        if self.use_span_masking:
            # Span Masking策略 - 修复版
            masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
            seq_len = len(input_ids)

            # 计算需要mask的span数量
            num_spans = max(1, int(seq_len * self.mask_prob / self.span_length))

            for _ in range(num_spans):
                # 确保不会mask特殊token（首尾的CLS和SEP）
                if seq_len > self.span_length + 2:
                    # 从1开始，到seq_len-span_length-1结束，避免越界
                    max_start = seq_len - self.span_length - 1
                    if max_start > 1:
                        start = random.randint(1, max_start)
                        masked_indices[start:start + self.span_length] = True
        else:
            # 随机mask
            masked_indices = torch.bernoulli(probability_matrix).bool()

        # 不mask特殊token
        masked_indices &= ~special_tokens_mask

        # 80% mask, 10% random, 10% keep
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # 未mask的位置label设为-100
        labels[~masked_indices] = -100

        return input_ids, labels

    def __getitem__(self, idx):
        """获取一个训练样本 - 修复版"""
        # ⭐ 确保texts是字符串，不是tensor
        text = str(self.texts[idx]) if idx < len(self.texts) else ""


        # 文本编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # 创建MLM标签
        mlm_input_ids, mlm_labels = self.create_mlm_labels(input_ids.clone())

        # 对齐标签（用于对比学习）
        # 这里简化处理：随机决定是否对齐
        is_aligned = torch.tensor(random.random() > 0.5, dtype=torch.float32)

        return {
            'input_ids': mlm_input_ids,
            'attention_mask': attention_mask,
            'mlm_labels': mlm_labels,
            'is_aligned': is_aligned
        }


class ContrastivePretrainDataset(Dataset):
    """对比学习预训练数据集 - 阶段2"""

    def __init__(self, texts, labels, tokenizer, processor,
                 max_length=128, mask_prob=0.15, use_span_masking=True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.use_span_masking = use_span_masking

    def __len__(self):
        return len(self.texts)

    def create_mlm_labels(self, input_ids):
        """创建MLM标签 - 修复版（同上）"""
        labels = input_ids.clone()

        # ✅ 修复：直接传入整个序列
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            input_ids.tolist(),
            already_has_special_tokens=True
        )
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices &= ~special_tokens_mask

        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        labels[~masked_indices] = -100

        return input_ids, labels

    def __getitem__(self, idx):
        text = str(self.texts[idx]) if idx < len(self.texts) else ""

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # 创建MLM标签
        mlm_input_ids, mlm_labels = self.create_mlm_labels(input_ids.clone())

        # 对齐标签（用于对比学习）
        is_aligned = torch.tensor(random.random() > 0.5, dtype=torch.float32)

        return {
            'input_ids': mlm_input_ids,
            'attention_mask': attention_mask,
            'mlm_labels': mlm_labels,
            'is_aligned': is_aligned
        }