"""
推理脚本 - inference.py
✅ 修复：模型初始化参数匹配问题
"""

import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from typing import Dict, List, Any, Optional
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# 使用正确的引用
from model import MultiModalComplaintModel
from data_processor import ComplaintDataProcessor
from config import Config


class ComplaintPredictor:
    """重复投诉预测器"""

    def __init__(self, model_path: str, pretrained_path: str = None):
        """
        初始化预测器

        Args:
            model_path: 模型权重文件路径
            pretrained_path: 预训练模型路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ✅ 使用Config对象
        self.config = Config()

        # 加载模型
        print(f"正在加载模型: {model_path}")

        # 如果没有指定预训练路径，使用默认的
        if pretrained_path is None:
            pretrained_path = self.config.training.pretrain_save_dir

        # 加载处理器
        self.processor = ComplaintDataProcessor(
            config=self.config,
            user_dict_file=self.config.data.user_dict_file
        )

        processor_path = os.path.join(pretrained_path, 'processor.pkl') if pretrained_path else None
        if processor_path and os.path.exists(processor_path):
            print(f"加载处理器: {processor_path}")
            self.processor.load(processor_path)

        # 获取词汇表大小
        vocab_size = len(self.processor.node_to_id) if self.processor.node_to_id else 1000

        # ✅ 从checkpoint恢复训练config（修复维度不匹配问题）
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            ckpt_config = checkpoint.get('config', None)
            if ckpt_config is not None and hasattr(ckpt_config, 'model'):
                self.config = ckpt_config
                print(f"  ✅ 从checkpoint恢复训练config")
            vocab_size_from_ckpt = checkpoint.get('vocab_size', None)
            if vocab_size_from_ckpt is not None:
                vocab_size = vocab_size_from_ckpt
        else:
            checkpoint = None

        self.model = MultiModalComplaintModel(
            config=self.config,
            vocab_size=vocab_size,
            mode='full',
            pretrained_path=pretrained_path
        )

        # 加载模型权重（如果存在）
        if checkpoint is not None:
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            # 检查并扩展BERT词表
            for k, v in state_dict.items():
                if 'text_encoder.embeddings.word_embeddings.weight' in k:
                    bert_vocab_size = v.shape[0]
                    if self.model.text_encoder is not None:
                        current_size = self.model.text_encoder.embeddings.word_embeddings.weight.shape[0]
                        if bert_vocab_size > current_size:
                            self.model.text_encoder.resize_token_embeddings(bert_vocab_size)
                            print(f"  📝 扩展BERT词表: {current_size} → {bert_vocab_size}")
                    break
            self.model.load_state_dict(state_dict, strict=False)
            print("模型权重加载成功")

        self.model.to(self.device)
        self.model.eval()

        print(f"预测器初始化完成，使用设备: {self.device}")

    def preprocess_single_sample(self, complaint_text: str, complaint_label: str,
                                structured_features: List[float]) -> Dict[str, torch.Tensor]:
        """预处理单个样本"""
        # 文本处理
        from transformers import BertTokenizer

        # 加载tokenizer
        if hasattr(self.processor, 'tokenizer'):
            tokenizer = self.processor.tokenizer
        else:
            tokenizer = BertTokenizer.from_pretrained(self.config.model.bert_model_name)

        # 文本编码
        text = complaint_text if complaint_text else ""
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.model.bert_max_length,
            return_tensors='pt'
        )

        # 标签处理
        if complaint_label and self.processor.node_to_id:
            node_ids, edges, node_levels = self.processor.encode_label_path_as_graph(complaint_label)
        else:
            node_ids, edges, node_levels = [0], [], [0]

        # 结构化特征处理
        if structured_features is None or len(structured_features) == 0:
            structured_features = np.zeros(53)
        else:
            structured_features = np.array(structured_features, dtype=np.float32)

        if len(structured_features.shape) == 1:
            structured_features = structured_features.reshape(1, -1)

        # 标准化
        if hasattr(self.processor, 'scaler') and self.processor.scaler:
            try:
                structured_features = self.processor.scaler.transform(structured_features)
            except:
                pass

        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device),
            'node_ids': node_ids,
            'edges': edges,
            'node_levels': node_levels,
            'struct_features': torch.tensor(structured_features[0], dtype=torch.float).to(self.device)
        }

    def predict_single(self, complaint_text: str, complaint_label: str,
                       structured_features: List[float] = None) -> Dict[str, Any]:
        """预测单个样本"""
        # 预处理
        inputs = self.preprocess_single_sample(complaint_text, complaint_label, structured_features)

        # 预测
        with torch.no_grad():
            logits, _ = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                node_ids_list=[inputs['node_ids']],
                edges_list=[inputs['edges']],
                node_levels_list=[inputs['node_levels']],
                struct_features=inputs['struct_features'].unsqueeze(0)
            )

            # 计算概率
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    'non_repeat': probabilities[0][0].item(),
                    'repeat': probabilities[0][1].item()
                },
                'prediction_text': '重复投诉' if predicted_class == 1 else '非重复投诉'
            }

            return result

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """从DataFrame预测"""
        results = []

        for idx, row in df.iterrows():
            # 提取特征
            complaint_text = row.get('biz_cntt', '')
            complaint_label = row.get('Complaint label', '')

            # 结构化特征 - 修复版本
            col_names = df.columns.tolist()
            exclude_cols = {'new_code', 'biz_cntt', 'Complaint label', 'Repeat complaint'}

            # 精确定位结构化特征列
            if 'Complaint label' in col_names and 'Repeat complaint' in col_names:
                label_idx = col_names.index('Complaint label')
                target_idx = col_names.index('Repeat complaint')
                struct_columns = col_names[label_idx + 1: target_idx]
            else:
                struct_columns = col_names[3:56]

            # 排除非特征列
            struct_columns = [col for col in struct_columns if col not in exclude_cols][:53]

            # 提取特征值
            structured_features = []
            for col in struct_columns:
                try:
                    value = pd.to_numeric(row.get(col, 0), errors='coerce')
                    value = 0 if pd.isna(value) else value
                    structured_features.append(value)
                except:
                    structured_features.append(0)

            # 确保正好53个特征
            while len(structured_features) < 53:
                structured_features.append(0)
            structured_features = structured_features[:53]

            # 预测
            result = self.predict_single(
                complaint_text, complaint_label, structured_features
            )
            results.append(result)

        # 将结果添加到DataFrame
        result_df = df.copy()
        result_df['predicted_class'] = [r['predicted_class'] for r in results]
        result_df['confidence'] = [r['confidence'] for r in results]
        result_df['prediction_text'] = [r['prediction_text'] for r in results]
        result_df['repeat_probability'] = [r['probabilities']['repeat'] for r in results]

        return result_df

    def analyze_predictions(self, result_df: pd.DataFrame):
        """分析预测结果"""
        print("\n" + "=" * 60)
        print("预测结果分析")
        print("=" * 60)

        # 基本统计
        total_samples = len(result_df)
        repeat_predictions = (result_df['predicted_class'] == 1).sum()
        non_repeat_predictions = (result_df['predicted_class'] == 0).sum()

        print(f"总样本数: {total_samples}")
        print(f"预测为重复投诉: {repeat_predictions} ({repeat_predictions / total_samples * 100:.1f}%)")
        print(f"预测为非重复投诉: {non_repeat_predictions} ({non_repeat_predictions / total_samples * 100:.1f}%)")

        # 置信度分析
        avg_confidence = result_df['confidence'].mean()
        low_confidence = (result_df['confidence'] < 0.7).sum()

        print(f"\n平均置信度: {avg_confidence:.3f}")
        print(f"低置信度预测数(<0.7): {low_confidence} ({low_confidence / total_samples * 100:.1f}%)")

        # 如果有真实标签，计算准确率
        if 'Repeat complaint' in result_df.columns:
            correct = (result_df['predicted_class'] == result_df['Repeat complaint']).sum()
            accuracy = correct / total_samples
            print(f"\n预测准确率: {accuracy:.3f}")


def main():
    """推理示例"""
    import argparse

    parser = argparse.ArgumentParser(description='投诉预测推理')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='预训练模型路径')
    parser.add_argument('--data_file', type=str, default='小案例ai问询.xlsx',
                        help='数据文件')
    args = parser.parse_args()

    # 自动搜索已保存的模型
    if args.model_path is None:
        _auto_paths = [
            './outputs/baseline_comparison/default/tmcrpp_models/tmcrpp_default.pth',
            './outputs/cross_dataset/models/telecom_full_model.pth',
            './models/best_model.pth',
        ]
        for _p in _auto_paths:
            if os.path.exists(_p):
                args.model_path = _p
                break
        if args.model_path is None:
            args.model_path = './models/best_model.pth'

    # 检查文件
    if not os.path.exists(args.data_file):
        print(f"数据文件不存在: {args.data_file}")
        return

    # 创建预测器
    try:
        predictor = ComplaintPredictor(args.model_path, args.pretrained_path)

        # 示例1：单个样本预测
        print("\n" + "=" * 60)
        print("示例1：单个样本预测")
        print("=" * 60)

        result = predictor.predict_single(
            complaint_text="客户反映移动信号不好，多次投诉未解决，要求尽快处理",
            complaint_label="移动业务→网络问题→信号覆盖→室内信号差",
            structured_features=[0.8, 0.6, 0.9, 0.7] + [0] * 49  # 示例特征
        )

        print(f"预测结果: {result['prediction_text']}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"重复投诉概率: {result['probabilities']['repeat']:.4f}")

        # 示例2：批量预测
        print("\n" + "=" * 60)
        print("示例2：批量预测")
        print("=" * 60)

        df = pd.read_excel(args.data_file)

        # 只预测前10行
        sample_df = df.head(10)
        result_df = predictor.predict_from_dataframe(sample_df)

        print("\n预测结果:")
        display_columns = ['biz_cntt', 'prediction_text', 'confidence', 'repeat_probability']
        available_columns = [col for col in display_columns if col in result_df.columns]

        # 限制文本长度
        if 'biz_cntt' in result_df.columns:
            result_df['biz_cntt_display'] = result_df['biz_cntt'].astype(str).apply(
                lambda x: x[:50] + '...' if len(x) > 50 else x
            )
            available_columns[0] = 'biz_cntt_display'

        print(result_df[available_columns].to_string(index=False))

        # 分析预测结果
        predictor.analyze_predictions(result_df)

        # 保存结果
        output_file = "prediction_results.xlsx"
        result_df.to_excel(output_file, index=False)
        print(f"\n完整预测结果已保存到: {output_file}")

    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()