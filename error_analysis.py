"""
Error Correction Analysis - error_analysis.py
Quantitatively demonstrates how semi-structured labels correct text-only model errors.

Core Hypothesis: "Label as Anchor"
When text is emotionally charged or semantically ambiguous, semi-structured labels
act as noise filters to provide stable prediction anchors.

This script:
1. Finds samples where Text-Only model predicts wrong but Full model predicts correct
2. Analyzes the characteristics of these corrected samples
3. Generates case studies for qualitative discussion in the paper

Usage:
    python error_analysis.py --data_file data.xlsx --text_model text_model.pth --full_model full_model.pth
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MultiModalComplaintModel
from data_processor import ComplaintDataProcessor
from config import Config
from transformers import BertTokenizer


class ErrorCorrectionAnalyzer:
    """
    Analyzes cases where the full model corrects errors made by text-only model.
    
    Key Analysis:
    - Condition A: Text-Only Model predicts wrong (Pred != Truth)
    - Condition B: Full Model (Text+Label+Struct) predicts correct (Pred == Truth)
    
    These are samples where adding labels/struct helped correct the prediction.
    """
    
    def __init__(self,
                 config: Config = None,
                 device: str = None):
        """
        Initialize analyzer
        
        Args:
            config: Configuration object
            device: Device to use
        """
        self.config = config or Config()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Update device in config
        self.config.training.device = self.device
        
        # Initialize processor with correct parameters
        user_dict_file = self.config.data.user_dict_file if hasattr(self.config.data, 'user_dict_file') else 'new_user_dict.txt'
        self.processor = ComplaintDataProcessor(
            config=self.config,
            user_dict_file=user_dict_file
        )
        
        # Try to load pretrained processor
        pretrained_path = self.config.training.pretrain_save_dir
        processor_path = os.path.join(pretrained_path, 'processor.pkl') if pretrained_path else None
        if processor_path and os.path.exists(processor_path):
            try:
                self.processor.load(processor_path)
                print(f"✅ Loaded processor from {processor_path}")
            except Exception as e:
                print(f"⚠️ Could not load processor: {e}")
        
        # Get vocabulary size
        self.vocab_size = len(self.processor.node_to_id) if self.processor.node_to_id else 1000
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.config.model.bert_model_name)
        
        # Create output directory
        os.makedirs('./outputs/error_analysis', exist_ok=True)
        
        print(f"✅ ErrorCorrectionAnalyzer initialized on {self.device}")
    
    def load_models(self,
                    text_only_path: str = None,
                    full_model_path: str = None,
                    pretrained_path: str = None):
        """
        Load text-only and full models
        
        Args:
            text_only_path: Path to text-only model weights
            full_model_path: Path to full model weights
            pretrained_path: Path to pretrained weights
        """
        pretrained_path = pretrained_path or self.config.training.pretrain_save_dir

        # Load text-only model
        print("📦 Loading Text-Only model...")
        # ✅ 从checkpoint恢复训练config（修复维度不匹配问题）
        text_config = self.config
        text_vocab = self.vocab_size
        if text_only_path and os.path.exists(text_only_path):
            checkpoint_text = torch.load(text_only_path, map_location=self.device)
            ckpt_cfg = checkpoint_text.get('config', None)
            if ckpt_cfg is not None and hasattr(ckpt_cfg, 'model'):
                text_config = ckpt_cfg
                print(f"  ✅ 从checkpoint恢复训练config (text_only)")
            v_size = checkpoint_text.get('vocab_size', None)
            if v_size is not None:
                text_vocab = v_size
        else:
            checkpoint_text = None

        self.text_only_model = MultiModalComplaintModel(
            config=text_config,
            vocab_size=text_vocab,
            mode='text_only',
            pretrained_path=pretrained_path
        )

        if checkpoint_text is not None:
            state_dict = checkpoint_text.get('model_state_dict', checkpoint_text)
            # 检查并扩展BERT词表
            for k, v in state_dict.items():
                if 'text_encoder.embeddings.word_embeddings.weight' in k:
                    bert_vocab_size = v.shape[0]
                    if self.text_only_model.text_encoder is not None:
                        current_size = self.text_only_model.text_encoder.embeddings.word_embeddings.weight.shape[0]
                        if bert_vocab_size > current_size:
                            self.text_only_model.text_encoder.resize_token_embeddings(bert_vocab_size)
                            print(f"  📝 扩展BERT词表: {current_size} → {bert_vocab_size}")
                    break
            self.text_only_model.load_state_dict(state_dict, strict=False)
            print(f"  ✅ Loaded weights from {text_only_path}")
        else:
            print(f"  ⚠️ No weights found, using random initialization")
        
        self.text_only_model.to(self.device)
        self.text_only_model.eval()

        # Load full model
        print("📦 Loading Full model...")
        # ✅ 从checkpoint恢复训练config（修复维度不匹配问题）
        full_config = self.config
        full_vocab = self.vocab_size
        if full_model_path and os.path.exists(full_model_path):
            checkpoint_full = torch.load(full_model_path, map_location=self.device)
            ckpt_cfg = checkpoint_full.get('config', None)
            if ckpt_cfg is not None and hasattr(ckpt_cfg, 'model'):
                full_config = ckpt_cfg
                print(f"  ✅ 从checkpoint恢复训练config (full)")
            v_size = checkpoint_full.get('vocab_size', None)
            if v_size is not None:
                full_vocab = v_size
        else:
            checkpoint_full = None

        self.full_model = MultiModalComplaintModel(
            config=full_config,
            vocab_size=full_vocab,
            mode='full',
            pretrained_path=pretrained_path
        )

        if checkpoint_full is not None:
            state_dict = checkpoint_full.get('model_state_dict', checkpoint_full)
            # 检查并扩展BERT词表
            for k, v in state_dict.items():
                if 'text_encoder.embeddings.word_embeddings.weight' in k:
                    bert_vocab_size = v.shape[0]
                    if self.full_model.text_encoder is not None:
                        current_size = self.full_model.text_encoder.embeddings.word_embeddings.weight.shape[0]
                        if bert_vocab_size > current_size:
                            self.full_model.text_encoder.resize_token_embeddings(bert_vocab_size)
                            print(f"  📝 扩展BERT词表: {current_size} → {bert_vocab_size}")
                    break
            self.full_model.load_state_dict(state_dict, strict=False)
            print(f"  ✅ Loaded weights from {full_model_path}")
        else:
            print(f"  ⚠️ No weights found, using random initialization")
        
        self.full_model.to(self.device)
        self.full_model.eval()
    
    def predict_single(self,
                       model: nn.Module,
                       text: str,
                       label: str = None,
                       struct_features: np.ndarray = None,
                       mode: str = 'full') -> Tuple[int, float]:
        """
        Make prediction for a single sample
        
        Returns:
            prediction: Predicted class (0 or 1)
            confidence: Prediction confidence
        """
        # Text encoding
        text_encoding = self.tokenizer(
            text,
            max_length=self.config.model.bert_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = text_encoding['input_ids'].to(self.device)
        attention_mask = text_encoding['attention_mask'].to(self.device)
        
        # Label encoding (if needed)
        node_ids_list = None
        edges_list = None
        node_levels_list = None
        
        if mode == 'full' and label:
            node_ids, edges, levels = self.processor.encode_label_path_as_graph(label)
            node_ids_list = [node_ids]
            edges_list = [edges]
            node_levels_list = [levels]
        
        # Struct features (if needed)
        struct_tensor = None
        if mode == 'full' and struct_features is not None:
            struct_tensor = torch.tensor(struct_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            if mode == 'text_only':
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_attention=False
                )
            else:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_ids_list=node_ids_list,
                    edges_list=edges_list,
                    node_levels_list=node_levels_list,
                    struct_features=struct_tensor,
                    return_attention=False
                )
            
            # Handle different return types
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            confidence = probs[0, pred].item()
        
        return pred, confidence
    
    def find_corrected_samples(self,
                               data_file: str,
                               max_samples: int = None) -> List[Dict]:
        """
        Find samples where Full model corrects Text-Only model's errors
        
        Conditions:
        - Condition A: Text-Only model predicts wrong
        - Condition B: Full model predicts correct
        
        Args:
            data_file: Path to data file
            max_samples: Maximum number of samples to process
            
        Returns:
            List of corrected samples with metadata
        """
        # Load data
        print(f"\n📂 Loading data: {data_file}")
        if data_file.endswith('.xlsx'):
            df = pd.read_excel(data_file)
        else:
            df = pd.read_csv(data_file)
        
        # Get structured feature columns
        col_names = df.columns.tolist()
        if 'Complaint label' in col_names and 'Repeat complaint' in col_names:
            label_idx = col_names.index('Complaint label')
            target_idx = col_names.index('Repeat complaint')
            struct_cols = col_names[label_idx + 1: target_idx]
        else:
            struct_cols = col_names[3:56]
        
        struct_cols = struct_cols[:53]
        
        # Process samples
        corrected_samples = []
        text_only_correct = 0
        full_correct = 0
        both_wrong = 0
        both_correct = 0
        
        n_samples = len(df) if max_samples is None else min(max_samples, len(df))
        
        print(f"\n🔍 Analyzing {n_samples} samples...")
        
        for idx in tqdm(range(n_samples), desc="Processing"):
            row = df.iloc[idx]
            
            # Extract data
            text = str(row.get('biz_cntt', row.get('text', '')))
            label = str(row.get('Complaint label', row.get('label', '')))
            true_label = int(row.get('Repeat complaint', row.get('target', 0)))
            struct_features = row[struct_cols].fillna(0).values.astype(np.float32)
            
            # Text-only prediction
            text_pred, text_conf = self.predict_single(
                self.text_only_model, text, mode='text_only'
            )
            
            # Full model prediction
            full_pred, full_conf = self.predict_single(
                self.full_model, text, label, struct_features, mode='full'
            )
            
            # Categorize
            text_correct = (text_pred == true_label)
            full_correct_flag = (full_pred == true_label)
            
            if text_correct and full_correct_flag:
                both_correct += 1
            elif not text_correct and full_correct_flag:
                # CORRECTED: Text wrong, Full correct
                corrected_samples.append({
                    'idx': idx,
                    'text': text,
                    'label': label,
                    'struct_features': struct_features.tolist(),
                    'true_label': true_label,
                    'text_pred': text_pred,
                    'text_conf': text_conf,
                    'full_pred': full_pred,
                    'full_conf': full_conf,
                    'new_code': str(row.get('new_code', f'sample_{idx}'))
                })
            elif text_correct and not full_correct_flag:
                text_only_correct += 1
            else:
                both_wrong += 1
        
        # Statistics
        stats = {
            'total_samples': n_samples,
            'both_correct': both_correct,
            'text_only_correct': text_only_correct,
            'corrected_by_full': len(corrected_samples),
            'both_wrong': both_wrong,
            'text_accuracy': (both_correct + text_only_correct) / n_samples,
            'full_accuracy': (both_correct + len(corrected_samples)) / n_samples,
            'correction_rate': len(corrected_samples) / max(1, n_samples - both_correct)
        }
        
        print(f"\n📊 Analysis Results:")
        print(f"  - Total samples: {stats['total_samples']}")
        print(f"  - Both correct: {stats['both_correct']} ({stats['both_correct']/n_samples*100:.1f}%)")
        print(f"  - Text-only correct (Full wrong): {stats['text_only_correct']} ({stats['text_only_correct']/n_samples*100:.1f}%)")
        print(f"  - Corrected by Full model: {stats['corrected_by_full']} ({stats['corrected_by_full']/n_samples*100:.1f}%)")
        print(f"  - Both wrong: {stats['both_wrong']} ({stats['both_wrong']/n_samples*100:.1f}%)")
        print(f"  - Text-Only Accuracy: {stats['text_accuracy']*100:.2f}%")
        print(f"  - Full Model Accuracy: {stats['full_accuracy']*100:.2f}%")
        print(f"  - Error Correction Rate: {stats['correction_rate']*100:.2f}%")
        
        self.stats = stats
        
        return corrected_samples
    
    def analyze_correction_patterns(self, corrected_samples: List[Dict]) -> Dict:
        """
        Analyze patterns in corrected samples
        
        Examines:
        - Text characteristics (length, emotional words)
        - Label characteristics
        - Confidence differences
        """
        if not corrected_samples:
            print("⚠️ No corrected samples to analyze")
            return {}
        
        analysis = {
            'text_length_stats': {},
            'label_distribution': {},
            'confidence_improvement': [],
            'common_patterns': []
        }
        
        # Text length analysis
        text_lengths = [len(s['text']) for s in corrected_samples]
        analysis['text_length_stats'] = {
            'mean': np.mean(text_lengths),
            'std': np.std(text_lengths),
            'min': np.min(text_lengths),
            'max': np.max(text_lengths),
            'median': np.median(text_lengths)
        }
        
        # Label distribution
        labels = [s['label'] for s in corrected_samples]
        from collections import Counter
        label_counts = Counter(labels)
        analysis['label_distribution'] = dict(label_counts.most_common(10))
        
        # Confidence improvement
        for s in corrected_samples:
            analysis['confidence_improvement'].append({
                'text_conf': s['text_conf'],
                'full_conf': s['full_conf'],
                'improvement': s['full_conf'] - s['text_conf']
            })
        
        avg_improvement = np.mean([c['improvement'] for c in analysis['confidence_improvement']])
        analysis['avg_confidence_improvement'] = avg_improvement
        
        print(f"\n📈 Correction Pattern Analysis:")
        print(f"  - Avg text length: {analysis['text_length_stats']['mean']:.1f} chars")
        print(f"  - Top labels: {list(analysis['label_distribution'].keys())[:3]}")
        print(f"  - Avg confidence improvement: {avg_improvement:.3f}")
        
        return analysis
    
    def generate_case_studies(self,
                              corrected_samples: List[Dict],
                              n_cases: int = 5) -> List[Dict]:
        """
        Generate case studies for paper discussion
        
        Selects representative cases showing:
        1. Short/ambiguous text that label helps clarify
        2. Emotional text where label provides stability
        3. Clear correction with high confidence improvement
        
        Args:
            corrected_samples: List of corrected samples
            n_cases: Number of cases to generate
            
        Returns:
            List of case study dictionaries
        """
        if not corrected_samples:
            print("⚠️ No corrected samples for case studies")
            return []
        
        # Sort by confidence improvement
        sorted_samples = sorted(
            corrected_samples,
            key=lambda x: x['full_conf'] - x['text_conf'],
            reverse=True
        )
        
        case_studies = []
        
        print(f"\n📝 Generating {min(n_cases, len(sorted_samples))} Case Studies...")
        print("="*60)
        
        for i, sample in enumerate(sorted_samples[:n_cases]):
            case = {
                'case_id': i + 1,
                'text': sample['text'][:200] + ('...' if len(sample['text']) > 200 else ''),
                'label': sample['label'],
                'true_label': 'Repeat' if sample['true_label'] == 1 else 'Non-repeat',
                'text_prediction': 'Repeat' if sample['text_pred'] == 1 else 'Non-repeat',
                'full_prediction': 'Repeat' if sample['full_pred'] == 1 else 'Non-repeat',
                'text_confidence': sample['text_conf'],
                'full_confidence': sample['full_conf'],
                'confidence_improvement': sample['full_conf'] - sample['text_conf'],
                'analysis': self._analyze_single_case(sample)
            }
            case_studies.append(case)
            
            # Print case
            print(f"\n--- Case {i+1} ---")
            print(f"Text: {case['text']}")
            print(f"Label: {case['label']}")
            print(f"True Label: {case['true_label']}")
            print(f"Text-Only Pred: {case['text_prediction']} (conf: {case['text_confidence']:.3f}) ❌")
            print(f"Full Model Pred: {case['full_prediction']} (conf: {case['full_confidence']:.3f}) ✅")
            print(f"Analysis: {case['analysis']}")
        
        print("="*60)
        
        return case_studies
    
    def _analyze_single_case(self, sample: Dict) -> str:
        """Generate analysis text for a single case"""
        text_len = len(sample['text'])
        conf_improvement = sample['full_conf'] - sample['text_conf']
        
        analysis_parts = []
        
        # Text length analysis
        if text_len < 20:
            analysis_parts.append("Text is very short and ambiguous")
        elif text_len < 50:
            analysis_parts.append("Text is brief with limited context")
        
        # Confidence analysis
        if conf_improvement > 0.3:
            analysis_parts.append("Large confidence improvement with label context")
        elif conf_improvement > 0.1:
            analysis_parts.append("Moderate confidence improvement")
        
        # Label contribution
        if sample['label']:
            analysis_parts.append(f"Label '{sample['label'][:30]}' provides semantic anchor")
        
        return "; ".join(analysis_parts) if analysis_parts else "Label provided disambiguation"
    
    def save_analysis_results(self,
                              corrected_samples: List[Dict],
                              analysis: Dict,
                              case_studies: List[Dict],
                              save_dir: str = './outputs/error_analysis'):
        """Save all analysis results"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save corrected samples
        samples_path = os.path.join(save_dir, 'corrected_samples.json')
        with open(samples_path, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_samples = []
            for s in corrected_samples:
                s_copy = s.copy()
                if isinstance(s_copy.get('struct_features'), np.ndarray):
                    s_copy['struct_features'] = s_copy['struct_features'].tolist()
                serializable_samples.append(s_copy)
            json.dump(serializable_samples, f, ensure_ascii=False, indent=2)
        
        # Save statistics
        stats_path = os.path.join(save_dir, 'correction_statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        # Save analysis
        analysis_path = os.path.join(save_dir, 'pattern_analysis.json')
        with open(analysis_path, 'w', encoding='utf-8') as f:
            # Convert numpy values
            analysis_serializable = {}
            for k, v in analysis.items():
                if isinstance(v, dict):
                    analysis_serializable[k] = {
                        kk: float(vv) if isinstance(vv, np.floating) else vv
                        for kk, vv in v.items()
                    }
                elif isinstance(v, list):
                    analysis_serializable[k] = [
                        {kk: float(vv) if isinstance(vv, np.floating) else vv for kk, vv in item.items()}
                        if isinstance(item, dict) else item
                        for item in v
                    ]
                elif isinstance(v, (np.floating, float)):
                    analysis_serializable[k] = float(v)
                elif isinstance(v, (np.integer, int)):
                    analysis_serializable[k] = int(v)
                else:
                    analysis_serializable[k] = v
                json.dump(analysis_serializable, f, ensure_ascii=False, indent=2,
                          default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o,
                                                                                                            np.floating) else str(
                              o))
        
        # Save case studies
        cases_path = os.path.join(save_dir, 'case_studies.json')
        with open(cases_path, 'w', encoding='utf-8') as f:
            json.dump(case_studies, f, ensure_ascii=False, indent=2)
        
        # Generate LaTeX table for case studies
        latex_path = os.path.join(save_dir, 'case_studies_table.tex')
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_case_latex(case_studies))

        print(f"\n✅ Results saved to: {save_dir}")
        print(f"  - corrected_samples.json")
        print(f"  - correction_statistics.json")
        print(f"  - pattern_analysis.json")
        print(f"  - case_studies.json")
        print(f"  - case_studies_table.tex")

        # === Auto-generate visualization and Excel ===
        viz_data = {
            'correction_stats': self.stats,
            'case_studies': case_studies,
            'analysis': analysis
        }
        try:
            viz_json_path = os.path.join(save_dir, '_temp_viz.json')
            with open(viz_json_path, 'w', encoding='utf-8') as f:
                json.dump(viz_data, f, ensure_ascii=False, indent=2, default=str)
            visualize_error_analysis(viz_json_path, save_dir)
            os.remove(viz_json_path)
        except Exception as e:
            print(f"  Warning: Visualization generation failed: {e}")

        try:
            if case_studies:
                df_cases = pd.DataFrame(case_studies)
                excel_path = os.path.join(save_dir, 'case_studies_table.xlsx')
                df_cases.to_excel(excel_path, index=False)
                print(f"  - case_studies_table.xlsx")
        except Exception as e:
            print(f"  Warning: Excel export failed: {e}")

    def _generate_case_latex(self, case_studies: List[Dict]) -> str:
        """Generate LaTeX table for case studies"""
        latex = r"""
\begin{table*}[htbp]
\centering
\caption{Error Correction Case Studies: Label as Anchor}
\label{tab:error_correction}
\begin{tabular}{p{0.25\textwidth}p{0.2\textwidth}p{0.15\textwidth}p{0.15\textwidth}p{0.15\textwidth}}
\toprule
Text (Truncated) & Label & True Label & Text-Only & Full Model \\
\midrule
"""
        for case in case_studies[:5]:
            text = case['text'][:50].replace('_', '\\_').replace('&', '\\&')
            label = case['label'][:30].replace('_', '\\_').replace('&', '\\&')
            latex += f"{text}... & {label} & {case['true_label']} & "
            latex += f"{case['text_prediction']} & {case['full_prediction']} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
        return latex
    
    def run_full_analysis(self,
                          data_file: str,
                          text_only_path: str = None,
                          full_model_path: str = None,
                          max_samples: int = None,
                          n_cases: int = 5):
        """
        Run complete error correction analysis
        
        Args:
            data_file: Path to data file
            text_only_path: Path to text-only model
            full_model_path: Path to full model
            max_samples: Maximum samples to analyze
            n_cases: Number of case studies
        """
        print("\n" + "="*60)
        print("🔬 Error Correction Analysis: Label as Anchor")
        print("="*60)
        
        # Load models
        self.load_models(text_only_path, full_model_path)
        
        # Find corrected samples
        corrected_samples = self.find_corrected_samples(data_file, max_samples)
        
        # Analyze patterns
        analysis = self.analyze_correction_patterns(corrected_samples)
        
        # Generate case studies
        case_studies = self.generate_case_studies(corrected_samples, n_cases)
        
        # Save results
        self.save_analysis_results(corrected_samples, analysis, case_studies)
        
        print("\n" + "="*60)
        print("✅ Error Correction Analysis Complete!")
        print("="*60)
        
        return {
            'statistics': self.stats,
            'corrected_samples': corrected_samples,
            'analysis': analysis,
            'case_studies': case_studies
        }


def visualize_error_analysis(error_json_path: str, save_dir: str = './outputs/figures'):
    """
    将error_analysis的JSON结果可视化

    Args:
        error_json_path: JSON结果文件路径
        save_dir: 保存目录
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(save_dir, exist_ok=True)

    # 读取JSON数据
    with open(error_json_path, 'r', encoding='utf-8') as f:
        error_data = json.load(f)

    # SCI级别配色
    COLORS = {
        'primary': '#E74C3C',
        'secondary': '#3498DB',
        'tertiary': '#2ECC71',
        'quaternary': '#9B59B6',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 错误类型分布 (饼图)
    ax1 = axes[0, 0]
    if 'correction_stats' in error_data:
        stats = error_data['correction_stats']
        labels = ['Corrected by Label', 'Remaining Errors']
        sizes = [stats.get('corrected_samples', 0),
                 stats.get('text_only_errors', 0) - stats.get('corrected_samples', 0)]
        sizes = [max(0, s) for s in sizes]
        if sum(sizes) > 0:
            colors = [COLORS['tertiary'], COLORS['primary']]
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=90, textprops={'fontsize': 10})
            ax1.set_title('Error Correction by Label Addition', fontsize=12, fontweight='bold')

    # 2. 置信度对比 (柱状图)
    ax2 = axes[0, 1]
    if 'case_studies' in error_data:
        cases = error_data['case_studies']
        text_confs = [c.get('text_confidence', 0) for c in cases if 'text_confidence' in c]
        full_confs = [c.get('full_confidence', 0) for c in cases if 'full_confidence' in c]

        if text_confs and full_confs:
            x = np.arange(min(10, len(text_confs)))
            width = 0.35
            ax2.bar(x - width / 2, text_confs[:10], width, label='Text-Only', color=COLORS['primary'], alpha=0.7)
            ax2.bar(x + width / 2, full_confs[:10], width, label='Full Model', color=COLORS['tertiary'], alpha=0.7)
            ax2.set_xlabel('Sample Index', fontsize=10)
            ax2.set_ylabel('Confidence', fontsize=10)
            ax2.set_title('Confidence Comparison', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(axis='y', alpha=0.3)

    # 3. 文本长度分布
    ax3 = axes[1, 0]
    if 'analysis' in error_data and 'text_length_distribution' in error_data['analysis']:
        length_data = error_data['analysis']['text_length_distribution']
        bins = list(length_data.keys())
        counts = list(length_data.values())
        ax3.bar(bins, counts, color=COLORS['secondary'], edgecolor='white', alpha=0.8)
        ax3.set_xlabel('Text Length Range', fontsize=10)
        ax3.set_ylabel('Correction Count', fontsize=10)
        ax3.set_title('Text Length Distribution', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)

    # 4. 案例表格
    ax4 = axes[1, 1]
    ax4.axis('off')
    if 'case_studies' in error_data and error_data['case_studies']:
        cases = error_data['case_studies'][:5]
        table_data = []
        for c in cases:
            text_snippet = c.get('text', '')[:25] + '...'
            label_snippet = c.get('label', '')[:20] + '...'
            true_label = 'Repeat' if c.get('true_label', 0) == 1 else 'Non-rep'
            text_pred = 'Repeat' if c.get('text_prediction', 0) == 1 else 'Non-rep'
            full_pred = 'Repeat' if c.get('full_prediction', 0) == 1 else 'Non-rep'
            table_data.append([text_snippet, label_snippet, true_label, text_pred, full_pred])

        if table_data:
            table = ax4.table(cellText=table_data,
                              colLabels=['Text', 'Label', 'True', 'Text-Only', 'Full'],
                              cellLoc='center', loc='center',
                              colColours=['#f0f0f0'] * 5)
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            ax4.set_title('Error Correction Case Studies', fontsize=12, fontweight='bold', y=0.95)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'error_analysis_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ Saved: {save_path}")

    # 导出为Excel表格
    if 'case_studies' in error_data:
        df = pd.DataFrame(error_data['case_studies'])
        excel_path = os.path.join(save_dir, 'error_analysis_table.xlsx')
        df.to_excel(excel_path, index=False)
        print(f"✅ Saved: {excel_path}")

    return save_path


def visualize_error_results(json_path: str, save_dir: str = './outputs/figures'):
    """
    将error_analysis的JSON结果可视化

    Args:
        json_path: JSON结果文件路径
        save_dir: 图片保存目录
    """
    import json
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        error_data = json.load(f)

    try:
        from visualization_enhanced import ErrorAnalysisVisualizer
        visualizer = ErrorAnalysisVisualizer(save_dir=save_dir)

        # 可视化
        visualizer.visualize_error_patterns(error_data)

        # 导出表格
        visualizer.export_to_table(error_data)

        print("✅ Error analysis visualization completed!")
    except ImportError:
        print("⚠️ visualization_enhanced not available, exporting as Excel table only")

        # 降级方案：直接导出为Excel
        if 'case_studies' in error_data:
            df = pd.DataFrame(error_data['case_studies'])
            excel_path = os.path.join(save_dir, 'error_analysis_table.xlsx')
            df.to_excel(excel_path, index=False)
            print(f"✅ Saved: {excel_path}")


def main():
    parser = argparse.ArgumentParser(description='Error Correction Analysis')
    parser.add_argument('--data_file', type=str, default='小案例ai问询.xlsx',
                        help='Path to data file')
    parser.add_argument('--text_model', type=str, default='./models/best_text_only_model.pth',
                        help='Path to text-only model')
    parser.add_argument('--full_model', type=str, default='./models/best_full_model.pth',
                        help='Path to full model')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to analyze')
    parser.add_argument('--n_cases', type=int, default=5,
                        help='Number of case studies')

    args = parser.parse_args()

    analyzer = ErrorCorrectionAnalyzer()

    analyzer.run_full_analysis(
        data_file=args.data_file,
        text_only_path=args.text_model,
        full_model_path=args.full_model,
        max_samples=args.max_samples,
        n_cases=args.n_cases
    )

    # 自动可视化JSON结果
    json_path = './outputs/reports/error_correction_report.json'
    if os.path.exists(json_path):
        print("\n📊 Visualizing error analysis results...")
        visualize_error_results(json_path, './outputs/figures')


if __name__ == "__main__":
    main()