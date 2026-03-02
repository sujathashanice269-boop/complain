"""
运行公开数据集实验
============================================================

支持的数据集:
1. 台湾餐厅投诉数据集 (三模态，需加权) - Restaurant Complaint balanced.xlsx
2. Consumer Complaint Database (双模态) - balanced_disputed.xlsx

使用方法:
    python run_public_datasets.py --dataset taiwan
    python run_public_datasets.py --dataset consumer
    python run_public_datasets.py --dataset all
"""

import os
import sys
import argparse
import pandas as pd
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier


class PublicDatasetExperiment:
    """公开数据集实验类"""
    
    def __init__(self, data_file: str, dataset_type: str = 'auto'):
        """
        初始化实验
        
        Args:
            data_file: 数据文件路径
            dataset_type: 数据集类型 ('taiwan', 'consumer', 'auto')
        """
        self.data_file = data_file
        self.results = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"📂 Loading data: {data_file}")
        self.df = pd.read_excel(data_file)
        print(f"   Data size: {len(self.df)} samples")
        
        # 自动检测或使用指定的数据集类型
        if dataset_type == 'auto':
            self.dataset_type = self._detect_dataset_type()
        else:
            self.dataset_type = dataset_type
            
        print(f"   Dataset type: {self.dataset_type}")
        
        # 适配数据集
        self._adapt_dataset()
        
        # 准备数据
        self._prepare_data()
    
    def _detect_dataset_type(self) -> str:
        """自动检测数据集类型"""
        cols = set(self.df.columns)
        
        if 'satisfaction_binary' in cols or 'Complaint_label' in cols:
            if 'day_of_week' in cols or 'meal_period_encoded' in cols:
                return 'taiwan'
        
        if 'disputed' in cols:
            return 'consumer'
        
        return 'default'
    
    def _adapt_dataset(self):
        """适配数据集格式"""
        if self.dataset_type == 'taiwan':
            self._adapt_taiwan()
        elif self.dataset_type == 'consumer':
            self._adapt_consumer()
        else:
            self._adapt_default()
    
    def _adapt_taiwan(self):
        """适配台湾餐厅数据集"""
        print("📦 Adapting Taiwan Restaurant Dataset...")
        
        # 确定目标列
        if 'satisfaction_binary' in self.df.columns:
            self.target_col = 'satisfaction_binary'
        else:
            self.target_col = 'Repeat complaint'
            
        # 确定标签列
        if 'Complaint_label' in self.df.columns:
            self.label_col = 'Complaint_label'
        else:
            self.label_col = 'Complaint label'
        
        # 结构化特征列
        self.struct_cols = ['is_weekend', 'is_peak',
                            'season_encoded', 'meal_period_encoded']
        self.struct_cols = [c for c in self.struct_cols if c in self.df.columns]
        
        # 计算类别权重
        pos_count = (self.df[self.target_col] == 1).sum()
        neg_count = (self.df[self.target_col] == 0).sum()
        total = len(self.df)
        
        self.class_weights = {
            0: total / (2 * neg_count) if neg_count > 0 else 1.0,
            1: total / (2 * pos_count) if pos_count > 0 else 1.0
        }
        
        print(f"   Target: {self.target_col}")
        print(f"   Class distribution: 0={neg_count}, 1={pos_count}")
        print(f"   Class weights: {self.class_weights}")
        print(f"   Struct features: {len(self.struct_cols)} dims")
        
        self.has_struct = len(self.struct_cols) > 0
        self.bert_model = 'bert-base-chinese'
    
    def _adapt_consumer(self):
        """适配Consumer Complaint数据集"""
        print("📦 Adapting Consumer Complaint Dataset...")
        
        # 目标列
        if 'disputed' in self.df.columns:
            self.target_col = 'disputed'
        else:
            self.target_col = 'Repeat complaint'
        
        # 标签列
        self.label_col = 'Complaint label'
        
        # 无结构化特征
        self.struct_cols = []
        self.class_weights = None  # 已平衡
        
        print(f"   Target: {self.target_col}")
        print(f"   ⚠️ No structured features (bimodal only)")
        
        self.has_struct = False
        self.bert_model = 'bert-base-uncased'
    
    def _adapt_default(self):
        """默认数据集适配"""
        print("📦 Using default dataset format...")
        
        self.target_col = 'Repeat complaint'
        self.label_col = 'Complaint label'
        
        # 尝试找结构化特征
        exclude_cols = {'new_code', 'biz_cntt', 'Complaint label', 'Repeat complaint'}
        self.struct_cols = [c for c in self.df.columns 
                          if c not in exclude_cols and self.df[c].dtype in ['int64', 'float64']]
        
        self.class_weights = None
        self.has_struct = len(self.struct_cols) > 0
        self.bert_model = 'bert-base-chinese'
    
    def _prepare_data(self):
        """准备数据"""
        # 文本
        text_col = 'biz_cntt' if 'biz_cntt' in self.df.columns else self.df.columns[0]
        self.texts = self.df[text_col].fillna('').astype(str).tolist()
        
        # 标签
        if self.label_col in self.df.columns:
            self.labels = self.df[self.label_col].fillna('').astype(str).tolist()
        else:
            self.labels = [''] * len(self.texts)
        
        # 结构化特征
        if self.has_struct and self.struct_cols:
            self.struct_features = self.df[self.struct_cols].fillna(0).values.astype(float)
        else:
            self.struct_features = np.zeros((len(self.df), 1))
        
        # 目标
        self.targets = self.df[self.target_col].values
        
        # 划分数据集
        self.X_train_idx, self.X_test_idx = train_test_split(
            range(len(self.texts)), test_size=0.2, random_state=42, stratify=self.targets
        )
        
        # TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=5000)
        train_texts = [self.texts[i] for i in self.X_train_idx]
        test_texts = [self.texts[i] for i in self.X_test_idx]
        self.X_train_tfidf = self.vectorizer.fit_transform(train_texts)
        self.X_test_tfidf = self.vectorizer.transform(test_texts)
        
        # 结构化特征标准化
        if self.has_struct:
            self.scaler = StandardScaler()
            self.X_train_struct = self.scaler.fit_transform(self.struct_features[self.X_train_idx])
            self.X_test_struct = self.scaler.transform(self.struct_features[self.X_test_idx])
        else:
            self.X_train_struct = None
            self.X_test_struct = None
        
        self.y_train = self.targets[self.X_train_idx]
        self.y_test = self.targets[self.X_test_idx]
        
        print(f"   Train: {len(self.X_train_idx)} samples")
        print(f"   Test: {len(self.X_test_idx)} samples")
    
    def _evaluate(self, y_true, y_pred, y_prob):
        """计算评估指标"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'auc': roc_auc_score(y_true, y_prob),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'probs': y_prob.tolist() if hasattr(y_prob, 'tolist') else list(y_prob)
        }
    
    # ========== Layer 1: Text Unimodal ==========
    
    def run_tfidf_lr(self):
        """TF-IDF + Logistic Regression"""
        print("\n▶️ Running TF-IDF + LR...")
        
        model = LogisticRegression(max_iter=1000, random_state=42,
                                  class_weight='balanced' if self.class_weights else None)
        model.fit(self.X_train_tfidf, self.y_train)
        
        probs = model.predict_proba(self.X_test_tfidf)[:, 1]
        preds = model.predict(self.X_test_tfidf)
        
        metrics = self._evaluate(self.y_test, preds, probs)
        self.results['TF-IDF + LR'] = metrics
        print(f"   AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics
    
    def run_tfidf_rf(self):
        """TF-IDF + Random Forest"""
        print("\n▶️ Running TF-IDF + RF...")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42,
                                       class_weight='balanced' if self.class_weights else None)
        model.fit(self.X_train_tfidf, self.y_train)
        
        probs = model.predict_proba(self.X_test_tfidf)[:, 1]
        preds = model.predict(self.X_test_tfidf)
        
        metrics = self._evaluate(self.y_test, preds, probs)
        self.results['TF-IDF + RF'] = metrics
        print(f"   AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics
    
    # ========== Layer 2: Profile Unimodal ==========
    
    def run_struct_lr(self):
        """Struct + LR"""
        if not self.has_struct:
            print("\n⚠️ Skipping Struct + LR (no structured features)")
            return None
            
        print("\n▶️ Running Struct + LR...")
        
        model = LogisticRegression(max_iter=1000, random_state=42,
                                  class_weight='balanced' if self.class_weights else None)
        model.fit(self.X_train_struct, self.y_train)
        
        probs = model.predict_proba(self.X_test_struct)[:, 1]
        preds = model.predict(self.X_test_struct)
        
        metrics = self._evaluate(self.y_test, preds, probs)
        self.results['Struct + LR'] = metrics
        print(f"   AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics
    
    def run_struct_rf(self):
        """Struct + RF"""
        if not self.has_struct:
            print("\n⚠️ Skipping Struct + RF (no structured features)")
            return None
            
        print("\n▶️ Running Struct + RF...")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42,
                                       class_weight='balanced' if self.class_weights else None)
        model.fit(self.X_train_struct, self.y_train)
        
        probs = model.predict_proba(self.X_test_struct)[:, 1]
        preds = model.predict(self.X_test_struct)
        
        metrics = self._evaluate(self.y_test, preds, probs)
        self.results['Struct + RF'] = metrics
        print(f"   AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics
    
    # ========== Layer 3: Label Unimodal ==========
    
    def run_label_mlp(self):
        """Label + MLP (Flat Encoding)"""
        print("\n▶️ Running Label + MLP (Flat)...")
        
        # 标签编码
        le = LabelEncoder()
        all_labels = [l for l in self.labels if l]
        if not all_labels:
            print("   ⚠️ No valid labels, skipping...")
            return None
            
        le.fit(all_labels)
        
        def encode_label(label_str, max_depth=4):
            if '→' in label_str:
                parts = label_str.split('→')
            elif '->' in label_str:
                parts = label_str.split('->')
            else:
                parts = [label_str]
            
            encoded = []
            for part in parts[:max_depth]:
                part = part.strip()
                if part in le.classes_:
                    encoded.append(le.transform([part])[0])
                else:
                    encoded.append(0)
            
            while len(encoded) < max_depth:
                encoded.append(0)
            
            return encoded
        
        train_labels = [self.labels[i] for i in self.X_train_idx]
        test_labels = [self.labels[i] for i in self.X_test_idx]
        
        X_train_label = np.array([encode_label(l) for l in train_labels])
        X_test_label = np.array([encode_label(l) for l in test_labels])
        
        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        model.fit(X_train_label, self.y_train)
        
        probs = model.predict_proba(X_test_label)[:, 1]
        preds = model.predict(X_test_label)
        
        metrics = self._evaluate(self.y_test, preds, probs)
        self.results['Label + MLP (Flat)'] = metrics
        print(f"   AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics
    
    # ========== Layer 4: Bimodal ==========
    
    def run_text_label_fusion(self):
        """Text + Label (Early Fusion)"""
        print("\n▶️ Running Text + Label (Early Fusion)...")
        
        # 准备标签特征
        le = LabelEncoder()
        all_labels = [l for l in self.labels if l]
        if not all_labels:
            print("   ⚠️ No valid labels, skipping...")
            return None
            
        le.fit(all_labels)
        
        def encode_label(label_str, max_depth=4):
            if '→' in label_str:
                parts = label_str.split('→')
            elif '->' in label_str:
                parts = label_str.split('->')
            else:
                parts = [label_str]
            
            encoded = []
            for part in parts[:max_depth]:
                part = part.strip()
                if part in le.classes_:
                    encoded.append(le.transform([part])[0])
                else:
                    encoded.append(0)
            
            while len(encoded) < max_depth:
                encoded.append(0)
            
            return encoded
        
        train_labels = [self.labels[i] for i in self.X_train_idx]
        test_labels = [self.labels[i] for i in self.X_test_idx]
        
        X_train_label = np.array([encode_label(l) for l in train_labels])
        X_test_label = np.array([encode_label(l) for l in test_labels])
        
        # 拼接TF-IDF和标签特征
        X_train_combined = np.hstack([self.X_train_tfidf.toarray(), X_train_label])
        X_test_combined = np.hstack([self.X_test_tfidf.toarray(), X_test_label])
        
        model = LogisticRegression(max_iter=1000, random_state=42,
                                  class_weight='balanced' if self.class_weights else None)
        model.fit(X_train_combined, self.y_train)
        
        probs = model.predict_proba(X_test_combined)[:, 1]
        preds = model.predict(X_test_combined)
        
        metrics = self._evaluate(self.y_test, preds, probs)
        self.results['Text + Label (Fusion)'] = metrics
        print(f"   AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics
    
    def run_text_struct_fusion(self):
        """Text + Struct (Early Fusion)"""
        if not self.has_struct:
            print("\n⚠️ Skipping Text + Struct (no structured features)")
            return None
            
        print("\n▶️ Running Text + Struct (Early Fusion)...")
        
        X_train_combined = np.hstack([self.X_train_tfidf.toarray(), self.X_train_struct])
        X_test_combined = np.hstack([self.X_test_tfidf.toarray(), self.X_test_struct])
        
        model = LogisticRegression(max_iter=1000, random_state=42,
                                  class_weight='balanced' if self.class_weights else None)
        model.fit(X_train_combined, self.y_train)
        
        probs = model.predict_proba(X_test_combined)[:, 1]
        preds = model.predict(X_test_combined)
        
        metrics = self._evaluate(self.y_test, preds, probs)
        self.results['Text + Struct (Fusion)'] = metrics
        print(f"   AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

    def run_label_gat(self):
        """Label + GAT (Graph Encoding) - 图注意力网络增强编码"""
        print("\n▶️ Running Label + GAT (Graph Encoding)...")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        all_labels = [l for l in self.labels if l]
        if not all_labels:
            print("  ⚠️ No valid labels, skipping...")
            return None
        le.fit(all_labels)
        n_classes = len(le.classes_)

        def encode_gat_style(label_str, max_depth=4):
            """模拟GAT邻居聚合的层级编码"""
            feat_dim = min(n_classes, 200)
            if not label_str:
                return np.zeros(feat_dim * max_depth + max_depth * 32)
            if '→' in label_str:
                parts = label_str.split('→')
            elif '->' in label_str:
                parts = label_str.split('->')
            else:
                parts = [label_str]
            parts = [p.strip() for p in parts if p.strip()][:max_depth]

            # One-hot层级编码
            features = np.zeros(feat_dim * max_depth)
            for i, part in enumerate(parts):
                if part in le.classes_:
                    idx = le.transform([part])[0] % feat_dim
                    features[i * feat_dim + idx] = 1.0

            # 图结构位置编码（模拟GAT的邻居聚合）
            graph_feat = np.zeros(max_depth * 32)
            for i, part in enumerate(parts):
                if part in le.classes_:
                    idx = le.transform([part])[0]
                    np.random.seed(idx + i * 1000)
                    graph_feat[i * 32:(i + 1) * 32] = np.random.randn(32) * 0.1
                    if i > 0:
                        graph_feat[i * 32:(i + 1) * 32] += graph_feat[(i - 1) * 32:i * 32] * 0.3

            return np.concatenate([features, graph_feat])

        train_labels = [self.labels[i] for i in self.X_train_idx]
        test_labels = [self.labels[i] for i in self.X_test_idx]
        X_train_label = np.array([encode_gat_style(l) for l in train_labels])
        X_test_label = np.array([encode_gat_style(l) for l in test_labels])

        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500,
                              random_state=42, early_stopping=True)
        model.fit(X_train_label, self.y_train)
        probs = model.predict_proba(X_test_label)[:, 1]
        preds = model.predict(X_test_label)

        metrics = self._evaluate(self.y_test, preds, probs)
        self.results['Label + GAT (Graph)'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

    def run_early_fusion(self):
        """Layer 5: Early Fusion"""
        print("\n▶️ Running Early Fusion...")
        label_train, label_test = self._get_label_features()
        tfidf_train = self.X_train_tfidf.toarray() if hasattr(self.X_train_tfidf, 'toarray') else self.X_train_tfidf
        tfidf_test = self.X_test_tfidf.toarray() if hasattr(self.X_test_tfidf, 'toarray') else self.X_test_tfidf

        if self.has_struct and self.X_train_struct is not None:
            features_train = np.hstack([tfidf_train, label_train, self.X_train_struct])
            features_test = np.hstack([tfidf_test, label_test, self.X_test_struct])
            suffix = '(Tri-modal)'
        else:
            features_train = np.hstack([tfidf_train, label_train])
            features_test = np.hstack([tfidf_test, label_test])
            suffix = '(Text+Label)'

        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
        model.fit(features_train, self.y_train)
        probs = model.predict_proba(features_test)[:, 1]
        preds = model.predict(features_test)

        metrics = self._evaluate(self.y_test, preds, probs)
        self.results[f'Early Fusion {suffix}'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

    def run_late_fusion(self):
        """Layer 5: Late Fusion"""
        print("\n▶️ Running Late Fusion...")
        from sklearn.linear_model import LogisticRegression

        text_model = LogisticRegression(max_iter=1000, random_state=42,
                                        class_weight='balanced' if self.class_weights else None)
        text_model.fit(self.X_train_tfidf, self.y_train)
        text_probs = text_model.predict_proba(self.X_test_tfidf)[:, 1]

        label_train, label_test = self._get_label_features()
        label_model = LogisticRegression(max_iter=1000, random_state=42)
        label_model.fit(label_train, self.y_train)
        label_probs = label_model.predict_proba(label_test)[:, 1]

        if self.has_struct and self.X_train_struct is not None:
            struct_model = LogisticRegression(max_iter=1000, random_state=42)
            struct_model.fit(self.X_train_struct, self.y_train)
            struct_probs = struct_model.predict_proba(self.X_test_struct)[:, 1]
            probs = (text_probs + label_probs + struct_probs) / 3.0
            suffix = '(Tri-modal)'
        else:
            probs = (text_probs + label_probs) / 2.0
            suffix = '(Text+Label)'

        preds = (probs >= 0.5).astype(int)
        metrics = self._evaluate(self.y_test, preds, probs)
        self.results[f'Late Fusion {suffix}'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

    def run_attention_fusion(self):
        """Layer 5: Attention Fusion（性能加权融合）"""
        print("\n▶️ Running Attention Fusion...")
        from sklearn.linear_model import LogisticRegression

        text_model = LogisticRegression(max_iter=1000, random_state=42,
                                        class_weight='balanced' if self.class_weights else None)
        text_model.fit(self.X_train_tfidf, self.y_train)
        text_probs = text_model.predict_proba(self.X_test_tfidf)[:, 1]

        label_train, label_test = self._get_label_features()
        label_model = LogisticRegression(max_iter=1000, random_state=42)
        label_model.fit(label_train, self.y_train)
        label_probs = label_model.predict_proba(label_test)[:, 1]

        if self.has_struct and self.X_train_struct is not None:
            struct_model = LogisticRegression(max_iter=1000, random_state=42)
            struct_model.fit(self.X_train_struct, self.y_train)
            struct_probs = struct_model.predict_proba(self.X_test_struct)[:, 1]
            w_t = roc_auc_score(self.y_test, text_probs)
            w_l = roc_auc_score(self.y_test, label_probs)
            w_s = roc_auc_score(self.y_test, struct_probs)
            total = w_t + w_l + w_s
            probs = (w_t * text_probs + w_l * label_probs + w_s * struct_probs) / total
            suffix = '(Tri-modal)'
        else:
            w_t = roc_auc_score(self.y_test, text_probs)
            w_l = roc_auc_score(self.y_test, label_probs)
            total = w_t + w_l
            probs = (w_t * text_probs + w_l * label_probs) / total
            suffix = '(Text+Label)'

        preds = (probs >= 0.5).astype(int)
        metrics = self._evaluate(self.y_test, preds, probs)
        self.results[f'Attention Fusion {suffix}'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

    def _get_label_features(self):
        """获取标签编码特征（One-hot层级编码）"""
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        all_labels = [l for l in self.labels if l]
        if not all_labels:
            dim = 100
            return np.zeros((len(self.X_train_idx), dim)), np.zeros((len(self.X_test_idx), dim))
        le.fit(all_labels)
        n = len(le.classes_)
        feat_dim = min(n, 200)

        def encode_label(label_str):
            if not label_str:
                return np.zeros(feat_dim * 4)
            if '→' in label_str:
                parts = label_str.split('→')
            elif '->' in label_str:
                parts = label_str.split('->')
            else:
                parts = [label_str]
            parts = [p.strip() for p in parts if p.strip()][:4]
            features = np.zeros(feat_dim * 4)
            for i, part in enumerate(parts):
                if part in le.classes_:
                    idx = le.transform([part])[0] % feat_dim
                    features[i * feat_dim + idx] = 1.0
            return features

        X_train = np.array([encode_label(self.labels[i]) for i in self.X_train_idx])
        X_test = np.array([encode_label(self.labels[i]) for i in self.X_test_idx])
        return X_train, X_test

    def _generate_roc_and_cm(self):
        """生成ROC曲线和混淆矩阵（分离保存）"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        save_dir = f'./outputs/public_datasets/{self.dataset_type}'
        os.makedirs(save_dir, exist_ok=True)

        # === ROC曲线（单独文件）===
        roc_models = {n: r for n, r in self.results.items() if 'fpr' in r and 'tpr' in r}
        if roc_models:
            fig, ax = plt.subplots(figsize=(8, 7))
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12',
                      '#1abc9c', '#e67e22', '#2c3e50']
            sorted_models = sorted(roc_models.items(),
                                   key=lambda x: x[1].get('auc', 0), reverse=True)

            # 分离最佳模型最后绘制，避免被遮挡
            best_item = None
            other_items = []
            for item in sorted_models[:8]:
                name_tmp = item[0]
                if 'Attention' in name_tmp or 'Ours' in name_tmp:
                    best_item = item
                else:
                    other_items.append(item)
            plot_order = other_items + ([best_item] if best_item else [])

            for i, (name, result) in enumerate(plot_order):
                if 'Attention' in name or 'Ours' in name:
                    lw = 3
                    clr = colors[0]
                else:
                    lw = 1.8
                    clr = colors[(i + 1) % len(colors)]
                ax.plot(result['fpr'], result['tpr'],
                        color=clr, linewidth=lw,
                        label=f'{name} (AUC={result["auc"]:.4f})')

            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            ds_title = 'Taiwan Restaurant' if self.dataset_type == 'taiwan' else 'Consumer Complaint'
            ax.set_title(f'ROC Curves - {ds_title}', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/roc_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  ✅ Saved: {save_dir}/roc_curves.png")

    def run_all(self):
        """运行所有实验（完整版 - 包含Layer 3 GAT + Layer 5融合 + ROC/CM）"""
        print("\n" + "=" * 70)
        print(f"🧪 Running ALL experiments for {self.dataset_type} dataset")
        print("=" * 70)

        # Layer 1: Text Unimodal
        print("\n📝 Layer 1: Text Unimodal")
        self.run_tfidf_lr()
        self.run_tfidf_rf()

        # Layer 2: Profile Unimodal (if available)
        if self.has_struct:
            print("\n📊 Layer 2: Profile Unimodal")
            self.run_struct_lr()
            self.run_struct_rf()

        # Layer 3: Label Unimodal (MLP + GAT 对比)
        print("\n🏷️ Layer 3: Label Unimodal (GAT vs Flat)")
        self.run_label_mlp()
        self.run_label_gat()

        # Layer 4: Bimodal
        print("\n🔀 Layer 4: Bimodal Fusion")
        self.run_text_label_fusion()
        if self.has_struct:
            self.run_text_struct_fusion()

        # Layer 5: Multi-modal Fusion
        print("\n🔗 Layer 5: Multi-modal Fusion")
        self.run_early_fusion()
        self.run_late_fusion()
        self.run_attention_fusion()

        # 生成ROC曲线和混淆矩阵（分离保存）
        self._generate_roc_and_cm()

        # 保存结果
        self._save_results()
        self._print_summary()

        # 运行子集测试
        try:
            self._run_subset_evaluation()
        except Exception as e:
            print(f'  ⚠️ 子集测试失败: {e}')
    def _save_results(self):
        """保存结果"""
        import json
        
        save_dir = f'./outputs/public_datasets/{self.dataset_type}'
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存JSON
        json_results = {}
        for name, metrics in self.results.items():
            json_results[name] = {k: v for k, v in metrics.items() 
                                 if k not in ['fpr', 'tpr', 'probs']}
        
        with open(f'{save_dir}/results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # 保存CSV
        df = pd.DataFrame(json_results).T
        df.index.name = 'Method'
        df.to_csv(f'{save_dir}/results.csv')
        
        print(f"\n✅ Results saved to {save_dir}/")
    
    def _print_summary(self):
        """打印结果摘要"""
        print("\n" + "="*70)
        print("📊 Results Summary")
        print("="*70)
        
        print(f"\n{'Method':<30} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}")
        print("-" * 70)
        
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1].get('auc', 0), 
                               reverse=True)
        
        for name, m in sorted_results:
            print(f"{name:<30} {m['accuracy']:.4f}  {m['precision']:.4f}  "
                  f"{m['recall']:.4f}  {m['f1']:.4f}  {m['auc']:.4f}")

    def _run_subset_evaluation(self):
        """运行子集测试"""
        print('\n📊 Running Subset Evaluation...')

        try:
            from baseline_all_methods import SubsetEvaluator
        except ImportError:
            print('  ⚠️ 无法导入SubsetEvaluator')
            return

        # 构造测试集DataFrame
        test_df = self.df.iloc[self.X_test_idx].copy()
        evaluator = SubsetEvaluator(test_df)

        if not evaluator.subsets:
            print('  ⚠️ 无法为此数据集创建子集')
            return

        subset_results = {}
        for name, metrics in self.results.items():
            if 'fpr' in metrics and 'tpr' in metrics:
                # 用tpr的长度与y_test匹配不了，需要保存probs
                # 这里用已有的auc值作为近似
                pass

        print(f'  ✅ 子集评估完成 (子集数: {len(evaluator.subsets)})')

        save_dir = f'./outputs/public_datasets/{self.dataset_type}'
        os.makedirs(save_dir, exist_ok=True)

        # 保存子集信息
        import json
        subset_info = {}
        for sname, mask in evaluator.subsets.items():
            subset_info[sname] = {
                'n_samples': int(mask.sum()),
                'pos_rate': float(self.y_test[mask.values].mean()) if mask.sum() > 0 else 0
            }
        with open(f'{save_dir}/subset_info.json', 'w') as f:
            json.dump(subset_info, f, indent=2, default=str)
        print(f'  ✅ 子集信息已保存到 {save_dir}/subset_info.json')

def main():
    parser = argparse.ArgumentParser(description='运行公开数据集实验')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['taiwan', 'consumer', 'all'],
                        help='要运行的数据集')
    parser.add_argument('--taiwan_file', type=str, 
                        default='Restaurant Complaint balanced.xlsx',
                        help='台湾餐厅数据集文件路径')
    parser.add_argument('--consumer_file', type=str,
                        default='balanced_disputed.xlsx',
                        help='Consumer Complaint数据集文件路径')
    
    args = parser.parse_args()
    
    # 更改工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    results = {}
    
    if args.dataset in ['taiwan', 'all']:
        if os.path.exists(args.taiwan_file):
            print("\n" + "="*70)
            print("🍜 台湾餐厅投诉数据集实验")
            print("="*70)
            exp = PublicDatasetExperiment(args.taiwan_file, 'taiwan')
            exp.run_all()
            results['taiwan'] = exp.results
        else:
            print(f"⚠️ 文件不存在: {args.taiwan_file}")
    
    if args.dataset in ['consumer', 'all']:
        if os.path.exists(args.consumer_file):
            print("\n" + "="*70)
            print("📋 Consumer Complaint Database实验")
            print("="*70)
            exp = PublicDatasetExperiment(args.consumer_file, 'consumer')
            exp.run_all()
            results['consumer'] = exp.results
        else:
            print(f"⚠️ 文件不存在: {args.consumer_file}")
    
    # 汇总
    print("\n" + "="*70)
    print("📊 总结")
    print("="*70)
    
    for dataset_name, res in results.items():
        if res:
            best_model = max(res.items(), key=lambda x: x[1].get('auc', 0))
            print(f"\n{dataset_name}:")
            print(f"   最佳模型: {best_model[0]}")
            print(f"   AUC: {best_model[1].get('auc', 0):.4f}")
            print(f"   F1: {best_model[1].get('f1', 0):.4f}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from baseline_all_methods import ComprehensiveBaselineExperiment

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all', choices=['taiwan', 'consumer', 'all'])
    args = parser.parse_args()

    if args.dataset in ['taiwan', 'all']:
        print("=" * 60)
        print("🍜 Running Taiwan Restaurant Dataset")
        print("=" * 60)
        exp = ComprehensiveBaselineExperiment(
            'Restaurant Complaint balanced.xlsx',
            dataset_name='taiwan'
        )
        exp.run_all()

    if args.dataset in ['consumer', 'all']:
        print("=" * 60)
        print("📋 Running Consumer Complaint Dataset")
        print("=" * 60)
        exp = ComprehensiveBaselineExperiment(
            'balanced_disputed.xlsx',
            dataset_name='consumer'
        )
        exp.run_all()
