"""
Fusion Strategy Comparison Module
==================================
Module 2: Compare different multimodal fusion strategies

Three fusion strategies implemented:
1. Concat (Baseline): Direct concatenation H_final = Concat(H_text, H_label, H_struct)
2. Simple Attention: Standard dot-product attention without gating
3. Gated Cross-Attention (Our Method): Text-led gated attention with modal_gate mechanism

Purpose: Prove gated mechanism filters noise better than simple concatenation/attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import from existing model.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (
    GATLabelEncoder, 
    TextMultiTokenGenerator, 
    StructMultiTokenGenerator
)
from config import Config


# =============================================================================
# Fusion Strategy 1: Concat (Baseline)
# =============================================================================

class ConcatFusion(nn.Module):
    """
    Simple Concatenation Fusion
    
    Formula: H_final = Concat(H_text, H_label, H_struct)
    No learnable interaction between modalities
    """
    
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        # Just a projection layer to match expected output
        self.output_proj = nn.Linear(dim * 3, dim * 3)
        self.layer_norm = nn.LayerNorm(dim * 3)
    
    def forward(self, text_feat, label_feat, struct_feat, 
                label_mask=None, return_attention=False):
        """
        Args:
            text_feat: [batch, seq_len, dim] or [batch, dim]
            label_feat: [batch, num_nodes, dim]
            struct_feat: [batch, seq_len, dim] or [batch, dim]
            label_mask: [batch, num_nodes] - True indicates padding
            return_attention: Whether to return attention weights
            
        Returns:
            text_pooled: [batch, dim]
            label_pooled: [batch, dim]
            struct_pooled: [batch, dim]
            attention_weights: Empty dict for concat (no attention)
        """
        # Pool if needed
        if text_feat.dim() == 3:
            text_pooled = text_feat.mean(dim=1)
        else:
            text_pooled = text_feat
            
        if label_feat.dim() == 3:
            if label_mask is not None:
                # Masked average pooling
                valid_mask = ~label_mask
                mask_expanded = valid_mask.unsqueeze(-1).float()
                label_pooled = (label_feat * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
            else:
                label_pooled = label_feat.mean(dim=1)
        else:
            label_pooled = label_feat
            
        if struct_feat.dim() == 3:
            struct_pooled = struct_feat.mean(dim=1)
        else:
            struct_pooled = struct_feat
        
        attention_weights = {} if return_attention else None
        
        return text_pooled, label_pooled, struct_pooled, attention_weights


# =============================================================================
# Fusion Strategy 2: Simple Attention (Without Gating)
# =============================================================================

class SimpleAttentionFusion(nn.Module):
    """
    Simple Cross-Modal Attention Fusion (Without Gating)
    
    Standard dot-product attention for cross-modal interaction
    No gating mechanism, no text-bias
    """
    
    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        
        # Cross-modal attention (text to other modalities)
        self.text_to_label_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_struct_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Self-attention for each modality
        self.text_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.label_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.struct_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.text_norm = nn.LayerNorm(dim)
        self.label_norm = nn.LayerNorm(dim)
        self.struct_norm = nn.LayerNorm(dim)
        
        # Simple averaging weights (no gating)
        # Equal weights for all modalities
        self.register_buffer('modal_weights', torch.tensor([1/3, 1/3, 1/3]))
    
    def forward(self, text_tokens, label_tokens, struct_tokens,
                label_mask=None, return_attention=True):
        """
        Args:
            text_tokens: [batch, num_tokens, dim]
            label_tokens: [batch, num_nodes, dim]
            struct_tokens: [batch, num_tokens, dim]
            label_mask: [batch, num_nodes] - True indicates padding
            return_attention: Whether to return attention weights
        """
        attention_weights = {}
        
        # 1. Self-attention
        text_self, _ = self.text_self_attn(text_tokens, text_tokens, text_tokens)
        text_tokens = self.text_norm(text_tokens + text_self)
        
        label_self, attn_l = self.label_self_attn(
            label_tokens, label_tokens, label_tokens,
            key_padding_mask=label_mask,
            need_weights=return_attention,
            average_attn_weights=False
        )
        label_tokens = self.label_norm(label_tokens + label_self)
        
        struct_self, _ = self.struct_self_attn(struct_tokens, struct_tokens, struct_tokens)
        struct_tokens = self.struct_norm(struct_tokens + struct_self)
        
        # 2. Cross-modal attention (text queries other modalities)
        text_to_label, attn_t2l = self.text_to_label_attn(
            text_tokens, label_tokens, label_tokens,
            key_padding_mask=label_mask,
            need_weights=return_attention,
            average_attn_weights=False
        )
        
        text_to_struct, attn_t2s = self.text_to_struct_attn(
            text_tokens, struct_tokens, struct_tokens,
            need_weights=return_attention,
            average_attn_weights=False
        )
        
        # 3. Feature enhancement (simple addition, no learnable weights)
        text_enhanced = text_tokens + 0.5 * text_to_label + 0.5 * text_to_struct
        label_enhanced = label_tokens
        struct_enhanced = struct_tokens
        
        # 4. Pooling
        text_pooled = text_enhanced.mean(dim=1)
        
        if label_mask is not None:
            valid_mask = ~label_mask
            mask_expanded = valid_mask.unsqueeze(-1).float()
            label_pooled = (label_enhanced * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            label_pooled = label_enhanced.mean(dim=1)
            
        struct_pooled = struct_enhanced.mean(dim=1)
        
        # 5. No gating - just return pooled features
        batch_size = text_pooled.size(0)
        
        if return_attention:
            attention_weights = {
                'text_to_label': attn_t2l,
                'text_to_struct': attn_t2s,
                'label_self': attn_l,
                'modal_weights': self.modal_weights.unsqueeze(0).expand(batch_size, -1),
            }
        
        return text_pooled, label_pooled, struct_pooled, attention_weights


# =============================================================================
# Fusion Strategy 3: Gated Cross-Attention (Our Method)
# =============================================================================

class GatedCrossAttentionFusion(nn.Module):
    """
    Gated Cross-Modal Attention Fusion (Our Method)
    
    Key innovations:
    1. Text-led cross-modal attention (text queries other modalities)
    2. Learnable gating mechanism to weight modality contributions
    3. Text bias to ensure text modality has higher influence
    4. Learnable cross-modal fusion weights
    """
    
    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        
        # Cross-modal attention (text to other modalities)
        self.text_to_label_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_struct_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Self-attention for each modality
        self.text_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.label_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.struct_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.text_norm = nn.LayerNorm(dim)
        self.label_norm = nn.LayerNorm(dim)
        self.struct_norm = nn.LayerNorm(dim)
        
        # ========== Key Innovation: Gated Fusion ==========
        self.modal_gate = nn.Sequential(
            nn.Linear(dim * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        
        # Text bias (ensures text has higher influence)
        self.text_bias = nn.Parameter(torch.tensor(0.5))
        
        # Learnable cross-modal fusion weights
        self.cross_modal_weight_label = nn.Parameter(torch.tensor(0.0))
        self.cross_modal_weight_struct = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, text_tokens, label_tokens, struct_tokens,
                label_mask=None, return_attention=True):
        """
        Args:
            text_tokens: [batch, num_tokens, dim]
            label_tokens: [batch, num_nodes, dim]
            struct_tokens: [batch, num_tokens, dim]
            label_mask: [batch, num_nodes] - True indicates padding
            return_attention: Whether to return attention weights
        """
        attention_weights = {}
        
        # 1. Self-attention
        text_self, _ = self.text_self_attn(text_tokens, text_tokens, text_tokens)
        text_tokens = self.text_norm(text_tokens + text_self)
        
        label_self, attn_l = self.label_self_attn(
            label_tokens, label_tokens, label_tokens,
            key_padding_mask=label_mask,
            need_weights=return_attention,
            average_attn_weights=False
        )
        label_tokens = self.label_norm(label_tokens + label_self)
        
        struct_self, _ = self.struct_self_attn(struct_tokens, struct_tokens, struct_tokens)
        struct_tokens = self.struct_norm(struct_tokens + struct_self)
        
        # 2. Text-led cross-modal attention
        text_to_label, attn_t2l = self.text_to_label_attn(
            text_tokens, label_tokens, label_tokens,
            key_padding_mask=label_mask,
            need_weights=return_attention,
            average_attn_weights=False
        )
        
        text_to_struct, attn_t2s = self.text_to_struct_attn(
            text_tokens, struct_tokens, struct_tokens,
            need_weights=return_attention,
            average_attn_weights=False
        )
        
        # 3. Learnable feature enhancement
        weight_label = torch.sigmoid(self.cross_modal_weight_label)
        weight_struct = torch.sigmoid(self.cross_modal_weight_struct)
        text_enhanced = text_tokens + weight_label * text_to_label + weight_struct * text_to_struct
        label_enhanced = label_tokens
        struct_enhanced = struct_tokens
        
        # 4. Pooling
        text_pooled = text_enhanced.mean(dim=1)
        
        if label_mask is not None:
            valid_mask = ~label_mask
            mask_expanded = valid_mask.unsqueeze(-1).float()
            label_pooled = (label_enhanced * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            label_pooled = label_enhanced.mean(dim=1)
            
        struct_pooled = struct_enhanced.mean(dim=1)
        
        # 5. Gated fusion (text-led)
        gate_input = torch.cat([text_pooled, label_pooled, struct_pooled], dim=-1)
        gate_logits = self.modal_gate(gate_input)  # [batch, 3]
        gate_logits[:, 0] = gate_logits[:, 0] + self.text_bias  # Text bias
        gate_weights = F.softmax(gate_logits, dim=-1)  # [batch, 3]
        
        if return_attention:
            attention_weights = {
                'text_to_label': attn_t2l,
                'text_to_struct': attn_t2s,
                'label_self': attn_l,
                'modal_weights': gate_weights,
            }
        
        return text_pooled, label_pooled, struct_pooled, attention_weights


# =============================================================================
# Main Model with Configurable Fusion Strategy
# =============================================================================

class MultiModalComplaintModelWithFusion(nn.Module):
    """
    MultiModal Complaint Model with Configurable Fusion Strategy
    
    Supports three fusion types:
    - 'concat': Simple concatenation (baseline)
    - 'simple_attention': Standard attention without gating
    - 'gated': Gated cross-modal attention (our method)
    """
    
    def __init__(self, config, vocab_size, fusion_type='gated', 
                 pretrained_path=None, No_pretrain_bert=False):
        """
        Args:
            config: Configuration object
            vocab_size: Label vocabulary size
            fusion_type: 'concat', 'simple_attention', or 'gated'
            pretrained_path: Path to pretrained model
            No_pretrain_bert: Whether to use randomly initialized BERT
        """
        super().__init__()
        
        self.config = config
        self.fusion_type = fusion_type
        self.device = config.training.device
        
        print(f"🔧 Initializing model with fusion type: {fusion_type}")
        
        # ========== Text Encoder (BERT) ==========
        if No_pretrain_bert:
            print("🔄 Using randomly initialized BERT")
            from transformers import BertConfig
            bert_config = BertConfig.from_pretrained(config.model.bert_model_name)
            self.text_encoder = BertModel(bert_config)
        elif pretrained_path and os.path.exists(pretrained_path):
            print(f"✅ Loading domain-pretrained BERT: {pretrained_path}")
            self.text_encoder = BertModel.from_pretrained(pretrained_path)
        else:
            print("📦 Using original BERT pretrained weights")
            self.text_encoder = BertModel.from_pretrained(config.model.bert_model_name)
        
        # Text projection
        self.text_proj = nn.Linear(768, 256)
        
        # ========== Label Encoder (GAT) ==========
        self.label_encoder = GATLabelEncoder(
            vocab_size=vocab_size,
            hidden_dim=256,
            num_layers=3,
            num_heads=4
        )
        
        # ========== Struct Encoder ==========
        assert config.model.struct_feat_dim == 53, \
            f"Struct features must be 53-dim, got {config.model.struct_feat_dim}"
        
        self.struct_encoder = nn.Sequential(
            nn.Linear(53, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Feature importance weights
        self.feature_importance = nn.Parameter(torch.randn(53) * 0.1)
        
        # ========== Token Generators ==========
        self.text_token_gen = TextMultiTokenGenerator(
            bert_hidden_size=768, output_dim=256, num_tokens=4
        )
        self.struct_token_gen = StructMultiTokenGenerator(
            input_dim=53, output_dim=256, num_tokens=4
        )
        
        # ========== Fusion Module (Configurable) ==========
        if fusion_type == 'concat':
            self.fusion_module = ConcatFusion(dim=256)
        elif fusion_type == 'simple_attention':
            self.fusion_module = SimpleAttentionFusion(dim=256, num_heads=4, dropout=0.1)
        elif fusion_type == 'gated':
            self.fusion_module = GatedCrossAttentionFusion(dim=256, num_heads=4, dropout=0.1)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # ========== Fusion Layer ==========
        fusion_input_dim = 256 * 3  # text + label + struct
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.model.fusion_dim),
            nn.LayerNorm(config.model.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.fusion_dim, config.model.hidden_dim),
            nn.LayerNorm(config.model.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.model.dropout)
        )
        
        # ========== Classifier ==========
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(config.model.hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in param_name:
                        nn.init.zeros_(param)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        print("✅ Model weights initialized")
    
    def forward(self, input_ids=None, attention_mask=None,
                node_ids_list=None, edges_list=None, node_levels_list=None,
                struct_features=None, return_attention=False):
        """
        Forward pass
        
        Args:
            input_ids: Text input [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            node_ids_list: Node ID list (list of lists)
            edges_list: Edge list (list of lists)
            node_levels_list: Node level list (list of lists)
            struct_features: Structured features [batch, 53]
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Classification logits [batch, 2]
            attention_weights: Attention weights dict (if return_attention=True)
        """
        attention_weights = {}
        
        # ========== Text Features ==========
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        bert_hidden_states = text_output.hidden_states
        text_feat = text_output.last_hidden_state[:, 0, :]  # [batch, 768]
        text_feat_proj = self.text_proj(text_feat)  # [batch, 256]
        
        # Generate text tokens
        text_tokens = self.text_token_gen(bert_hidden_states)  # [batch, 4, 256]
        
        # ========== Label Features ==========
        batch_data = []
        for i in range(len(node_ids_list)):
            node_ids = torch.tensor(node_ids_list[i], dtype=torch.long, device=self.device)
            node_levels = torch.tensor(node_levels_list[i], dtype=torch.long, device=self.device)
            
            if edges_list[i]:
                edges = torch.tensor(edges_list[i], dtype=torch.long, device=self.device).t()
            else:
                num_nodes = len(node_ids)
                edges = torch.tensor([[j, j] for j in range(num_nodes)], device=self.device).t()
            
            data = Data(
                x=node_ids,
                edge_index=edges,
                node_levels=node_levels,
                batch=torch.full((len(node_ids),), i, dtype=torch.long, device=self.device)
            )
            batch_data.append(data)
        
        graph_batch = Batch.from_data_list(batch_data).to(self.device)
        label_feat = self.label_encoder(
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.node_levels,
            graph_batch.batch
        )
        
        # Process label features
        if isinstance(label_feat, tuple):
            label_node_feats, label_mask = label_feat
        else:
            label_node_feats = label_feat
            label_mask = None
        
        # ========== Struct Features ==========
        importance_weights = torch.softmax(self.feature_importance, dim=0)
        weighted_struct = struct_features * importance_weights
        struct_feat = self.struct_encoder(weighted_struct)  # [batch, 256]
        struct_tokens = self.struct_token_gen(struct_features)  # [batch, 4, 256]
        
        # ========== Cross-Modal Fusion ==========
        if self.fusion_type == 'concat':
            # For concat, we just pool the features
            text_pooled, label_pooled, struct_pooled, cross_attn = \
                self.fusion_module(
                    text_tokens, label_node_feats, struct_tokens,
                    label_mask=label_mask,
                    return_attention=return_attention
                )
        else:
            # For attention-based fusion
            text_pooled, label_pooled, struct_pooled, cross_attn = \
                self.fusion_module(
                    text_tokens, label_node_feats, struct_tokens,
                    label_mask=label_mask,
                    return_attention=return_attention
                )
        
        if return_attention and cross_attn:
            attention_weights.update(cross_attn)
        
        # ========== Final Fusion ==========
        combined_feat = torch.cat([text_pooled, label_pooled, struct_pooled], dim=-1)
        fused = self.fusion(combined_feat)
        
        # ========== Classification ==========
        logits = self.classifier(fused)
        
        if return_attention:
            return logits, attention_weights
        return logits


# =============================================================================
# Experiment Runner for Fusion Comparison
# =============================================================================

class FusionComparisonExperiment:
    """
    Run experiments comparing different fusion strategies
    """
    
    def __init__(self, config, vocab_size, data_processor, device='cuda'):
        self.config = config
        self.vocab_size = vocab_size
        self.data_processor = data_processor
        self.device = device
        self.results = {}
    
    def train_model(self, model, train_loader, val_loader, 
                    epochs=10, lr=2e-5, fusion_type='gated'):
        """Train a single model"""
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        model = model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        best_val_f1 = 0
        best_model_state = None
        
        print(f"\n{'='*60}")
        print(f"Training {fusion_type} fusion model")
        print(f"{'='*60}")
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_preds, train_labels = [], []
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                logits = model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    node_ids_list=batch['node_ids'],
                    edges_list=batch['edges'],
                    node_levels_list=batch['node_levels'],
                    struct_features=batch['struct_features'].to(self.device)
                )

                labels = batch['target'].to(self.device)
                loss = criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(logits.argmax(dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            scheduler.step()
            
            # Validation
            model.eval()
            val_preds, val_labels, val_probs = [], [], []
            
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(
                        input_ids=batch['input_ids'].to(self.device),
                        attention_mask=batch['attention_mask'].to(self.device),
                        node_ids_list=batch['node_ids'],
                        edges_list=batch['edges'],
                        node_levels_list=batch['node_levels'],
                        struct_features=batch['struct_features'].to(self.device)
                    )
                    
                    probs = F.softmax(logits, dim=1)[:, 1]
                    val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())
                    val_labels.extend(batch['target'].numpy())
            
            # Calculate metrics
            val_acc = accuracy_score(val_labels, val_preds)
            val_precision = precision_score(val_labels, val_preds, zero_division=0)
            val_recall = recall_score(val_labels, val_preds, zero_division=0)
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
            val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return model, {
            'accuracy': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'auc': val_auc
        }
    
    def run_comparison(self, train_loader, val_loader, test_loader=None, 
                       epochs=10, lr=2e-5):
        """
        Run comparison experiments for all fusion strategies
        
        Returns:
            results: Dict with metrics for each fusion type
        """
        fusion_types = ['concat', 'simple_attention', 'gated']
        
        for fusion_type in fusion_types:
            print(f"\n{'#'*60}")
            print(f"# Fusion Type: {fusion_type.upper()}")
            print(f"{'#'*60}")
            
            # Create model
            model = MultiModalComplaintModelWithFusion(
                config=self.config,
                vocab_size=self.vocab_size,
                fusion_type=fusion_type
            )
            
            # Train
            model, val_metrics = self.train_model(
                model, train_loader, val_loader,
                epochs=epochs, lr=lr, fusion_type=fusion_type
            )
            
            # Test if test_loader provided
            if test_loader:
                test_metrics = self.evaluate(model, test_loader)
                self.results[fusion_type] = {
                    'val': val_metrics,
                    'test': test_metrics
                }
            else:
                self.results[fusion_type] = {
                    'val': val_metrics
                }
            
            # Save model
            save_path = f"./models/fusion_{fusion_type}_model.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved: {save_path}")
        
        return self.results
    
    def evaluate(self, model, data_loader):
        """Evaluate model on dataset"""
        model.eval()
        preds, labels, probs = [], [], []
        
        with torch.no_grad():
            for batch in data_loader:
                logits = model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    node_ids_list=batch['node_ids'],
                    edges_list=batch['edges'],
                    node_levels_list=batch['node_levels'],
                    struct_features=batch['struct_features'].to(self.device)
                )
                
                prob = F.softmax(logits, dim=1)[:, 1]
                preds.extend(logits.argmax(dim=1).cpu().numpy())
                probs.extend(prob.cpu().numpy())
                labels.extend(batch['target'].numpy())
        
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
            'auc': roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0
        }
    
    def save_results(self, output_dir='./outputs/fusion_comparison'):
        """Save comparison results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON
        with open(os.path.join(output_dir, 'fusion_comparison_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save CSV
        import csv
        csv_path = os.path.join(output_dir, 'fusion_comparison_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Fusion Type', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
            
            for fusion_type, metrics in self.results.items():
                m = metrics.get('test', metrics.get('val', {}))
                writer.writerow([
                    fusion_type,
                    f"{m.get('accuracy', 0):.4f}",
                    f"{m.get('precision', 0):.4f}",
                    f"{m.get('recall', 0):.4f}",
                    f"{m.get('f1', 0):.4f}",
                    f"{m.get('auc', 0):.4f}"
                ])

        # Save Excel file
        excel_path = os.path.join(output_dir, 'fusion_comparison_results.xlsx')
        excel_rows = []
        for fusion_type, metrics in self.results.items():
            m = metrics.get('test', metrics.get('val', {}))
            excel_rows.append({
                'Fusion Type': fusion_type,
                'Accuracy': round(m.get('accuracy', 0), 4),
                'Precision': round(m.get('precision', 0), 4),
                'Recall': round(m.get('recall', 0), 4),
                'F1': round(m.get('f1', 0), 4),
                'AUC': round(m.get('auc', 0), 4),
            })
        df = pd.DataFrame(excel_rows)
        df.to_excel(excel_path, index=False, sheet_name='Fusion Comparison')
        print(f"  - {excel_path}")
        
        # Save LaTeX table
        tex_path = os.path.join(output_dir, 'fusion_comparison_table.tex')
        with open(tex_path, 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Comparison of Fusion Strategies}\n")
            f.write("\\label{tab:fusion_comparison}\n")
            f.write("\\begin{tabular}{lccccc}\n")
            f.write("\\toprule\n")
            f.write("Fusion Strategy & Accuracy & Precision & Recall & F1 & AUC \\\\\n")
            f.write("\\midrule\n")
            
            for fusion_type, metrics in self.results.items():
                m = metrics.get('test', metrics.get('val', {}))
                display_name = {
                    'concat': 'Concat (Baseline)',
                    'simple_attention': 'Simple Attention',
                    'gated': 'Gated Attention (Ours)'
                }.get(fusion_type, fusion_type)
                
                f.write(f"{display_name} & "
                       f"{m.get('accuracy', 0):.4f} & "
                       f"{m.get('precision', 0):.4f} & "
                       f"{m.get('recall', 0):.4f} & "
                       f"{m.get('f1', 0):.4f} & "
                       f"{m.get('auc', 0):.4f} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"\n✅ Results saved to {output_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function for standalone testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fusion Strategy Comparison')
    parser.add_argument('--data_file', type=str, default='小案例ai问询.xlsx',
                        help='Data file path')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--fusion_type', type=str, default='all',
                        choices=['concat', 'simple_attention', 'gated', 'all'],
                        help='Fusion type to train')
    args = parser.parse_args()
    
    print("="*60)
    print("Fusion Strategy Comparison Experiment")
    print("="*60)
    
    # Load config
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.training.device = device
    print(f"Device: {device}")
    
    # Load data processor - 兼容项目的ComplaintDataProcessor
    from data_processor import ComplaintDataProcessor, ComplaintDataset, custom_collate_fn
    from torch.utils.data import DataLoader
    import pandas as pd
    
    # 初始化processor
    user_dict_file = config.data.user_dict_file if hasattr(config.data, 'user_dict_file') else 'new_user_dict.txt'
    data_processor = ComplaintDataProcessor(
        config=config,
        user_dict_file=user_dict_file
    )
    
    # 尝试加载已保存的processor（包含全局词汇表）
    processor_path = os.path.join(config.training.pretrain_save_dir, 'processor.pkl')
    if os.path.exists(processor_path):
        try:
            data_processor.load(processor_path)
            print(f"✅ Loaded processor from {processor_path}")
        except Exception as e:
            print(f"⚠️ Could not load processor: {e}")
            # 如果加载失败，重新构建词汇表
            df = pd.read_excel(args.data_file) if args.data_file.endswith('.xlsx') else pd.read_csv(args.data_file)
            data_processor.build_global_ontology_tree(df['Complaint label'].tolist())
    else:
        # 构建词汇表
        df = pd.read_excel(args.data_file) if args.data_file.endswith('.xlsx') else pd.read_csv(args.data_file)
        data_processor.build_global_ontology_tree(df['Complaint label'].tolist())
    
    # 准备数据集
    data = data_processor.prepare_datasets(
        train_file=args.data_file,
        for_pretrain=False
    )
    
    # 划分数据集
    total_size = len(data['targets'])
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_size = total_size - train_size - val_size
    
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
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
            'targets': data['targets'][indices],
            'vocab_size': data['vocab_size']
        }
    
    train_data = split_data(data, train_indices)
    val_data = split_data(data, val_indices)
    test_data = split_data(data, test_indices)
    
    # 创建DataLoader
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
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    
    vocab_size = data['vocab_size']
    print(f"Label vocabulary size: {vocab_size}")
    print(f"Train/Val/Test: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")
    
    if args.fusion_type == 'all':
        # Run comparison for all fusion types
        experiment = FusionComparisonExperiment(
            config=config,
            vocab_size=vocab_size,
            data_processor=data_processor,
            device=device
        )
        
        results = experiment.run_comparison(
            train_loader, val_loader, test_loader,
            epochs=args.epochs, lr=args.lr
        )
        
        experiment.save_results()
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY: Fusion Strategy Comparison")
        print("="*60)
        for fusion_type, metrics in results.items():
            m = metrics.get('test', metrics.get('val', {}))
            print(f"\n{fusion_type.upper()}:")
            print(f"  Accuracy:  {m.get('accuracy', 0):.4f}")
            print(f"  Precision: {m.get('precision', 0):.4f}")
            print(f"  Recall:    {m.get('recall', 0):.4f}")
            print(f"  F1 Score:  {m.get('f1', 0):.4f}")
            print(f"  AUC:       {m.get('auc', 0):.4f}")
    else:
        # Train single fusion type
        model = MultiModalComplaintModelWithFusion(
            config=config,
            vocab_size=vocab_size,
            fusion_type=args.fusion_type
        )
        
        experiment = FusionComparisonExperiment(
            config=config,
            vocab_size=vocab_size,
            data_processor=data_processor,
            device=device
        )
        
        model, metrics = experiment.train_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr,
            fusion_type=args.fusion_type
        )
        
        print(f"\n{args.fusion_type.upper()} Results:")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUC:      {metrics['auc']:.4f}")


if __name__ == '__main__':
    main()
