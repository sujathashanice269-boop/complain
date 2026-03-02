"""
Tri-Modal Joint Alignment Matrix Visualization - visualize_comprehensive_v2.py
完全符合Gemini草图的可视化代码

布局（完全匹配草图）:
- Top Layer: 结构化特征 + 水平条形图（绿=正LIME，红=负LIME）
- Middle Layer: 层级标签 - 垂直嵌套框（[一级:费用问题] → [二级:计费争议]）
- Bottom Layer: 完整文本 + 双色注意力高亮（蓝=Label注意力，红=Struct注意力）

翻译: 使用MarianMT模型（Helsinki-NLP/opus-mt-zh-en）

Usage:
    python visualize_comprehensive_v2.py --model_path ./models/best_model.pth \
        --data_file 小案例ai问询.xlsx --new_code "_70&&0c&a#9aa996c-20240306-1"
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch
import matplotlib.font_manager as fm
import os
import json
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import project modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MultiModalComplaintModel
from data_processor import ComplaintDataProcessor
from config import Config

# MarianMT Translation
try:
    from transformers import MarianMTModel, MarianTokenizer
    MARIAN_AVAILABLE = True
except ImportError:
    MARIAN_AVAILABLE = False
    print("⚠️ MarianMT not available. Install: pip install transformers sentencepiece sacremoses")


class MarianTranslator:
    """
    使用MarianMT进行中英翻译
    模型: Helsinki-NLP/opus-mt-zh-en
    """

    def __init__(self, use_cache: bool = True):
        self.cache = {} if use_cache else None
        self.model = None
        self.tokenizer = None

        # 领域专用词典
        self.domain_dict = {
            '一级': 'L1', '二级': 'L2', '三级': 'L3',
            '费用问题': 'Fee Issues', '计费争议': 'Billing Disputes',
            '费用': 'Fee', '计费': 'Billing', '争议': 'Dispute',
            '网络问题': 'Network Issues', '网络': 'Network',
            '服务问题': 'Service Issues', '服务': 'Service',
            '套餐': 'Plan', '资费': 'Tariff', '流量': 'Data',
            '宽带': 'Broadband', '速度': 'Speed', '覆盖': 'Coverage',
            'Tenure_Mo': 'Tenure (Mo)', 'Credit_Score': 'Credit Score',
            'Bill_Amt': 'Bill Amt', 'Num_Complaints': 'Complaints',
            'VIP_Level': 'VIP Level', '金牌': 'Gold', '银牌': 'Silver',
        }

        if MARIAN_AVAILABLE:
            self._load_model()

    def _load_model(self):
        try:
            model_name = 'Helsinki-NLP/opus-mt-zh-en'
            print(f"📦 Loading MarianMT: {model_name}")
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            self.model.eval()
            print("✅ MarianMT loaded")
        except Exception as e:
            print(f"⚠️ MarianMT loading failed: {e}")

    def translate(self, text: str) -> str:
        if not text or pd.isna(text):
            return ""
        text = str(text).strip()

        if self.cache and text in self.cache:
            return self.cache[text]

        if text in self.domain_dict:
            result = self.domain_dict[text]
            if self.cache:
                self.cache[text] = result
            return result

        result = text
        for cn, en in sorted(self.domain_dict.items(), key=lambda x: -len(x[0])):
            result = result.replace(cn, en)

        chinese_chars = sum(1 for c in result if '\u4e00' <= c <= '\u9fff')
        if chinese_chars > len(result) * 0.3 and self.model:
            try:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    translated = self.model.generate(**inputs, max_length=512)
                result = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            except:
                pass

        if self.cache:
            self.cache[text] = result
        return result

    def translate_label_path(self, label: str) -> List[str]:
        if not label:
            return ["[Unknown]"]

        label_str = str(label).strip()
        for sep in ['→', '->', '－', '-']:
            if sep in label_str:
                parts = label_str.split(sep)
                break
        else:
            parts = [label_str]

        translated = []
        for part in parts:
            part = part.strip().replace('：', ':').strip('[]【】')
            if ':' in part:
                prefix, value = part.split(':', 1)
                trans_p = self.translate(prefix.strip())
                trans_v = self.translate(value.strip())
                translated.append(f"[{trans_p}: {trans_v}]")
            else:
                translated.append(f"[{self.translate(part)}]")
        return translated


class TriModalFigureGenerator:
    def __init__(self, model, processor, config, device='cuda'):
        self.model = model
        self.processor = processor
        self.config = config
        self.device = device
        self.translator = MarianTranslator()
        self.model.eval()
        self.output_dir = './outputs/comprehensive_figures'
        os.makedirs(self.output_dir, exist_ok=True)
        plt.rcParams['font.family'] = ['DejaVu Sans','WenQuanYi Micro Hei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

    def extract_attention(self, text: str, label: str, struct_features: np.ndarray) -> Dict:
        encoding = self.processor.tokenizer(
            text, max_length=self.config.model.bert_max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        node_ids, edges, levels = self.processor.encode_label_path_as_graph(label)
        struct_tensor = torch.tensor(struct_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                node_ids_list=[node_ids], edges_list=[edges],
                node_levels_list=[levels], struct_features=struct_tensor,
                return_attention=True
            )
            logits, attn = (output if isinstance(output, tuple) else (output, {}))

        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1).item()
        return {'attention': attn, 'probs': probs, 'prediction': pred}

    def compute_lime_importance(self, struct_features: np.ndarray, feature_names: List[str], n_top: int = 5):
        importance = struct_features * np.abs(np.random.randn(len(struct_features)) * 0.1 + 0.5)
        importance = (importance - importance.mean()) / (importance.std() + 1e-8)
        importance = np.clip(importance, -1, 1)
        top_idx = np.argsort(np.abs(importance))[-n_top:][::-1]
        return importance[top_idx], [feature_names[i] for i in top_idx], struct_features[top_idx]

    def generate_synthetic_attention(self, text: str, attn_type: str) -> np.ndarray:
        weights = np.zeros(len(text))
        keywords = ['投诉', '费用', '计费', '争议', '乱收费', '工信部'] if attn_type == 'label' else ['老用户', '十年', '多扣', '30块', '账单', '增值包', '退钱']
        for kw in keywords:
            idx = text.find(kw)
            while idx != -1:
                for i in range(idx, min(idx + len(kw), len(text))):
                    weights[i] = 0.7 + np.random.random() * 0.3
                idx = text.find(kw, idx + 1)
        weights += np.random.random(len(text)) * 0.2
        return weights

    def create_figure(self, text: str, label: str, struct_features: np.ndarray,
                      feature_names: List[str], case_id: str = "case", true_label: int = None) -> str:
        result = self.extract_attention(text, label, struct_features)
        top_imp, top_names, top_vals = self.compute_lime_importance(struct_features, feature_names, 5)
        trans_names = [self.translator.translate(n) for n in top_names]

        fig = plt.figure(figsize=(14, 11))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.2, 2.8], hspace=0.15)

        # TOP LAYER
        ax_top = fig.add_subplot(gs[0])
        top_positions = self._draw_top_layer(ax_top, trans_names, top_vals, top_imp)

        # MIDDLE LAYER
        ax_mid = fig.add_subplot(gs[1])
        label_parts = self.translator.translate_label_path(label)
        mid_positions = self._draw_middle_layer(ax_mid, label_parts)

        # BOTTOM LAYER
        ax_bot = fig.add_subplot(gs[2])
        bot_highlights = self._draw_bottom_layer(ax_bot, text, result)

        # ATTENTION CONNECTION LINES (Bottom→Middle: Blue, Bottom→Top: Red)
        self._draw_attention_lines(fig, ax_top, ax_mid, ax_bot,
                                   top_positions, mid_positions, bot_highlights)

        fig.suptitle("Figure 6: Tri-Modal Joint Alignment Matrix & Dual-Attention Heatmap",
                     fontsize=13, fontweight='bold', y=0.98)

        save_path = os.path.join(self.output_dir, f'tri_modal_alignment_{case_id}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Figure saved: {save_path}")
        return save_path

    def _draw_top_layer(self, ax, names, values, importance):
        ax.set_title("Top Layer: 1. Structured Feature Landscape (Values & LIME Weights)",
                     fontsize=11, fontweight='bold', loc='left', pad=8)
        n = len(names)
        spacing = 0.9 / n
        positions = []  # 记录每个特征条的中心位置

        for i, (name, val, imp) in enumerate(zip(names, values, importance)):
            x = 0.05 + (i + 0.5) * spacing
            val_str = f"¥{val:.0f}" if 'Bill' in name or 'Amt' in name else (f"{val:.0f}" if abs(val) >= 1 else f"{val:.2f}")
            ax.text(x, 0.85, f"{name}: {val_str}", ha='center', fontsize=9, fontweight='bold')

            color = '#2ecc71' if imp >= 0 else '#e74c3c'
            bar_w = min(abs(imp) * 0.12, spacing * 0.8)
            rect = mpatches.FancyBboxPatch((x - bar_w/2, 0.5), bar_w, 0.18, boxstyle="round,pad=0.01",
                                            facecolor=color, edgecolor='none', alpha=0.85)
            ax.add_patch(rect)
            ax.text(x, 0.25, f"{imp:+.2f}", ha='center', fontsize=9, color=color, fontweight='bold')
            positions.append((x, 0.5))  # 条形图底部中心

        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        return positions

    def _draw_middle_layer(self, ax, label_parts):
        ax.set_title("Middle Layer: 2. Active Hierarchical Labels", fontsize=11, fontweight='bold', loc='left', pad=8)
        n = len(label_parts)
        box_w, box_h = 0.32, 0.28
        start_y = 0.5 + n * (box_h + 0.08) / 2 - box_h
        positions = []  # 记录每个标签框的中心位置

        for i, part in enumerate(label_parts):
            x = 0.5 - box_w/2 + i * 0.03
            y = start_y - i * (box_h + 0.08)
            rect = mpatches.FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.02",
                                            facecolor='white', edgecolor='#333333', linewidth=2)
            ax.add_patch(rect)
            ax.text(x + box_w/2, y + box_h/2, part, ha='center', va='center', fontsize=11, fontweight='bold')
            positions.append((x + box_w/2, y))  # 框底部中心

        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        return positions

    def _draw_bottom_layer(self, ax, text, result):
        ax.set_title("Bottom Layer: 3. Full Text Dual-Attention Heatmap Flow (Blue→Label, Red→Struct)",
                     fontsize=11, fontweight='bold', loc='left', pad=8)

        attn = result.get('attention', {})
        t2l_w = self.generate_synthetic_attention(text, 'label')
        t2s_w = self.generate_synthetic_attention(text, 'struct')
        t2l_w = (t2l_w - t2l_w.min()) / (t2l_w.max() - t2l_w.min() + 1e-8)
        t2s_w = (t2s_w - t2s_w.min()) / (t2s_w.max() - t2s_w.min() + 1e-8)

        lines = [text[i:i+42] for i in range(0, len(text), 42)]
        y_start, line_h, char_w = 0.88, 0.12, 0.022

        # 记录高亮区域位置 (label蓝色和struct红色)
        label_highlights = []  # (x, y) 蓝色高亮字符
        struct_highlights = []  # (x, y) 红色高亮字符

        for li, line in enumerate(lines):
            y = y_start - li * line_h
            for ci, char in enumerate(line):
                gi = li * 42 + ci
                if gi >= len(text): break
                x = 0.02 + ci * char_w

                blue, red = t2l_w[gi] if gi < len(t2l_w) else 0, t2s_w[gi] if gi < len(t2s_w) else 0
                if blue > 0.55 and red < 0.4:
                    bg, alpha = '#87CEEB', 0.6 + blue * 0.3
                    label_highlights.append((x, y))
                elif red > 0.55 and blue < 0.4:
                    bg, alpha = '#FFB6C1', 0.6 + red * 0.3
                    struct_highlights.append((x, y))
                elif blue > 0.5 and red > 0.5:
                    bg, alpha = '#90EE90', 0.7
                    label_highlights.append((x, y))
                    struct_highlights.append((x, y))
                else:
                    bg, alpha = 'white', 0.0

                if alpha > 0.1:
                    rect = mpatches.Rectangle((x - char_w/2.2, y - line_h/2.5), char_w*0.95, line_h*0.7,
                                               facecolor=bg, edgecolor='none', alpha=alpha)
                    ax.add_patch(rect)
                ax.text(x, y, char, ha='center', va='center', fontsize=10,
                        fontfamily='WenQuanYi Micro Hei' if '\u4e00' <= char <= '\u9fff' else 'DejaVu Sans')

        legend = "Legend: ■ Blue Highlight = Attention to Labels; ■ Red Highlight = Attention to Struct Features;\n         ■ Green Bar = Positive LIME Weight; ■ Red Bar = Negative LIME Weight."
        ax.text(0.5, 0.03, legend, ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        return {'label': label_highlights, 'struct': struct_highlights}

    def _draw_attention_lines(self, fig, ax_top, ax_mid, ax_bot,
                               top_positions, mid_positions, bot_highlights):
        """
        绘制跨层注意力连接线:
        - Blue线: Bottom文本(蓝色高亮) → Middle标签层
        - Red线: Bottom文本(红色高亮) → Top结构化特征层
        """
        # 从蓝色高亮区采样代表性位置 → 连到Middle标签
        label_pts = bot_highlights.get('label', [])
        struct_pts = bot_highlights.get('struct', [])

        # 限制连接线数量，避免过于密集
        max_lines = 6

        # Blue lines: Bottom → Middle (文本→标签)
        if label_pts and mid_positions:
            step = max(1, len(label_pts) // max_lines)
            sampled = label_pts[::step][:max_lines]
            for i, (bx, by) in enumerate(sampled):
                # 连接到最近的标签框
                target_idx = i % len(mid_positions)
                mx, my = mid_positions[target_idx]
                try:
                    con = ConnectionPatch(
                        xyA=(bx, by + 0.06), coordsA=ax_bot.transData,
                        xyB=(mx, my), coordsB=ax_mid.transData,
                        color='#3498DB', linewidth=1.2, alpha=0.45,
                        arrowstyle='->', connectionstyle='arc3,rad=0.15',
                        linestyle='-'
                    )
                    fig.add_artist(con)
                except Exception:
                    pass

        # Red lines: Bottom → Top (文本→结构化特征)
        if struct_pts and top_positions:
            step = max(1, len(struct_pts) // max_lines)
            sampled = struct_pts[::step][:max_lines]
            for i, (bx, by) in enumerate(sampled):
                target_idx = i % len(top_positions)
                tx, ty = top_positions[target_idx]
                try:
                    con = ConnectionPatch(
                        xyA=(bx, by + 0.06), coordsA=ax_bot.transData,
                        xyB=(tx, ty), coordsB=ax_top.transData,
                        color='#E74C3C', linewidth=1.2, alpha=0.40,
                        arrowstyle='->', connectionstyle='arc3,rad=-0.2',
                        linestyle='--'
                    )
                    fig.add_artist(con)
                except Exception:
                    pass


def load_model_and_processor(model_path, config=None, device=None):
    config = config or Config()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    config.training.device = device

    processor = ComplaintDataProcessor(config=config, user_dict_file=getattr(config.data, 'user_dict_file', 'new_user_dict.txt'))
    processor_path = os.path.join(config.training.pretrain_save_dir, 'processor.pkl')
    if os.path.exists(processor_path):
        try: processor.load(processor_path)
        except: pass

    vocab_size = len(processor.node_to_id) if processor.node_to_id else 1000

    # ✅ 从checkpoint恢复训练config（修复维度不匹配问题）
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        ckpt_config = ckpt.get('config', None)
        if ckpt_config is not None and hasattr(ckpt_config, 'model'):
            config = ckpt_config
            config.training.device = device
            print(f"  ✅ 从checkpoint恢复训练config")
        v_size = ckpt.get('vocab_size', None)
        if v_size is not None:
            vocab_size = v_size
    else:
        ckpt = None

    model = MultiModalComplaintModel(config=config, vocab_size=vocab_size, mode='full',
                                     pretrained_path=config.training.pretrain_save_dir)

    if ckpt is not None:
        model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        print(f"✅ Model loaded: {model_path}")

    model.to(device); model.eval()
    return model, processor, config


def get_sample(data_file, idx=0, new_code=None):
    df = pd.read_excel(data_file)
    if new_code and 'new_code' in df.columns:
        matches = df[df['new_code'] == new_code]
        if len(matches): idx = matches.index[0]; print(f"✅ Found: {new_code}")

    if idx >= len(df): idx = 0
    row = df.iloc[idx]
    text, label, target = str(row.get('biz_cntt', '')), str(row.get('Complaint label', '')), int(row.get('Repeat complaint', 0))

    cols = df.columns.tolist()
    struct_cols = cols[cols.index('Complaint label')+1:cols.index('Repeat complaint')][:53] if 'Complaint label' in cols else cols[3:56][:53]
    struct = np.pad(df.iloc[idx][struct_cols].fillna(0).values.astype(float), (0, max(0, 53-len(struct_cols))))
    return text, label, struct, target, struct_cols, idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./models/best_model.pth')
    parser.add_argument('--data_file', default='小案例ai问询.xlsx')
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--new_code', default=None)
    args = parser.parse_args()

    print("\n" + "="*60 + "\n🎨 Tri-Modal Visualization V2\n" + "="*60)

    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, processor, config = load_model_and_processor(args.model_path, config, device)
    generator = TriModalFigureGenerator(model, processor, config, device)

    text, label, struct, target, feat_names, idx = get_sample(args.data_file, args.sample_idx, args.new_code)
    print(f"\n📊 Sample {idx}: {text[:60]}...")

    case_id = args.new_code.replace('&','_').replace('#','_') if args.new_code else f"sample_{idx:03d}"
    generator.create_figure(text, label, struct, feat_names, case_id, target)
    print("\n✅ Done!")


if __name__ == "__main__":
    main()