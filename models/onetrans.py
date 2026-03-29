"""
OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer
================================================================================

arXiv 2025 — 腾讯

核心思想:

1. 将所有异构特征 (ID、数值、序列事件) 统一 tokenize 成一条序列
2. 在序列最前面加一个可学习的 [CLS] token
3. 用单一 Transformer encoder 对这条混合序列做 self-attention
4. 取 [CLS] 的输出做预测

与 UniScaleFormer 的关键差异:

* 不区分静态和序列，所有 token 平等参与 self-attention
* 复杂度 O((S + Ns*L)²)，序列很长时非常慢
* 没有 memory 压缩、没有分序列建模
* 没有 query 机制、没有 FM 头
* 优势: 极简设计，Scale Law 最直接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .tokenizer import UnifiedTokenizer
from .blocks import SelfAttentionBlock, FeedForwardBlock, RMSNorm

class OneTransLayer(nn.Module):
    """标准 Pre-Norm Transformer 块 = Self-Attention + FFN"""

def __init__(self, d_model, n_heads, dropout=0.0):
    super().__init__()
    self.attn = SelfAttentionBlock(d_model, n_heads, dropout)
    self.ff = FeedForwardBlock(d_model, 4, dropout)

def forward(self, x, mask=None):
    x = self.attn(x, mask)
    x = self.ff(x)
    return x

class OneTrans(nn.Module):
    """OneTrans 完整模型。

    
    架构图:
    ┌────────────────────────────────────────────────────────────┐
    │  [CLS] + static_tokens + seq_events (所有序列拼接)         │
    │    ↓                                                      │
    │  Unified Transformer Encoder × N layers                   │
    │    ↓                                                      │
    │  取 [CLS] 位置的输出 → MLP → logit                         │
    └────────────────────────────────────────────────────────────┘

    Token 排列示例 (S=128 static, 3 seq × L=128):
    [CLS, user_id, item_id, ts, uf_1, ..., if_n, act_1, ..., act_L, cont_1, ..., item_1, ...]
    总长度: 1 + S + Ns*L = 1 + 128 + 384 = 513
    """

def __init__(self, cfg):
    super().__init__()
    dc, mc = cfg['data'], cfg['model']
    self.d_model = int(mc['d_model'])
    self.num_sequences = len(dc.get('sequence_names', ['action_seq', 'content_seq', 'item_seq']))

    # 共享 tokenizer
    self.tokenizer = UnifiedTokenizer(
        int(mc['hash_size']),
        int(mc.get('feature_vocab_size', 16384)),
        int(mc.get('type_vocab_size', 16)),
        int(mc.get('seq_vocab_size', 16)),
        int(dc['max_seq_len']) + 8,
        self.d_model,
        int(dc['max_float_dim']),
        float(mc['dropout']),
    )

    # [CLS] token
    self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

    # 区分 segment 的 embedding: 0=CLS, 1=static, 2=seq
    self.segment_emb = nn.Embedding(4, self.d_model, padding_idx=0)

    # 统一 Transformer encoder
    self.layers = nn.ModuleList([
        OneTransLayer(self.d_model, int(mc['n_heads']), float(mc['dropout']))
        for _ in range(int(mc['n_layers']))
    ])

    self.final_norm = RMSNorm(self.d_model)

    # 预测头
    self.head = nn.Sequential(
        nn.LayerNorm(self.d_model),
        nn.Linear(self.d_model, int(mc['head_hidden'])),
        nn.GELU(),
        nn.Dropout(float(mc['dropout'])),
        nn.Linear(int(mc['head_hidden']), 1),
    )

def forward(self, batch):
    B = batch['static_token_ids'].size(0)
    device = batch['static_token_ids'].device

    # ── Step 1: 编码静态特征 ──
    static_tokens = self.tokenizer.encode_static(
        batch['static_token_ids'],
        batch['static_feature_ids'],
        batch['static_type_ids'],
        batch['static_float_values'],
    )  # [B, S, D]
    static_mask = batch['static_mask']  # [B, S]
    S = static_tokens.size(1)

    # ── Step 2: 编码序列特征 → 拼成一条长序列 ──
    seq = self.tokenizer.encode_sequence_events(
        batch['seq_token_ids'],
        batch['seq_feature_ids'],
        batch['seq_type_ids'],
        batch['seq_pos_ids'],
        batch['seq_name_ids'],
    )  # [B, Ns, L, D]

    Ns, L = seq.shape[1], seq.shape[2]
    seq_tokens = seq.reshape(B, Ns * L, D := self.d_model)  # [B, Ns*L, D]
    seq_mask = batch['seq_mask'].reshape(B, Ns * L)          # [B, Ns*L]

    # ── Step 3: 拼接 [CLS] + static + seq ──
    cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]

    tokens = torch.cat([cls, static_tokens, seq_tokens], dim=1)  # [B, 1+S+Ns*L, D]

    # Segment embedding
    seg_ids = torch.cat([
        torch.zeros(B, 1, dtype=torch.long, device=device),        # CLS = 0
        torch.ones(B, S, dtype=torch.long, device=device),         # static = 1
        torch.full((B, Ns * L), 2, dtype=torch.long, device=device),  # seq = 2
    ], dim=1)
    tokens = tokens + self.segment_emb(seg_ids)

    # Mask
    mask = torch.cat([
        torch.ones(B, 1, dtype=torch.bool, device=device),  # CLS 总是有效
        static_mask,
        seq_mask,
    ], dim=1)  # [B, 1+S+Ns*L]

    # ── Step 4: Transformer encoder ──
    for layer in self.layers:
        tokens = layer(tokens, mask)

    tokens = self.final_norm(tokens)

    # ── Step 5: 取 [CLS] → 预测 ──
    cls_output = tokens[:, 0]  # [B, D]
    logit = self.head(cls_output).squeeze(-1)  # [B]
    return {'logits': logit, 'probs': torch.sigmoid(logit)}

def compute_loss(self, batch, out):
    y = batch['targets']
    bce = F.binary_cross_entropy_with_logits(out['logits'], y)
    return {'loss': bce, 'bce_loss': bce.detach(), 'aux_loss': torch.tensor(0.0)}