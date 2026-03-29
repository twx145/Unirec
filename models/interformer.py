"""
InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction
============================================================================

CIKM 2025 — 腾讯

核心思想:

1. 将特征分为 Sequential 路 和 Non-Sequential 路
2. 每层内部: 各自 self-attention → 然后双向 cross-attention
3. 多层堆叠后拼接两路 pooled 表征 → MLP 预测

与 UniScaleFormer 的关键差异:

* 没有 memory 压缩 → 序列长时复杂度高
* 没有 query decoding → 用双向 cross-attn 替代
* 没有 FM 显式交叉
* 没有辅助对比损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .tokenizer import UnifiedTokenizer
from .blocks import (
RMSNorm, SelfAttentionBlock, FeedForwardBlock,
CrossAttentionBlock, SimpleMHA
)

class InterFormerLayer(nn.Module):
    """一层 InterFormer 交替更新块。

    ```
    信息流:
    seq_tokens  ──self-attn──→ seq'  ──┐
                                        ├── 双向 cross-attn ──→ seq'', static''
    static_tokens ─self-attn─→ static' ┘

    双向 cross-attention 的含义:
    seq''    = seq'    + CrossAttn(q=seq',    kv=static') — 序列看静态
    static'' = static' + CrossAttn(q=static', kv=seq')   — 静态看序列
    """

def __init__(self, d_model, n_heads, dropout=0.0):
    super().__init__()
    # 各自的 self-attention
    self.seq_self_attn = SelfAttentionBlock(d_model, n_heads, dropout)
    self.static_self_attn = SelfAttentionBlock(d_model, n_heads, dropout)

    # 双向 cross-attention
    self.seq_cross_attn = CrossAttentionBlock(d_model, n_heads, dropout)   # seq ← static
    self.static_cross_attn = CrossAttentionBlock(d_model, n_heads, dropout) # static ← seq

    # 各自的 FFN
    self.seq_ff = FeedForwardBlock(d_model, 4, dropout)
    self.static_ff = FeedForwardBlock(d_model, 4, dropout)

def forward(self, seq_tokens, seq_mask, static_tokens, static_mask):
    """
    输入:
      seq_tokens:    [B, L_seq, D]
      seq_mask:      [B, L_seq]
      static_tokens: [B, L_static, D]
      static_mask:   [B, L_static]

    输出:
      seq_tokens':    [B, L_seq, D]
      static_tokens': [B, L_static, D]
    """
    # Step 1: 各自 self-attention
    seq_tokens = self.seq_self_attn(seq_tokens, seq_mask)
    static_tokens = self.static_self_attn(static_tokens, static_mask)

    # Step 2: 双向 cross-attention
    #   seq 向 static 查询 → 序列 token 获得静态特征信息
    seq_updated = self.seq_cross_attn(seq_tokens, static_tokens, static_mask)
    #   static 向 seq 查询 → 静态 token 获得序列行为信息
    static_updated = self.static_cross_attn(static_tokens, seq_tokens, seq_mask)

    # Step 3: FFN
    seq_tokens = self.seq_ff(seq_updated)
    static_tokens = self.static_ff(static_updated)

    return seq_tokens, static_tokens

class InterFormer(nn.Module):
    """InterFormer 完整模型。

    ```
    架构图:
    ┌─────────────────────────────────────────────────────────┐
    │  Static Features → UnifiedTokenizer → static_tokens    │
    │  All Sequences   → UnifiedTokenizer → seq_tokens       │
    │                                                        │
    │  for layer in InterFormerLayers:                        │
    │    seq, static = layer(seq, static)  ← 双向交叉        │
    │                                                        │
    │  seq_pool = masked_mean(seq)                           │
    │  static_pool = masked_mean(static)                     │
    │  logit = MLP(cat[seq_pool, static_pool])               │
    └─────────────────────────────────────────────────────────┘
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

    # 序列 token 的前处理: 将多序列拼成一条长序列后编码
    self.seq_pre_encoder = nn.ModuleList([
        SelfAttentionBlock(self.d_model, int(mc['n_heads']), float(mc['dropout']))
        for _ in range(int(mc.get('seq_layers', 1)))
    ])

    # InterFormer 交替更新层
    self.layers = nn.ModuleList([
        InterFormerLayer(self.d_model, int(mc['n_heads']), float(mc['dropout']))
        for _ in range(int(mc['n_layers']))
    ])

    # 预测头
    head_in = self.d_model * 2
    self.head = nn.Sequential(
        nn.LayerNorm(head_in),
        nn.Linear(head_in, int(mc['head_hidden'])),
        nn.GELU(),
        nn.Dropout(float(mc['dropout'])),
        nn.Linear(int(mc['head_hidden']), 1),
    )

def _mean(self, x, mask):
    m = mask.float().unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

def forward(self, batch):
    # ── Step 1: 编码静态特征 ──
    static_tokens = self.tokenizer.encode_static(
        batch['static_token_ids'],
        batch['static_feature_ids'],
        batch['static_type_ids'],
        batch['static_float_values'],
    )  # [B, S, D]
    static_mask = batch['static_mask']  # [B, S]

    # ── Step 2: 编码序列特征 → 拼成一条长序列 ──
    seq = self.tokenizer.encode_sequence_events(
        batch['seq_token_ids'],
        batch['seq_feature_ids'],
        batch['seq_type_ids'],
        batch['seq_pos_ids'],
        batch['seq_name_ids'],
    )  # [B, Ns, L, D]

    B, Ns, L, D = seq.shape
    seq_tokens = seq.reshape(B, Ns * L, D)                    # [B, Ns*L, D]
    seq_mask = batch['seq_mask'].reshape(B, Ns * L)            # [B, Ns*L]

    # 序列前处理
    for blk in self.seq_pre_encoder:
        seq_tokens = blk(seq_tokens, seq_mask)

    # ── Step 3: N 层交替更新 ──
    for layer in self.layers:
        seq_tokens, static_tokens = layer(
            seq_tokens, seq_mask, static_tokens, static_mask
        )

    # ── Step 4: Pooling + 预测 ──
    seq_pool = self._mean(seq_tokens, seq_mask)       # [B, D]
    static_pool = self._mean(static_tokens, static_mask)  # [B, D]

    logit = self.head(torch.cat([seq_pool, static_pool], dim=-1)).squeeze(-1)  # [B]
    return {'logits': logit, 'probs': torch.sigmoid(logit)}

def compute_loss(self, batch, out):
    y = batch['targets']
    bce = F.binary_cross_entropy_with_logits(out['logits'], y)
    return {'loss': bce, 'bce_loss': bce.detach(), 'aux_loss': torch.tensor(0.0)}