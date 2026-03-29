"""
HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR
==================================================================================

arXiv 2026 — 腾讯

核心思想:

1. 分序列建模: 每条行为序列独立用 self-attention 编码
2. Query Decoding: 可学习的 query token 通过 cross-attention 从各序列中提取兴趣
3. 静态特征路: 静态特征用 self-attention 编码
4. Query Boosting: query 表征反过来增强静态特征 (cross-attention)
5. 最终融合: query_pool + boosted_static_pool → MLP

与 UniScaleFormer 的关键差异:

* 没有 memory 压缩 → query 直接对原始序列做 cross-attention
* 没有交替更新 (InterFormer 的核心) → query decode 和 boosting 只各做一次
* 没有 FM 头和辅助损失
* 分序列 + query 机制是 HyFormer 的核心贡献
  """

import torch
import torch.nn as nn
import torch.nn.functional as F
from .tokenizer import UnifiedTokenizer
from .blocks import (
RMSNorm, SelfAttentionBlock, FeedForwardBlock,
CrossAttentionBlock
)

class QueryDecoder(nn.Module):
    """Query Decoding 模块。

    ```
    用可学习的 query token 从一条序列中提取兴趣信号。
    每条序列有独立的 QueryDecoder (参数不共享)。

    操作:
    query ──(cross-attn)──→ sequence ──→ decoded query
    decoded query ──(self-attn)──→ refined query
    refined query ──(FFN)──→ output
    """

def __init__(self, d_model, n_heads, num_queries, n_decode_layers=2, dropout=0.0):
    super().__init__()
    self.query_seed = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)

    self.decode_layers = nn.ModuleList()
    for _ in range(n_decode_layers):
        self.decode_layers.append(nn.ModuleDict({
            'cross': CrossAttentionBlock(d_model, n_heads, dropout),
            'self_attn': SelfAttentionBlock(d_model, n_heads, dropout),
            'ff': FeedForwardBlock(d_model, 4, dropout),
        }))

def forward(self, seq_tokens, seq_mask):
    """
    输入:
      seq_tokens: [B, L, D] — 编码后的序列 token
      seq_mask:   [B, L]    — 有效性 mask

    输出: [B, Q, D] — 解码后的 query token
    """
    B = seq_tokens.size(0)
    queries = self.query_seed.expand(B, -1, -1)  # [B, Q, D]

    for layer in self.decode_layers:
        queries = layer['cross'](queries, seq_tokens, seq_mask)
        queries = layer['self_attn'](queries)  # query 间 self-attn (无 mask)
        queries = layer['ff'](queries)

    return queries

class QueryBooster(nn.Module):
    """Query Boosting 模块。

    用 query 表征去增强/调制静态特征的表征。
    相当于 "我已经知道用户的兴趣了 (query)，用这个信息去重新理解物品特征 (static)"。

    操作:
    static ──(cross-attn, q=static, kv=query)──→ boosted_static
    boosted_static ──(FFN)──→ output
    """

def __init__(self, d_model, n_heads, dropout=0.0):
    super().__init__()
    self.cross = CrossAttentionBlock(d_model, n_heads, dropout)
    self.ff = FeedForwardBlock(d_model, 4, dropout)

def forward(self, static_tokens, queries):
    """
    输入:
      static_tokens: [B, S, D]
      queries:       [B, Q, D]  — 所有序列的 decoded query 拼接

    输出: [B, S, D] — 增强后的静态特征
    """
    # static 向 queries 查询 (queries 作为 KV)
    boosted = self.cross(static_tokens, queries)  # 无 mask，queries 全有效
    boosted = self.ff(boosted)
    return boosted

class HyFormer(nn.Module):
    """HyFormer 完整模型。

    架构图:
    ┌───────────────────────────────────────────────────────────────────┐
    │  Static Features → tokenizer → static_tokens                     │
    │                        ↓                                         │
    │                   static_encoder (self-attn × N)                 │
    │                        ↓                                         │
    │                   static_encoded                                  │
    │                                                                   │
    │  Seq 1 → tokenizer → seq_encoder → QueryDecoder₁ → queries₁     │
    │  Seq 2 → tokenizer → seq_encoder → QueryDecoder₂ → queries₂     │
    │  Seq 3 → tokenizer → seq_encoder → QueryDecoder₃ → queries₃     │
    │                                                                   │
    │  all_queries = cat[queries₁, queries₂, queries₃]                 │
    │                        ↓                                         │
    │  QueryBooster(static_encoded, all_queries) → boosted_static      │
    │                                                                   │
    │  query_pool = mean(all_queries)                                   │
    │  boosted_pool = masked_mean(boosted_static)                       │
    │  logit = MLP(cat[query_pool, boosted_pool])                       │
    └───────────────────────────────────────────────────────────────────┘
    """

def __init__(self, cfg):
    super().__init__()
    dc, mc = cfg['data'], cfg['model']
    self.d_model = int(mc['d_model'])
    self.num_sequences = len(dc.get('sequence_names', ['action_seq', 'content_seq', 'item_seq']))
    num_queries = int(mc['num_queries'])

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

    # 静态特征编码器
    self.static_encoder = nn.ModuleList([
        SelfAttentionBlock(self.d_model, int(mc['n_heads']), float(mc['dropout']))
        for _ in range(int(mc.get('static_layers', 1)))
    ])

    # 序列编码器 (所有序列共享参数，和 UniScaleFormer 一致)
    self.seq_encoder = nn.ModuleList([
        SelfAttentionBlock(self.d_model, int(mc['n_heads']), float(mc['dropout']))
        for _ in range(int(mc.get('seq_layers', 1)))
    ])

    # 每条序列独立的 QueryDecoder (不共享参数)
    n_decode = int(mc.get('n_decode_layers', 2))
    self.query_decoders = nn.ModuleList([
        QueryDecoder(self.d_model, int(mc['n_heads']), num_queries, n_decode, float(mc['dropout']))
        for _ in range(self.num_sequences)
    ])

    # Query Boosting
    self.booster = QueryBooster(self.d_model, int(mc['n_heads']), float(mc['dropout']))

    # 融合后的 self-attention (可选)
    self.fusion_layers = nn.ModuleList([
        SelfAttentionBlock(self.d_model, int(mc['n_heads']), float(mc['dropout']))
        for _ in range(int(mc.get('fusion_layers', 1)))
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

    for blk in self.static_encoder:
        static_tokens = blk(static_tokens, static_mask)

    # ── Step 2: 编码每条序列 + Query Decoding ──
    seq = self.tokenizer.encode_sequence_events(
        batch['seq_token_ids'],
        batch['seq_feature_ids'],
        batch['seq_type_ids'],
        batch['seq_pos_ids'],
        batch['seq_name_ids'],
    )  # [B, Ns, L, D]

    all_queries = []
    for s in range(self.num_sequences):
        x = seq[:, s]  # [B, L, D]
        m = batch['seq_mask'][:, s]  # [B, L]

        for blk in self.seq_encoder:
            x = blk(x, m)

        queries_s = self.query_decoders[s](x, m)  # [B, Q, D]
        all_queries.append(queries_s)

    all_queries = torch.cat(all_queries, dim=1)  # [B, Ns*Q, D]

    # ── Step 3: Query Boosting ──
    boosted_static = self.booster(static_tokens, all_queries)  # [B, S, D]

    # ── Step 4: 融合层 (可选) ──
    fused = torch.cat([all_queries, boosted_static], dim=1)  # [B, Ns*Q+S, D]
    fused_mask = torch.cat([
        torch.ones(all_queries.shape[:2], dtype=torch.bool, device=fused.device),
        static_mask,
    ], dim=1)

    for blk in self.fusion_layers:
        fused = blk(fused, fused_mask)

    # ── Step 5: Pooling + 预测 ──
    query_pool = all_queries.mean(dim=1)                 # [B, D]
    boosted_pool = self._mean(boosted_static, static_mask)  # [B, D]

    logit = self.head(torch.cat([query_pool, boosted_pool], dim=-1)).squeeze(-1)
    return {'logits': logit, 'probs': torch.sigmoid(logit)}

def compute_loss(self, batch, out):
    y = batch['targets']
    bce = F.binary_cross_entropy_with_logits(out['logits'], y)
    return {'loss': bce, 'bce_loss': bce.detach(), 'aux_loss': torch.tensor(0.0)}