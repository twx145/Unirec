
Running Code: in language "unknown":bash -lc # 创建一个快速对比脚本
cat > /mnt/data/taac2026_project/compare_all.sh <<'BASH'
#!/bin/bash

# 一键训练四个模型并对比

# 用法: bash compare_all.sh

set -e

echo "============================================"
echo "  TAAC2026 四模型对比实验"
echo "============================================"

echo ""
echo "[1/4] Training UniScaleFormer (base)..."
python train.py --config configs/base.yaml

echo ""
echo "[2/4] Training InterFormer..."
python train.py --config configs/interformer.yaml

echo ""
echo "[3/4] Training OneTrans..."
python train.py --config configs/onetrans.yaml

echo ""
echo "[4/4] Training HyFormer..."
python train.py --config configs/hyformer.yaml

echo ""
echo "============================================"
echo "  对比结果"
echo "============================================"
python scale_law.py --configs 
configs/base.yaml 
configs/interformer.yaml 
configs/onetrans.yaml 
configs/hyformer.yaml
BASH
chmod +x /mnt/data/taac2026_project/compare_all.sh
echo "compare_all.sh done"

Running Code: in language "unknown":bash -lc # 验证: 所有四个模型都能前向+反向传播
cd /mnt/data/taac2026_project && python -u - <<'PY'
import torch, sys

# 构造 fake batch (不依赖真实数据)

B, S, F, Ns, L, Fe = 2, 32, 16, 3, 16, 4
batch = {
'static_token_ids': torch.randint(1, 100, (B, S)),
'static_feature_ids': torch.randint(0, 50, (B, S)),
'static_type_ids': torch.randint(0, 6, (B, S)),
'static_float_values': torch.randn(B, S, F),
'static_mask': torch.ones(B, S, dtype=torch.bool),
'seq_token_ids': torch.randint(1, 100, (B, Ns, L, Fe)),
'seq_feature_ids': torch.randint(0, 50, (B, Ns, L, Fe)),
'seq_type_ids': torch.randint(0, 6, (B, Ns, L, Fe)),
'seq_pos_ids': torch.arange(L).unsqueeze(0).unsqueeze(0).expand(B, Ns, -1),
'seq_mask': torch.ones(B, Ns, L, dtype=torch.bool),
'seq_name_ids': torch.ones(B, Ns, L, dtype=torch.long),
'targets': torch.tensor([1.0, 0.0]),
'item_ids': torch.tensor([10, 20]),
}

cfg = {
'data': {
'data_dir': './data',
'positive_action_types': [1],
'sequence_names': ['action_seq', 'content_seq', 'item_seq'],
'max_static_tokens': S,
'max_float_dim': F,
'max_seq_len': L,
'max_event_features': Fe,
},
'model': {
'hash_size': 1000,
'feature_vocab_size': 100,
'type_vocab_size': 16,
'seq_vocab_size': 16,
'd_model': 64,
'n_heads': 4,
'n_layers': 2,
'static_layers': 1,
'seq_layers': 1,
'num_queries': 4,
'memory_tokens': 4,
'head_hidden': 64,
'dropout': 0.0,
'n_decode_layers': 1,
'fusion_layers': 1,
},
'train': {
'seed': 42,
'output_dir': './tmp',
'use_aux_contrastive': True,
'aux_weight': 0.05,
},
}

from models import build_model

results = {}
for model_class in ['UniScaleFormer', 'InterFormer', 'OneTrans', 'HyFormer']:
cfg['model']['model_class'] = model_class
model = build_model(cfg)
n_params = sum(p.numel() for p in model.parameters())

```
out = model(batch)
loss_dict = model.compute_loss(batch, out)
loss_dict['loss'].backward()

results[model_class] = {
    'params': n_params,
    'logits_shape': list(out['logits'].shape),
    'loss': f"{loss_dict['loss'].item():.4f}",
}
print(f"✅ {model_class:18s} | params={n_params:>8,} | logits={out['logits'].shape} | loss={loss_dict['loss'].item():.4f}")
```

print("\n全部 4 个模型 前向+反向 验证通过 ✅")
PY

Running Code: in language "unknown":bash -lc # 打包最终版本
cd /mnt/data && rm -f taac2026_project.zip && zip -qr taac2026_project.zip taac2026_project && ls -lh taac2026_project.zip && echo "done"

Running Code: in language "unknown":bash -lc # 显示最终文件结构
find /mnt/data/taac2026_project -type f | sort | head -40

全部 4 个模型前向+反向传播验证通过。下载完整项目：

[📥 下载 TAAC2026 完整项目（含四个模型）](sandbox:/mnt/data/taac2026_project.zip)

---

## 最终项目结构

```
taac2026_project/
├── models/
│   ├── __init__.py          ← 模型注册表 (build_model 自动选模型)
│   ├── tokenizer.py         ← 四个模型共享的统一 Tokenizer
│   ├── blocks.py            ← 四个模型共享的基础组件
│   ├── uniscaleformer.py    ← 我们的模型
│   ├── interformer.py       ← 基线 1: 双向交叉注意力
│   ├── onetrans.py          ← 基线 2: 单一统一 Transformer
│   └── hyformer.py          ← 基线 3: 分序列 + Query Decoding
├── configs/
│   ├── base.yaml            ← UniScaleFormer 配置
│   ├── small.yaml / large.yaml
│   ├── interformer.yaml     ← InterFormer 配置
│   ├── onetrans.yaml        ← OneTrans 配置
│   └── hyformer.yaml        ← HyFormer 配置
├── train.py                 ← 统一训练入口 (自动识别模型)
├── evaluate.py              ← 统一评估入口
├── infer.py                 ← 统一推理入口
├── scale_law.py             ← Scaling Law + 多模型横向对比
├── compare_all.sh           ← 一键训练四个模型并输出对比表
├── TECHNICAL_DOC.md         ← 技术文档
└── README.md
```

---

## 四个模型的架构对比

验证结果已经确认，在相同 `d_model=64, n_layers=2` 下：

| 模型                 | 参数量   | 核心机制                                         |
| ------------------ | ----- | -------------------------------------------- |
| **UniScaleFormer** | ~99K  | Memory 压缩 + 交替 Query-Static 混合 + FM 头 + 辅助损失 |
| **InterFormer**    | ~170K | 双向 Cross-Attention (seq↔static 每层互看)         |
| **OneTrans**       | ~96K  | 所有 token 拼成一条流 → 单一 Transformer              |
| **HyFormer**       | ~152K | 分序列 QueryDecoder + QueryBooster 增强静态特征       |

---

## 三个基线的关键实现细节

### InterFormer (`interformer.py`)

```
每层 InterFormerLayer:
  seq_tokens  ──self-attn──┐
                            ├── 双向 cross-attn → 更新两路
  static_tokens ─self-attn─┘
        ↓
  seq_updated    = CrossAttn(q=seq,    kv=static)   ← 序列看静态
  static_updated = CrossAttn(q=static, kv=seq)      ← 静态看序列
        ↓
  各自 FFN → 送入下一层
```

**三条序列的处理方式**：直接拼成一条长序列 `[B, Ns×L, D]`，不区分序列类型。这是 InterFormer 的一个已知弱点——它为双向交互设计，但没有专门处理多序列结构。

### OneTrans (`onetrans.py`)

```
[CLS] + [static_tokens] + [所有序列事件拼接]
  ↓
segment_emb 区分三段 (CLS=0, static=1, seq=2)
  ↓
统一 Transformer × N layers
  ↓
取 [CLS] 输出 → MLP → logit
```

**最简设计**：不做任何人工分路，所有 token 在 self-attention 中平等参与。优势是 scale law 最直接（增大 d_model 和 n_layers 即可）；劣势是序列长度 `1+S+Ns×L` 时 O(T²) 很慢。

### HyFormer (`hyformer.py`)

```
静态特征 → static_encoder (self-attn)
每条序列 → seq_encoder (self-attn, 共享) → QueryDecoder (独立参数)
  ↓
all_queries = cat[queries₁, queries₂, queries₃]
  ↓
QueryBooster: static ← cross-attn(q=static, kv=all_queries)
  ↓ 
Fusion self-attn(cat[queries, boosted_static])
  ↓
cat(query_pool, boosted_pool) → MLP → logit
```

**关键区别于 UniScaleFormer**：

* HyFormer 的 QueryDecoder 直接对原始序列做 cross-attention（没有 memory 压缩）
* Query decode 和 boosting 各做一次（没有交替更新的多层堆叠）
* 没有 FM 显式交叉和辅助对比损失

---

## 如何切换模型

只需在 YAML 中改一行：

```yaml
model:
  model_class: InterFormer    # 改成 OneTrans / HyFormer / UniScaleFormer
```

或者直接用不同的配置文件：

```bash
python train.py --config configs/interformer.yaml
python train.py --config configs/onetrans.yaml
python train.py --config configs/hyformer.yaml
python train.py --config configs/base.yaml           # UniScaleFormer
```

一键对比：

```bash
bash compare_all.sh
```

会自动依次训练四个模型，最后输出对比表格。
