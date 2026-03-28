# TAAC2026 UniScaleFormer

一个面向 TAAC2026 的可训练项目模板，核心模型为 **UniScaleFormer**：

- **统一 tokenizer**：将 `item_feature / user_feature / seq_feature / item_id / user_id / timestamp` 统一转成 token。
- **分类型 value adapter**：`int_value/int_array` 走哈希 embedding，`float_value/float_array` 走 MLP。
- **多序列 memory 压缩**：每条序列先编码，再压缩成 memory token，兼顾长序列与时延约束。
- **Interleaved Query Block**：用 target-aware queries 对序列 memory 做 cross-attention，再和静态 token 交替混合。
- **Hybrid Head**：Transformer pooled representation + FM 显式交叉。

## 训练

```bash
pip install -r requirements.txt
python train.py --config configs/base.yaml
```

## 评估

```bash
python evaluate.py --config configs/base.yaml --checkpoint outputs/base/best.pt
```

## 推理

```bash
python infer.py --config configs/base.yaml --checkpoint outputs/base/best.pt --split test --output outputs/base/test_predictions.csv
```

## scaling law

```bash
python scale_law.py --configs configs/small.yaml configs/base.yaml configs/large.yaml
```

## 数据假设

当前代码按你给出的 schema 实现：

- `item_id`
- `item_feature`
- `label`
- `seq_feature`
- `timestamp`
- `user_feature`
- `user_id`

默认将 `positive_action_types=[1]` 当作正样本。若官方正式赛定义不同，只需改 `configs/*.yaml`。
