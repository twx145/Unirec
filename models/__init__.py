"""
模型注册表
=====

通过配置文件中的 model.model_class 字段选择模型:

model:
model_class: UniScaleFormer   # 默认
# model_class: InterFormer
# model_class: OneTrans
# model_class: HyFormer
"""

from .uniscaleformer import UniScaleFormer
from .interformer import InterFormer
from .onetrans import OneTrans
from .hyformer import HyFormer

MODEL_REGISTRY = {
'UniScaleFormer': UniScaleFormer,
'InterFormer': InterFormer,
'OneTrans': OneTrans,
'HyFormer': HyFormer,
}

def build_model(cfg):
    """根据配置构建模型。

    ```
    输入: cfg — 完整配置 dict
    输出: nn.Module — 实例化的模型

    使用方式:
    model = build_model(cfg)
    等价于:
    model = UniScaleFormer(cfg)  # 当 model_class 未指定或为 UniScaleFormer 时
    """
    name = cfg['model'].get('model_class', 'UniScaleFormer')
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_class '{name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](cfg)