# -*- coding: utf-8 -*-
"""
OPERA/opera_data/label_schema.py

使用 Pydantic 定义和验证数据标签的结构。
这为我们生成的所有训练数据提供了一个统一、可靠的“真值”格式。
(已更新至 Pydantic V2 语法)
"""
from enum import Enum
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field, ValidationError


# 1. 定义可控的枚举类型
class CompositionType(str, Enum):
    """定义了现象可以被组合的方式"""
    SEQUENTIAL = 'sequential'
    SUPERPOSITION = 'superposition'


# 2. 定义描述单个现象组件的模型
class PhenomenonComponent(BaseModel):
    """描述一个基础光学现象及其参数"""
    phenomenon_name: str = Field(
        ...,  # '...' 表示这是一个必需字段
        description="用于生成现象的函数名 (例如, 'generate_airy_disk')."
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="一个包含现象函数所需参数的字典。"
    )


# 3. 定义描述噪声和畸变配置的模型
class NoiseProfile(BaseModel):
    """描述应用于图像的噪声和畸变的配置"""
    distortion_k1: Optional[float] = Field(None, description="桶形/枕形畸变系数 k1.")
    gauss_std: Optional[float] = Field(None, description="高斯读出噪声的标准差。")
    poisson_strength: Optional[float] = Field(None, description="泊松散粒噪声的强度。")
    salt_pepper_amount: Optional[float] = Field(None, description="椒盐噪声的比例。")
    bit_depth: Optional[int] = Field(None, description="传感器的量化位深度。")
    saturation_level: Optional[float] = Field(1.0, description="传感器的饱和水平 [0, 1]。")


# 4. 定义顶层的、完整的标签模型
class Label(BaseModel):
    """
    定义了一个图像的完整标签，包含了生成它的所有信息。
    这是我们神经网络需要学习预测的目标。
    """
    image_id: str = Field(..., description="图像的唯一标识符。")
    resolution_px: int = Field(..., description="图像的方形分辨率。")
    composition_type: CompositionType = Field(..., description="现象的组合方式。")

    components: List[PhenomenonComponent] = Field(
        ...,
        description="构成此图像的基础现象组件列表。"
    )

    noise_profile: NoiseProfile = Field(
        default_factory=NoiseProfile,
        description="应用于此图像的噪声配置。"
    )


# 当该文件被直接执行时，运行以下测试代码
if __name__ == '__main__':
    print("--- Testing Label Schema with Pydantic (V2 Syntax) ---")

    # 1. 创建一个符合schema的Python字典 (一个复杂的例子)
    print("\n[1] Creating a valid sample label dictionary...")
    valid_data = {
        'image_id': 'exp_0001',
        'resolution_px': 256,
        'composition_type': 'sequential',  # 或 'superposition'
        'components': [
            {
                'phenomenon_name': 'generate_airy_disk',
                'params': {'aperture_diameter_mm': 1.0, 'wavelength_nm': 550.0}
            },
            {
                'phenomenon_name': 'apply_zernike_aberration',
                'params': {'zernike_coeffs': {11: 0.75}}  # 球差
            }
        ],
        'noise_profile': {
            'gauss_std': 0.05,
            'bit_depth': 8
        }
    }

    try:
        # 2. 【修正处】使用 Pydantic V2 的 model_validate 方法
        label_obj = Label.model_validate(valid_data)
        print("    -> SUCCESS: Dictionary was successfully parsed and validated.")

        # 3. 访问和使用模型对象
        print(f"    -> Accessing data: Image ID = {label_obj.image_id}")
        print(f"    -> Accessing nested data: Aberration coeffs = {label_obj.components[1].params['zernike_coeffs']}")

        # 4. 【修正处】使用 Pydantic V2 的 model_dump_json 方法
        print("\n[2] Exporting the validated object back to a pretty JSON string:")
        print(label_obj.model_dump_json(indent=2))

    except ValidationError as e:
        print(f"    -> FAILURE: Validation failed. Details:\n{e}")

    # --- 测试一个无效的例子 ---
    print("\n[3] Testing with an invalid dictionary (missing required field)...")
    invalid_data = {
        'image_id': 'exp_0002',
        # 'resolution_px': 256, # <-- 故意缺少 'resolution_px'
        'composition_type': 'superposition',
        'components': []
    }

    try:
        # 【修正处】使用 Pydantic V2 的 model_validate 方法
        Label.model_validate(invalid_data)
    except ValidationError as e:
        print("    -> SUCCESS: Pydantic correctly caught the validation error!")
        print("    -> Error details:")
        print(f"       {e.errors()[0]['loc'][0]}: {e.errors()[0]['msg']}")

    print("\n--- Label Schema Test Complete ---")