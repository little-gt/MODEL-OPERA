# -*- coding: utf-8 -*-
"""
OPERA/opera_data/composition.py

该模块提供了现象组合器引擎。
它接收一个描述了多个基础光学现象及其参数的“配方”，
并根据指定的组合类型（序贯或叠加）生成最终的复杂图像。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Callable


class CompositionEngine:
    """
    一个用于组合多个光学现象的引擎。
    """

    def compose(
            self,
            components: List[Dict[str, Any]],
            composition_type: str = 'sequential',
            display_plot: bool = False
    ) -> np.ndarray:
        """
        根据给定的组件列表和组合类型生成一个复合图像。

        Args:
            components (List[Dict]):
                一个组件列表。每个组件是一个字典，应包含:
                - 'phenomenon': 一个可调用的函数 (例如, from phenomena.diffraction import generate_airy_disk)。
                - 'params': 一个包含该函数所需参数的字典。
                - 'weight' (可选): 在'superposition'模式下使用的权重。

            composition_type (str):
                组合类型, 'sequential' (序贯) 或 'superposition' (叠加)。

            display_plot (bool):
                如果为True, 则显示最终生成的图像。

        Returns:
            np.ndarray: 最终生成的复合图像。
        """
        if composition_type == 'sequential':
            final_image = self._compose_sequential(components)
        elif composition_type == 'superposition':
            final_image = self._compose_superposition(components)
        else:
            raise ValueError(f"未知的组合类型: '{composition_type}'. "
                             "有效选项为 'sequential' 或 'superposition'。")

        if display_plot:
            plt.figure(figsize=(8, 8))
            title = f"Final Composed Image ({composition_type})"
            plt.imshow(final_image, cmap='gray')
            plt.title(title)
            plt.show()

        return final_image

    def _compose_sequential(self, components: List[Dict]) -> np.ndarray:
        """
        处理序贯组合。第一个组件必须是生成器，后续的为修改器。
        """
        current_image = None
        for i, comp in enumerate(components):
            phenomenon_func: Callable = comp['phenomenon']
            params: Dict = comp.get('params', {})

            if i == 0:
                # 第一个组件必须是“生成器”，它从零开始创建图像。
                current_image = phenomenon_func(**params)
                if not isinstance(current_image, np.ndarray):
                    raise TypeError(f"序贯模式的第一个现象 '{phenomenon_func.__name__}' "
                                    "必须是一个图像生成器，但它没有返回一个Numpy数组。")
            else:
                # 后续组件是“修改器”，它们接收并修改前一个图像。
                if current_image is None:
                    raise ValueError("序贯模式在处理非首个组件时，图像不存在。")
                current_image = phenomenon_func(current_image, **params)

        return current_image

    def _compose_superposition(self, components: List[Dict]) -> np.ndarray:
        """
        处理叠加组合。所有组件都必须是生成器。
        """
        final_image = None

        for i, comp in enumerate(components):
            phenomenon_func: Callable = comp['phenomenon']
            params: Dict = comp.get('params', {})
            weight: float = comp.get('weight', 1.0)

            # 所有组件都必须是生成器
            generated_image = phenomenon_func(**params)

            if final_image is None:
                # 使用第一个图像的形状初始化最终图像
                final_image = np.zeros_like(generated_image, dtype=np.float64)

            if generated_image.shape != final_image.shape:
                raise ValueError(f"在叠加模式下，所有图像的分辨率必须相同。 "
                                 f"图像 {i} 的形状为 {generated_image.shape}，"
                                 f"而期望的形状为 {final_image.shape}。")

            final_image += generated_image * weight

        # 归一化最终图像，使其值在 [0, 1] 区间内
        max_val = np.max(final_image)
        if max_val > 0:
            final_image /= max_val

        return final_image


# 当该文件被直接执行时，运行以下测试代码
if __name__ == '__main__':
    # 导入我们需要的基础现象函数
    from phenomena.diffraction import generate_airy_disk
    from phenomena.interference import generate_youngs_double_slit
    from phenomena.aberrations import apply_zernike_aberration

    print("--- Testing Composition Engine ---")
    engine = CompositionEngine()
    res_px = 256

    # --- 测试 1: 序贯组合 ---
    # 目标: 生成一个艾里斑，然后对其应用球差
    print("\n[1] Testing 'sequential' composition: Airy Disk + Spherical Aberration...")
    sequential_components = [
        {
            'phenomenon': generate_airy_disk,
            'params': {'aperture_diameter_mm': 1.0, 'resolution_px': res_px}
        },
        {
            'phenomenon': apply_zernike_aberration,
            'params': {'zernike_coeffs': {11: 0.8}, 'display_plot': True}  # Z11: Spherical Aberration
        }
    ]
    sequential_image = engine.compose(sequential_components,
                                      composition_type='sequential',
                                      display_plot=True)
    print(f"    -> Sequential composition successful. Image shape: {sequential_image.shape}")

    # --- 测试 2: 叠加组合 ---
    # 目标: 叠加两种不同波长（颜色）光的杨氏双缝干涉图样
    print("\n[2] Testing 'superposition' composition: Double-Slit (Red) + Double-Slit (Blue)...")
    superposition_components = [
        {
            'phenomenon': generate_youngs_double_slit,
            'params': {
                'wavelength_nm': 650,  # 红色
                'slit_width_mm': 0.05,
                'slit_separation_mm': 0.2,
                'resolution_px': res_px,
            },
            'weight': 1.0
        },
        {
            'phenomenon': generate_youngs_double_slit,
            'params': {
                'wavelength_nm': 450,  # 蓝色
                'slit_width_mm': 0.05,
                'slit_separation_mm': 0.2,
                'resolution_px': res_px,
            },
            'weight': 1.0
        }
    ]
    superposition_image = engine.compose(superposition_components,
                                         composition_type='superposition',
                                         display_plot=True)
    print(f"    -> Superposition successful. Image shape: {superposition_image.shape}")

    print("\n--- Composition Engine Test Complete ---")