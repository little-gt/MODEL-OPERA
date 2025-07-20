# -*- coding: utf-8 -*-
"""
OPERA/opera_data/generator.py

该模块提供了最终的数据集生成器。
它整合了现象、组合、噪声和标签模块，以编程方式创建
大量带标签的图像，用于模型训练。
"""
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Callable, Any, Tuple

# 导入我们所有的构建模块
from opera_data.phenomena.diffraction import generate_fraunhofer_single_slit, generate_airy_disk
from opera_data.phenomena.interference import generate_youngs_double_slit
from opera_data.phenomena.aberrations import apply_zernike_aberration
from opera_data.composition import CompositionEngine
from opera_data.noise_and_distortions import NoiseEngine
from opera_data.label_schema import Label, PhenomenonComponent, NoiseProfile, CompositionType, ValidationError


class DatasetGenerator:
    """
    一个用于生成完整、带标签的光学图像数据集的类。
    """

    def __init__(self, resolution_px: int = 256):
        self.resolution_px = resolution_px
        self.composition_engine = CompositionEngine()
        self.noise_engine = NoiseEngine()

        # 将现象函数的字符串名称映射到实际的函数对象
        self._function_map: Dict[str, Callable] = {
            'generate_fraunhofer_single_slit': generate_fraunhofer_single_slit,
            'generate_airy_disk': generate_airy_disk,
            'generate_youngs_double_slit': generate_youngs_double_slit,
            'apply_zernike_aberration': apply_zernike_aberration
        }

        # 定义哪些是生成器 (从无到有创建图像)
        self._generators = ['generate_fraunhofer_single_slit', 'generate_airy_disk', 'generate_youngs_double_slit']
        # 定义哪些是修改器 (接收并修改图像)
        self._modifiers = ['apply_zernike_aberration']

    def generate_one_sample(self, display_plots: bool = False) -> Tuple[np.ndarray, Label]:
        """
        生成单个随机的 (图像, 标签) 对。

        Args:
            display_plots (bool): 如果为True，则显示生成过程中的中间图像。

        Returns:
            Tuple[np.ndarray, Label]: 返回生成的图像和经过验证的标签对象。
        """
        image_id = f"syn_{random.randint(1000000, 9999999)}"
        composition_type = random.choice(list(CompositionType))

        components_recipe = []
        if composition_type == CompositionType.SEQUENTIAL:
            generator_name = random.choice(self._generators)
            components_recipe.append(
                {'phenomenon_name': generator_name, 'params': self._get_random_params(generator_name)})
            if random.random() < 0.5:
                modifier_name = random.choice(self._modifiers)
                components_recipe.append(
                    {'phenomenon_name': modifier_name, 'params': self._get_random_params(modifier_name)})
        else:  # SUPERPOSITION
            for _ in range(2):
                generator_name = random.choice(self._generators)
                components_recipe.append(
                    {'phenomenon_name': generator_name, 'params': self._get_random_params(generator_name)})

        noise_recipe = self._get_random_noise_profile()

        label = Label(
            image_id=image_id,
            resolution_px=self.resolution_px,
            composition_type=composition_type,
            components=[PhenomenonComponent(**comp) for comp in components_recipe],
            noise_profile=NoiseProfile(**noise_recipe)
        )

        composition_components = []
        for comp in label.components:
            comp_dict = comp.model_dump()
            comp_dict['phenomenon'] = self._function_map[comp.phenomenon_name]
            if comp.phenomenon_name in self._modifiers:
                comp_dict['params']['display_plot'] = display_plots
            composition_components.append(comp_dict)

        clean_image = self.composition_engine.compose(composition_components, label.composition_type.value)
        final_image = self.noise_engine.apply(clean_image, display_plot=display_plots,
                                              **label.noise_profile.model_dump())

        return final_image, label

    def _get_random_params(self, phenomenon_name: str) -> Dict[str, Any]:
        """为给定的现象生成随机参数"""
        # --- 核心修正处 ---
        params: Dict[str, Any] = {}  # 从一个空字典开始

        # 只为需要它的“生成器”函数添加 resolution_px
        if phenomenon_name in self._generators:
            params['resolution_px'] = self.resolution_px
        # --- 修正结束 ---

        if "slit" in phenomenon_name:
            params['wavelength_nm'] = random.uniform(400, 700)
            params['slit_width_mm'] = random.uniform(0.02, 0.1)
            if "double" in phenomenon_name:
                params['slit_separation_mm'] = random.uniform(params['slit_width_mm'] * 2.5, 0.5)
        elif "airy" in phenomenon_name:
            params['wavelength_nm'] = random.uniform(400, 700)
            params['aperture_diameter_mm'] = random.uniform(0.1, 1.5)
        elif "aberration" in phenomenon_name:
            coeffs = {}
            for _ in range(random.randint(1, 2)):
                z_index = random.choice([4, 5, 8, 11])
                coeffs[z_index] = round(random.uniform(0.2, 1.0), 2)
            params['zernike_coeffs'] = coeffs
        return params

    def _get_random_noise_profile(self) -> Dict[str, Any]:
        """生成一个随机的噪声配置"""
        profile = {}
        if random.random() < 0.3: profile['distortion_k1'] = random.uniform(-0.15, 0.15)
        if random.random() < 0.7: profile['gauss_std'] = random.uniform(0.01, 0.1)
        if random.random() < 0.7: profile['poisson_strength'] = random.uniform(20, 100)
        if random.random() < 0.1: profile['salt_pepper_amount'] = random.uniform(0.001, 0.01)
        if random.random() < 0.8: profile['bit_depth'] = random.choice([8, 10, 12])
        if random.random() < 0.4: profile['saturation_level'] = random.uniform(0.85, 1.0)
        return profile

    def generate_and_save_dataset(self, num_samples: int, save_path: str):
        """生成并保存一个完整的数据集，支持断点续传。"""
        img_path = os.path.join(save_path, 'images')
        lbl_path = os.path.join(save_path, 'labels')
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(lbl_path, exist_ok=True)

        start_index = len(os.listdir(img_path))
        print(f"Dataset found with {start_index} samples. Resuming generation...")

        for i in range(start_index, num_samples):
            try:
                image, label = self.generate_one_sample()
                image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                img_filename = os.path.join(img_path, f"{label.image_id}.png")
                plt.imsave(img_filename, image_uint8, cmap='gray')

                lbl_filename = os.path.join(lbl_path, f"{label.image_id}.json")
                with open(lbl_filename, 'w') as f:
                    f.write(label.model_dump_json(indent=2))

                print(f"[{i + 1}/{num_samples}] Successfully generated and saved {label.image_id}")

            except (ValidationError, TypeError, ValueError) as e:
                print(f"Error generating sample {i + 1}: {e}. Skipping.")


if __name__ == '__main__':
    print("--- Testing Dataset Generator ---")

    generator = DatasetGenerator(resolution_px=256)

    print("\n[1] Generating a single random sample for visual inspection...")
    try:
        image, label = generator.generate_one_sample(display_plots=True)
        print("\n--- Generated Label ---")
        print(label.model_dump_json(indent=2))
        print("\n--- Final Image ---")
        plt.imshow(image, cmap='gray')
        plt.title("Final Output Image")
        plt.show()
    except Exception as e:
        print(f"An error occurred during single sample generation: {e}")

    # print("\n[2] Generating a small test dataset (5 samples)...")
    # generator.generate_and_save_dataset(num_samples=5, save_path='./data/synthetic_dataset_v1_test')

    print("\n--- Dataset Generator Test Complete ---")