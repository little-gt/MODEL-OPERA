# -*- coding: utf-8 -*-
"""
OPERA/opera_data/noise_and_distortions.py

该模块提供了一个噪声与畸变引擎，用于降低合成图像的理想度，
使其更接近真实世界中通过光学系统和传感器捕获的图像。
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional


class NoiseEngine:
    """
    一个将真实世界噪声和畸变应用于理想图像的引擎。
    """

    def apply(
            self,
            image: np.ndarray,
            distortion_k1: Optional[float] = None,
            gauss_std: Optional[float] = None,
            poisson_strength: Optional[float] = None,
            salt_pepper_amount: Optional[float] = None,
            bit_depth: Optional[int] = None,
            saturation_level: float = 1.0,
            display_plot: bool = False
    ) -> np.ndarray:
        """
        按照物理上合理的顺序将一系列效果应用于图像。

        Args:
            image (np.ndarray): 输入的理想图像 (float, 范围 [0, 1])。
            distortion_k1 (float, optional): 桶形/枕形畸变系数。>0为桶形, <0为枕形。
            gauss_std (float, optional): 高斯读出噪声的标准差。
            poisson_strength (float, optional): 泊松散粒噪声的强度。
            salt_pepper_amount (float, optional): 椒盐噪声的比例。
            bit_depth (int, optional): 模拟ADC量化的位深度 (例如 8 for 8-bit)。
            saturation_level (float): 传感器饱和点 (0到1之间)。
            display_plot (bool): 如果为True，则显示处理前后的图像对比。

        Returns:
            np.ndarray: 应用效果后的图像。
        """
        processed_image = image.copy().astype(np.float64)

        # 1. 应用几何畸变 (模拟镜头)
        if distortion_k1 is not None:
            processed_image = self._apply_barrel_distortion(processed_image, distortion_k1)

        # 2. 应用物理噪声 (模拟光子和电子)
        if poisson_strength is not None:
            processed_image = self._apply_poisson_noise(processed_image, poisson_strength)

        if gauss_std is not None:
            processed_image = self._apply_gaussian_noise(processed_image, gauss_std)

        if salt_pepper_amount is not None:
            processed_image = self._apply_salt_pepper_noise(processed_image, salt_pepper_amount)

        # 3. 应用探测器电子效应 (模拟ADC和放大器)
        if saturation_level < 1.0:
            processed_image = np.clip(processed_image, 0, saturation_level)

        if bit_depth is not None:
            processed_image = self._apply_quantization(processed_image, bit_depth)

        if display_plot:
            self._display_comparison(image, processed_image)

        return processed_image

    def _apply_barrel_distortion(self, image: np.ndarray, k1: float) -> np.ndarray:
        h, w = image.shape[:2]
        center_x, center_y = w / 2, h / 2

        # 创建一个从目标像素到源像素的映射
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                # 归一化并平移坐标
                dx, dy = (x - center_x) / center_x, (y - center_y) / center_y
                r_sq = dx * dx + dy * dy
                # 畸变公式
                factor = 1 + k1 * r_sq
                # 计算源坐标
                src_x = factor * dx
                src_y = factor * dy
                # 反归一化
                map_x[y, x] = (src_x * center_x) + center_x
                map_y[y, x] = (src_y * center_y) + center_y

        # 使用OpenCV的remap函数应用畸变
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    def _apply_gaussian_noise(self, image: np.ndarray, std: float) -> np.ndarray:
        noise = np.random.normal(0, std, image.shape)
        return np.clip(image + noise, 0, 1)

    def _apply_poisson_noise(self, image: np.ndarray, strength: float) -> np.ndarray:
        # 泊松噪声与信号强度有关。我们用一个缩放因子来模拟光子计数。
        scaled_image = image * strength
        noisy_image = np.random.poisson(scaled_image) / strength
        return np.clip(noisy_image, 0, 1)

    def _apply_salt_pepper_noise(self, image: np.ndarray, amount: float) -> np.ndarray:
        noisy_image = image.copy()
        num_pixels = image.size
        # Salt
        num_salt = int(num_pixels * amount * 0.5)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        noisy_image[tuple(coords)] = 1
        # Pepper
        num_pepper = int(num_pixels * amount * 0.5)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        noisy_image[tuple(coords)] = 0
        return noisy_image

    def _apply_quantization(self, image: np.ndarray, bit_depth: int) -> np.ndarray:
        levels = 2 ** bit_depth
        quantized_image = np.round(image * (levels - 1)) / (levels - 1)
        return quantized_image

    def _display_comparison(self, original, processed):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray')
        plt.title("Original Ideal Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(processed, cmap='gray')
        plt.title("Image with Noise & Distortions")
        plt.axis('off')

        plt.tight_layout()
        plt.show()


# 当该文件被直接执行时，运行以下测试代码
if __name__ == '__main__':
    # 从我们的现象库中导入一个生成器
    from phenomena.diffraction import generate_airy_disk

    print("--- Testing Noise Engine ---")

    # 1. 生成一个干净的测试图像
    print("\n[1] Generating a clean test image (Airy Disk)...")
    clean_image = generate_airy_disk(
        aperture_diameter_mm=1.0,
        resolution_px=256,
        display_plot=False
    )

    # 2. 实例化并应用引擎
    print("\n[2] Applying a full suite of noise and distortions...")
    noise_engine = NoiseEngine()

    noisy_image = noise_engine.apply(
        clean_image,
        distortion_k1=0.2,  # 中等程度的桶形畸变
        gauss_std=0.05,  # 明显的读出噪声
        poisson_strength=50,  # 中等强度的散粒噪声
        salt_pepper_amount=0.005,  # 0.5%的像素是坏点
        bit_depth=8,  # 8-bit相机
        saturation_level=0.9,  # 90%的亮度即饱和
        display_plot=True
    )

    print(f"\n    -> Noise engine applied successfully. Image shape: {noisy_image.shape}")
    print("\n--- Noise Engine Test Complete ---")