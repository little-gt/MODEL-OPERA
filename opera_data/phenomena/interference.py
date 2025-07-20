# -*- coding: utf-8 -*-
"""
OPERA/opera_data/phenomena/interference.py

该模块负责生成各种干涉现象的图像。
每个函数都模拟一种特定的干涉类型，并返回一个代表光强分布的2D Numpy数组。

包含的函数:
- generate_youngs_double_slit: 模拟杨氏双缝干涉。
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_youngs_double_slit(
        slit_width_mm: float = 0.05,
        slit_separation_mm: float = 0.25,
        wavelength_nm: float = 550.0,
        distance_m: float = 1.5,
        screen_height_mm: float = 80.0,
        resolution_px: int = 1024,
        display_plot: bool = False
) -> np.ndarray:
    """
    生成杨氏双缝干涉的2D图像。
    该模型是衍射效应和干涉效应的乘积。

    Args:
        slit_width_mm (float): 单个缝的宽度 (a)，单位为毫米 (mm)。
        slit_separation_mm (float): 两条缝中心的间距 (d)，单位为毫米 (mm)。
        wavelength_nm (float): 入射光的波长，单位为纳米 (nm)。
        distance_m (float): 缝到屏幕的距离，单位为米 (m)。
        screen_height_mm (float): 模拟屏幕的高度，单位为毫米 (mm)。
        resolution_px (int): 生成图像的方形分辨率 (像素)。
        display_plot (bool): 如果为True，则显示生成的图像和强度剖面图。

    Returns:
        np.ndarray: 一个 (resolution_px, resolution_px) 的二维数组，
                    值在 [0, 1] 范围内，代表归一化的光强分布。
    """
    # 1. 单位转换至国际单位制 (SI)
    a = slit_width_mm * 1e-3
    d = slit_separation_mm * 1e-3
    lambda_ = wavelength_nm * 1e-9
    L = distance_m
    h = screen_height_mm * 1e-3

    # 2. 计算屏幕上的 Y 轴坐标
    y = np.linspace(-h / 2, h / 2, resolution_px)

    # 3. 计算衍射角 (远场近似)
    sin_theta = y / L

    # 4. 计算衍射项 (由单个缝的宽度 a 决定)
    beta = (np.pi * a * sin_theta) / (lambda_ + 1e-15)
    diffraction_term = (np.sinc(beta / np.pi)) ** 2

    # 5. 计算干涉项 (由双缝的间距 d 决定)
    alpha = (np.pi * d * sin_theta) / (lambda_ + 1e-15)
    interference_term = (np.cos(alpha)) ** 2

    # 6. 计算总强度 (衍射项 * 干涉项)
    intensity_1d = diffraction_term * interference_term

    # 7. 创建2D图像 (与单缝衍射相同的逻辑，生成水平条纹)
    intensity_column = intensity_1d.reshape((resolution_px, 1))
    image_2d = np.tile(intensity_column, (1, resolution_px))

    # 8. (可选) 绘图展示
    if display_plot:
        plt.figure(figsize=(10, 8))
        plt.suptitle(f"Young's Double-Slit (a={slit_width_mm}mm, d={slit_separation_mm}mm, λ={wavelength_nm}nm)",
                     fontsize=16)

        # 绘制分解项
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(y * 1000, diffraction_term, 'b--', alpha=0.7, label='Diffraction Envelope (from single slit)')
        plt.plot(y * 1000, interference_term, 'g:', alpha=0.7, label='Interference Fringes (from two slits)')
        plt.title("Component Profiles")
        plt.ylabel("Normalized Intensity")
        plt.legend()
        plt.grid(True)

        # 绘制总强度
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)  # 共享X轴
        plt.plot(y * 1000, intensity_1d, 'r', label='Total Intensity (Product)')
        plt.title("Resulting Intensity Profile")
        plt.xlabel("Position on screen (mm)")
        plt.ylabel("Normalized Intensity")
        plt.legend()
        plt.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # 单独显示最终的2D图像
        plt.figure(figsize=(8, 8))
        plt.imshow(image_2d, cmap='gray', extent=[-h / 2 * 1000, h / 2 * 1000, -h / 2 * 1000, h / 2 * 1000])
        plt.title("2D Interference Pattern")
        plt.xlabel("Screen X (mm)")
        plt.ylabel("Screen Y (mm)")
        plt.show()

    return image_2d


# 当该文件被直接执行时，运行以下测试代码
if __name__ == '__main__':
    print("--- Testing Interference Module ---")

    print("\n[1] Generating Young's double-slit interference pattern...")
    interference_image = generate_youngs_double_slit(display_plot=True)
    print(f"    -> Returned image shape: {interference_image.shape}")
    print(f"    -> Data type: {interference_image.dtype}")
    print(f"    -> Intensity range: [{interference_image.min():.4f}, {interference_image.max():.4f}]")

    print("\n--- Interference Module Test Complete ---")