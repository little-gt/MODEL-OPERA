# -*- coding: utf-8 -*-
"""
OPERA/opera_data/phenomena/diffraction.py

该模块负责生成各种衍射现象的图像。
每个函数都基于标量衍射理论，并模拟一种特定的衍射类型。
函数的核心功能是返回一个代表光强分布的2D Numpy数组。

包含的函数:
- generate_fraunhofer_single_slit: 模拟单缝的夫琅禾费衍射。
- generate_airy_disk: 模拟圆孔的夫琅禾费衍射 (艾里斑)。
"""

import numpy as np
from scipy.special import j1
import matplotlib.pyplot as plt


def generate_fraunhofer_single_slit(
        slit_width_mm: float = 0.1,
        wavelength_nm: float = 632.8,
        distance_m: float = 1.0,
        screen_height_mm: float = 50.0,  # 参数名修改，更符合物理意义
        resolution_px: int = 512,
        display_plot: bool = False
) -> np.ndarray:
    """
    生成(竖直)单缝在水平方向展开的夫琅禾费衍射2D图像。

    Args:
        slit_width_mm (float): 缝的宽度，单位为毫米 (mm)。
        wavelength_nm (float): 入射光的波长，单位为纳米 (nm)。
        distance_m (float): 缝到屏幕的距离，单位为米 (m)。
        screen_height_mm (float): 模拟屏幕的高度，单位为毫米 (mm)。衍射图样在此方向展开。
        resolution_px (int): 生成图像的方形分辨率 (像素)。
        display_plot (bool): 如果为True，则显示生成的图像和强度剖面图。

    Returns:
        np.ndarray: 一个 (resolution_px, resolution_px) 的二维数组，
                    值在 [0, 1] 范围内，代表归一化的光强分布。
    """
    # 1. 单位转换至国际单位制 (SI)
    a = slit_width_mm * 1e-3
    lambda_ = wavelength_nm * 1e-9
    L = distance_m
    h = screen_height_mm * 1e-3  # 使用 h 代表屏幕高度

    # 2. 计算屏幕上 Y 轴的坐标，因为衍射沿水平方向展开
    y = np.linspace(-h / 2, h / 2, resolution_px)

    # 3. 计算衍射角
    sin_theta = y / L

    # 4. 计算相位差 beta
    beta = (np.pi * a * sin_theta) / (lambda_ + 1e-15)

    # 5. 计算1D强度分布
    intensity_1d = (np.sinc(beta / np.pi)) ** 2

    # 6. 【修正处】创建2D图像
    #    为了生成水平条纹，我们需要将1D强度分布作为“列”向量，
    #    然后在水平方向上平铺 (tile) 它。
    #    (a) 将 intensity_1d 从 (res,) 变形为 (res, 1) 的列向量。
    intensity_column = intensity_1d.reshape((resolution_px, 1))
    #    (b) 在水平方向上平铺 resolution_px 次，垂直方向不变。
    image_2d = np.tile(intensity_column, (1, resolution_px))

    # 7. (可选) 绘图展示
    if display_plot:
        plt.figure(figsize=(12, 6))
        plt.suptitle(f"Single-Slit Diffraction (a={slit_width_mm}mm, λ={wavelength_nm}nm)", fontsize=16)

        plt.subplot(1, 2, 1)
        # 交换x, y轴来绘制垂直的强度剖面图
        plt.plot(intensity_1d, y * 1000)
        plt.title("1D Intensity Profile")
        plt.xlabel("Normalized Intensity")
        plt.ylabel("Position on screen (mm)")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.imshow(image_2d, cmap='gray', extent=[-h / 2 * 1000, h / 2 * 1000, -h / 2 * 1000, h / 2 * 1000])
        plt.title("2D Diffraction Pattern (Horizontal)")
        plt.xlabel("Screen X (mm)")
        plt.ylabel("Screen Y (mm)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return image_2d


def generate_airy_disk(
        aperture_diameter_mm: float = 0.5,
        wavelength_nm: float = 550.0,
        distance_m: float = 2.0,
        screen_width_mm: float = 20.0,
        resolution_px: int = 512,
        display_plot: bool = False
) -> np.ndarray:
    """
    生成圆孔夫琅禾费衍射图样 (艾里斑)。
    (此函数无需修改)
    """
    # 1. 单位转换
    D = aperture_diameter_mm * 1e-3
    lambda_ = wavelength_nm * 1e-9
    L = distance_m
    w = screen_width_mm * 1e-3

    # 2. 创建2D坐标网格
    ax = np.linspace(-w / 2, w / 2, resolution_px)
    xx, yy = np.meshgrid(ax, ax)

    # 3. 计算径向距离
    r = np.sqrt(xx ** 2 + yy ** 2)

    # 4. 计算贝塞尔函数参数
    u = (np.pi * D * r) / (lambda_ * (L + 1e-15))

    # 5. 计算强度
    intensity = np.ones_like(u)
    nonzero_u_indices = u != 0
    u_nonzero = u[nonzero_u_indices]
    intensity[nonzero_u_indices] = (2 * j1(u_nonzero) / u_nonzero) ** 2

    # 6. 绘图展示
    if display_plot:
        plt.figure(figsize=(12, 5))
        plt.suptitle(f"Airy Disk (D={aperture_diameter_mm}mm, λ={wavelength_nm}nm)", fontsize=16)

        plt.subplot(1, 2, 1)
        plt.imshow(np.log1p(intensity), cmap='inferno',
                   extent=[-w / 2 * 1000, w / 2 * 1000, -w / 2 * 1000, w / 2 * 1000])
        plt.title("2D Airy Disk (Log Scale)")
        plt.xlabel("Screen X (mm)")
        plt.ylabel("Screen Y (mm)")

        plt.subplot(1, 2, 2)
        center_line = intensity[resolution_px // 2, :]
        plt.plot(ax * 1000, center_line)
        plt.title("1D Center Profile")
        plt.xlabel("Position on screen (mm)")
        plt.ylabel("Normalized Intensity")
        plt.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return intensity


# 当该文件被直接执行时，运行以下测试代码
if __name__ == '__main__':
    print("--- Testing Diffraction Module (Corrected) ---")

    print("\n[1] Generating single-slit diffraction pattern (horizontal)...")
    single_slit_image = generate_fraunhofer_single_slit(display_plot=True)
    print(f"    -> Returned image shape: {single_slit_image.shape}")

    print("\n[2] Generating Airy disk pattern...")
    airy_disk_image = generate_airy_disk(display_plot=True)
    print(f"    -> Returned image shape: {airy_disk_image.shape}")

    print("\n--- Diffraction Module Test Complete ---")