# -*- coding: utf-8 -*-
"""
OPERA/opera_data/phenomena/aberrations.py

该模块负责将光学像差应用于输入图像。
它使用泽尼克多项式来建立一个物理上精确的像差模型，
计算出点扩散函数(PSF)，并通过卷积将其应用于图像。

包含的函数:
- apply_zernike_aberration: 将像差效果应用于一个图像上。
- _generate_wavefront_from_zernike_coeffs: (内部函数) 生成波前。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def _generate_wavefront_from_zernike_coeffs(
        zernike_coeffs: dict,
        resolution_px: int
) -> np.ndarray:
    """
    根据一个包含泽尼克系数的字典(Noll索引)生成一个波前误差图。
    系数的单位是“波长(waves)”。例如，系数0.5代表半个波长的光程差。

    Args:
        zernike_coeffs (dict): 一个字典，键是Noll索引(int)，值是系数(float)。
        resolution_px (int): 生成波前图的分辨率。

    Returns:
        np.ndarray: 一个(resolution_px, resolution_px)的二维数组，代表波前误差。
    """
    if not zernike_coeffs:
        return np.zeros((resolution_px, resolution_px))

    # 创建一个归一化的坐标网格 (-1 to 1)
    x = np.linspace(-1, 1, resolution_px)
    y = np.linspace(-1, 1, resolution_px)
    xx, yy = np.meshgrid(x, y)

    # 转换为极坐标
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)

    wavefront = np.zeros_like(rho)
    pupil_mask = (rho <= 1.0)  # 像差只在单位圆(光瞳)内定义

    # 常用泽尼克多项式的Noll索引及其公式
    # R_n^m(rho)是径向多项式
    zernike_map = {
        # n=1: Tip/Tilt (图像位移)
        2: lambda r, t: 2 * r * np.cos(t),  # Z(1, 1)  - Tilt X
        3: lambda r, t: 2 * r * np.sin(t),  # Z(1, -1) - Tilt Y
        # n=2: Defocus/Astigmatism (二阶像差)
        4: lambda r, t: np.sqrt(3) * (2 * r ** 2 - 1),  # Z(2, 0) - Defocus (离焦)
        5: lambda r, t: np.sqrt(6) * r ** 2 * np.cos(2 * t),  # Z(2, 2) - Astigmatism (像散) at 0/90 deg
        6: lambda r, t: np.sqrt(6) * r ** 2 * np.sin(2 * t),  # Z(2, -2) - Astigmatism at 45 deg
        # n=3: Coma/Trefoil (三阶像差)
        7: lambda r, t: np.sqrt(8) * (3 * r ** 3 - 2 * r) * np.sin(t),  # Z(3, -1) - Coma Y (彗差)
        8: lambda r, t: np.sqrt(8) * (3 * r ** 3 - 2 * r) * np.cos(t),  # Z(3, 1)  - Coma X
        # n=4: Spherical Aberration (四阶像差)
        11: lambda r, t: np.sqrt(5) * (6 * r ** 4 - 6 * r ** 2 + 1)  # Z(4, 0) - Spherical (球差)
    }

    for noll_index, coeff in zernike_coeffs.items():
        if noll_index in zernike_map:
            wavefront += coeff * zernike_map[noll_index](rho, theta)

    return wavefront * pupil_mask


def apply_zernike_aberration(
        input_image: np.ndarray,
        zernike_coeffs: dict,
        display_plot: bool = False
) -> np.ndarray:
    """
    通过卷积一个由泽尼克系数定义的PSF，将光学像差应用于图像。

    Args:
        input_image (np.ndarray): 输入的理想图像 (灰度图)。
        zernike_coeffs (dict): 描述像差的泽尼克系数字典。
        display_plot (bool): 如果为True，则显示波前、PSF和最终图像。

    Returns:
        np.ndarray: 应用像差后的图像。
    """
    if not zernike_coeffs or not any(zernike_coeffs.values()):
        return input_image

    resolution_px = input_image.shape[0]

    # 1. 生成波前误差 W (单位: waves)
    wavefront = _generate_wavefront_from_zernike_coeffs(zernike_coeffs, resolution_px)

    # 2. 计算光瞳函数 P(rho, theta) = A * exp(i * phi)
    pupil_amplitude = (wavefront != 0).astype(float)  # 振幅在光瞳内为1，外为0
    pupil_phase = 2 * np.pi * wavefront  # 将波前误差转换为相位 (radians)
    pupil_function = pupil_amplitude * np.exp(1j * pupil_phase)

    # 3. 计算点扩散函数 PSF
    # PSF 是光瞳函数的傅里叶变换的模的平方
    psf = np.abs(np.fft.fftshift(np.fft.fft2(pupil_function))) ** 2
    psf /= np.sum(psf)  # 归一化PSF，使其总能量为1

    # 4. 将PSF与输入图像进行卷积
    aberrated_image = convolve2d(input_image, psf, mode='same', boundary='wrap')

    if display_plot:
        title_str = ", ".join([f"Z{k}={v}λ" for k, v in zernike_coeffs.items()])
        plt.figure(figsize=(18, 6))
        plt.suptitle(f"Aberration: {title_str}", fontsize=16)

        plt.subplot(1, 4, 1)
        plt.imshow(input_image, cmap='gray')
        plt.title("Original Image")

        plt.subplot(1, 4, 2)
        plt.imshow(wavefront, cmap='viridis')
        plt.title("Wavefront Error (waves)")
        plt.colorbar()

        plt.subplot(1, 4, 3)
        plt.imshow(np.log1p(psf), cmap='inferno')
        plt.title("Point Spread Function (log scale)")

        plt.subplot(1, 4, 4)
        plt.imshow(aberrated_image, cmap='gray')
        plt.title("Aberrated Image")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return aberrated_image


if __name__ == '__main__':
    print("--- Testing Aberrations Module ---")

    # 创建一个测试图像 (一个理想的点光源网格)
    res = 256
    test_image = np.zeros((res, res))
    test_image[res // 4::res // 8, res // 4::res // 8] = 1.0

    # 1. 测试 Defocus (离焦)
    print("\n[1] Applying Defocus aberration (Z4 = 0.5 waves)...")
    defocus_coeffs = {4: 0.5}  # Noll index 4 for defocus
    apply_zernike_aberration(test_image.copy(), defocus_coeffs, display_plot=True)

    # 2. 测试 Spherical Aberration (球差)
    print("\n[2] Applying Spherical aberration (Z11 = 0.75 waves)...")
    spherical_coeffs = {11: 0.75}  # Noll index 11 for spherical
    apply_zernike_aberration(test_image.copy(), spherical_coeffs, display_plot=True)

    # 3. 测试混合像差 Coma + Astigmatism
    print("\n[3] Applying mixed aberrations (Coma + Astigmatism)...")
    mixed_coeffs = {
        5: 0.6,  # Astigmatism
        8: -0.5,  # Coma
    }
    apply_zernike_aberration(test_image.copy(), mixed_coeffs, display_plot=True)

    print("\n--- Aberrations Module Test Complete ---")