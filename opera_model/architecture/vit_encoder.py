# -*- coding: utf-8 -*-
"""
OPERA/opera_model/architecture/vit_encoder.py

该模块实现了 Vision Transformer (ViT) 的图像编码器部分。
它的作用是将输入的图像转换为一系列特征向量（tokens），以供后续的
解码器处理。
"""

import torch
import torch.nn as nn
from typing import Tuple


class ViTEncoder(nn.Module):
    """
    一个标准的Vision Transformer编码器。

    它执行以下步骤:
    1. 将图像分割成固定大小的非重叠块 (Patching)。
    2. 使用一个卷积层将每个块线性嵌入到一个向量空间 (Patch Embedding)。
    3. 在序列前添加一个可学习的 [CLS] token。
    4. 为序列中的每个向量添加可学习的位置编码 (Positional Encoding)。
    5. 通过一系列Transformer编码器层处理该序列。
    """

    def __init__(
            self,
            img_size: int = 256,
            patch_size: int = 16,
            in_channels: int = 1,
            embed_dim: int = 768,
            depth: int = 6,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout_p: float = 0.1
    ):
        """
        初始化ViT编码器。

        Args:
            img_size (int): 输入图像的方形尺寸。
            patch_size (int): 每个图像块的方形尺寸。
            in_channels (int): 输入图像的通道数 (我们的情况是1，灰度图)。
            embed_dim (int): 模型中向量的嵌入维度。
            depth (int): Transformer编码器层的数量。
            num_heads (int): 多头自注意力机制中的头数。
            mlp_ratio (float): Transformer编码器中MLP层的隐藏维度扩展比率。
            dropout_p (float): Dropout的概率。
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 1. Patch Embedding
        # 使用一个Conv2d层可以高效地实现分块和线性嵌入
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 计算块的数量
        num_patches = (img_size // patch_size) ** 2

        # 2. CLS Token 和 Positional Encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_p)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout_p,
            activation='gelu',  # GELU是Transformer中常用的激活函数
            batch_first=True  # !! 非常重要，让输入格式为 (B, Seq, Dim)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. 最终的层归一化
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入图像张量，形状为 (B, C, H, W)。

        Returns:
            torch.Tensor: 输出的特征向量序列，形状为 (B, num_patches+1, embed_dim)。
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"输入图像尺寸({H}x{W})与模型期望尺寸({self.img_size}x{self.img_size})不匹配。"

        # Patch Embedding: (B, C, H, W) -> (B, D, H/P, W/P)
        x = self.patch_embed(x)

        # Flatten and transpose: (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        # N = num_patches
        x = x.flatten(2).transpose(1, 2)

        # Prepend CLS token: (B, N, D) -> (B, N+1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through Transformer encoder
        x = self.encoder(x)

        # Final normalization
        x = self.norm(x)

        return x


# 当该文件被直接执行时，运行以下测试代码
if __name__ == '__main__':
    print("--- Testing ViT Encoder ---")

    # 1. 创建一个模型实例 (使用较小的参数以便快速测试)
    model = ViTEncoder(
        img_size=256,
        patch_size=32,  # 256 / 32 = 8x8 patches
        in_channels=1,
        embed_dim=192,
        depth=4,
        num_heads=6
    )
    print("\n[1] Model Architecture:")
    print(model)

    # 2. 创建一个虚拟的输入张量
    # Batch size = 4, Channels = 1, Height = 256, Width = 256
    dummy_input = torch.randn(4, 1, 256, 256)
    print(f"\n[2] Input tensor shape: {dummy_input.shape}")

    # 3. 执行前向传播
    try:
        output = model(dummy_input)
        print(f"\n[3] Forward pass successful!")
        print(f"    -> Output tensor shape: {output.shape}")

        # 4. 验证输出形状
        # 期望形状: (Batch, NumPatches + CLS, EmbedDim)
        # NumPatches = (256/32)^2 = 64
        # NumPatches + CLS = 65
        # EmbedDim = 192
        expected_shape = (4, 65, 192)
        assert output.shape == expected_shape, "Output shape is incorrect!"
        print(f"    -> Verified: Output shape {output.shape} matches expected shape {expected_shape}.")

    except Exception as e:
        print(f"\nAn error occurred during forward pass: {e}")

    print("\n--- ViT Encoder Test Complete ---")