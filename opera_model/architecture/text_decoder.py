# -*- coding: utf-8 -*-
"""
OPERA/opera_model/architecture/text_decoder.py

该模块实现了标准的Transformer解码器。
它接收来自ViT编码器的图像特征（memory）和当前已生成的文本序列，
并预测序列中的下一个token。
"""

import torch
import torch.nn as nn
from typing import Optional


class TextDecoder(nn.Module):
    """
    一个标准的自回归Transformer解码器。

    它执行以下步骤:
    1. 将输入的token ID嵌入到向量空间 (Token Embedding)。
    2. 为嵌入的token向量添加位置编码 (Positional Encoding)。
    3. 通过一系列Transformer解码器层处理该序列。每个层都包含：
       a. 一个带因果掩码的自注意力机制。
       b. 一个交叉注意力机制，用于关注编码器的输出 (memory)。
    4. 使用一个线性层将最终输出投影到词汇表大小，以获得每个token的logits。
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 768,
            depth: int = 6,
            num_heads: int = 8,
            max_seq_len: int = 128,
            mlp_ratio: float = 4.0,
            dropout_p: float = 0.1
    ):
        """
        初始化TextDecoder。

        Args:
            vocab_size (int): 词汇表的大小 (即可能的输出token的总数)。
            embed_dim (int): 嵌入维度，必须与ViT编码器的embed_dim相同。
            depth (int): Transformer解码器层的数量。
            num_heads (int): 注意力机制的头数。
            max_seq_len (int): 模型可以处理的最大序列长度。
            mlp_ratio (float): MLP层的隐藏维度扩展比率。
            dropout_p (float): Dropout的概率。
        """
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Token Embedding 和 Positional Encoding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_p)

        # 2. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout_p,
            activation='gelu',
            batch_first=True  # !! 保持与编码器一致
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

        # 3. 输出头 (Output Head)
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def forward(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            tgt (torch.Tensor): 目标序列 (即已生成的token)，形状 (B, T)。T是目标序列长度。
            memory (torch.Tensor): 来自ViT编码器的上下文，形状 (B, S, D)。S是源序列长度, D是嵌入维度。
            tgt_mask (torch.Tensor, optional): 目标序列的因果掩码，形状 (T, T)。
            tgt_key_padding_mask (torch.Tensor, optional): 目标序列的填充掩码，形状 (B, T)。

        Returns:
            torch.Tensor: 输出的logits，形状 (B, T, vocab_size)。
        """
        B, T = tgt.shape

        # 1. 嵌入和位置编码
        tgt_embed = self.token_embed(tgt)  # (B, T) -> (B, T, D)
        tgt_embed = tgt_embed + self.pos_embed[:, :T, :]
        tgt_embed = self.pos_drop(tgt_embed)

        # 2. 如果没有提供掩码，则自动生成一个标准的因果掩码
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(T, tgt.device)

        # 3. 通过解码器层
        output = self.decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # 4. 投影到词汇表
        logits = self.output_head(output)  # (B, T, D) -> (B, T, vocab_size)

        return logits

    @staticmethod
    def _generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """生成一个上三角因果掩码。"""
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)


# 当该文件被直接执行时，运行以下测试代码
if __name__ == '__main__':
    print("--- Testing Text Decoder ---")

    # 1. 定义模型参数
    vocab_size = 100  # 假设我们的词汇表有100个token
    embed_dim = 192  # 必须与ViT编码器的输出维度匹配
    max_len = 50  # 标签的最大长度

    # 2. 创建模型实例
    decoder = TextDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        depth=4,
        num_heads=6,
        max_seq_len=max_len
    )
    print("\n[1] Model Architecture:")
    print(decoder)

    # 3. 创建虚拟输入
    batch_size = 4
    encoder_seq_len = 65  # 来自ViT的输出 (64 patches + 1 CLS)
    decoder_seq_len = 15  # 假设我们已经生成了15个token

    # 模拟ViT编码器的输出 (memory)
    memory = torch.randn(batch_size, encoder_seq_len, embed_dim)
    # 模拟已经生成的目标序列 (tgt)
    tgt = torch.randint(0, vocab_size, (batch_size, decoder_seq_len))

    print(f"\n[2] Input tensor shapes:")
    print(f"    - Memory (from Encoder): {memory.shape}")
    print(f"    - Target (generated tokens): {tgt.shape}")

    # 4. 执行前向传播
    try:
        output_logits = decoder(tgt, memory)
        print(f"\n[3] Forward pass successful!")
        print(f"    -> Output logits shape: {output_logits.shape}")

        # 5. 验证输出形状
        # 期望形状: (Batch, TargetSeqLen, VocabSize)
        expected_shape = (batch_size, decoder_seq_len, vocab_size)
        assert output_logits.shape == expected_shape, "Output shape is incorrect!"
        print(f"    -> Verified: Output shape {output_logits.shape} matches expected shape {expected_shape}.")

    except Exception as e:
        print(f"\nAn error occurred during forward pass: {e}")

    print("\n--- Text Decoder Test Complete ---")