# -*- coding: utf-8 -*-
"""
OPERA/opera_model/architecture/complete_model.py

该模块将ViT编码器和文本解码器组合成一个单一的、端到端的模型，
用于将图像转换为文本序列。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# 导入我们已经构建的子模块
from .vit_encoder import ViTEncoder
from .text_decoder import TextDecoder


class ImageToTextModel(nn.Module):
    """
    一个端到端的图像到文本模型，整合了ViTEncoder和TextDecoder。
    """

    def __init__(self, encoder_cfg: Dict[str, Any], decoder_cfg: Dict[str, Any]):
        """
        初始化完整的模型。

        Args:
            encoder_cfg (Dict[str, Any]): 用于初始化ViTEncoder的配置字典。
            decoder_cfg (Dict[str, Any]): 用于初始化TextDecoder的配置字典。
        """
        super().__init__()

        assert encoder_cfg['embed_dim'] == decoder_cfg['embed_dim'], \
            "编码器和解码器的嵌入维度(embed_dim)必须相同！"

        self.encoder = ViTEncoder(**encoder_cfg)
        self.decoder = TextDecoder(**decoder_cfg)

    def forward(self, image: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        用于训练的前向传播 (使用Teacher Forcing)。
        """
        memory = self.encoder(image)
        logits = self.decoder(tgt, memory)
        return logits

    @torch.no_grad()
    def generate(
            self,
            image: torch.Tensor,
            sos_token_id: int,
            eos_token_id: int,
            max_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        用于推理的自回归生成。

        Args:
            image (torch.Tensor): 输入的单张图像张量，形状 (1, C, H, W)。
            sos_token_id (int): 序列开始（Start-of-Sequence）token的ID。
            eos_token_id (int): 序列结束（End-of-Sequence）token的ID。
            max_len (int, optional): 用户期望的最大生成长度。如果为None，则使用解码器的最大容量。

        Returns:
            torch.Tensor: 生成的token ID序列，形状 (1, SeqLen)。
        """
        self.eval()

        # --- 核心修正处 ---
        # 确定实际的最大生成长度。它不能超过解码器位置编码的容量。
        decoder_max_len = self.decoder.pos_embed.shape[1]
        if max_len is None:
            effective_max_len = decoder_max_len
        else:
            effective_max_len = min(max_len, decoder_max_len)
        # --- 修正结束 ---

        memory = self.encoder(image)
        generated_seq = torch.tensor([[sos_token_id]], device=image.device)

        # 循环 effective_max_len - 1 次，因为我们已经有了一个SOS token
        for _ in range(effective_max_len - 1):
            logits = self.decoder(generated_seq, memory)
            last_logits = logits[:, -1, :]
            next_token_id = torch.argmax(last_logits, dim=-1).unsqueeze(1)
            generated_seq = torch.cat([generated_seq, next_token_id], dim=1)

            if next_token_id.item() == eos_token_id:
                break

        return generated_seq


# 当该文件被直接执行时，运行以下测试代码
# python -m opera_model.architecture.complete_model

if __name__ == '__main__':
    print("--- Testing Complete ImageToText Model (Corrected) ---")

    encoder_config = {
        'img_size': 256, 'patch_size': 32, 'in_channels': 1,
        'embed_dim': 192, 'depth': 4, 'num_heads': 6
    }
    decoder_config = {
        'vocab_size': 100, 'embed_dim': 192, 'depth': 4,
        'num_heads': 6, 'max_seq_len': 50  # 解码器的物理极限是50
    }

    model = ImageToTextModel(encoder_config, decoder_config)
    print("\n[1] Complete Model Instantiated.")

    print("\n--- Testing Training Mode (forward pass) ---")
    batch_size = 4
    images = torch.randn(batch_size, 1, 256, 256)
    target_tokens = torch.randint(0, 100, (batch_size, 20))
    print(f"[2] Input shapes: image={images.shape}, target={target_tokens.shape}")

    try:
        output_logits = model(images, target_tokens)
        print(f"[3] Forward pass successful!")
        print(f"    -> Output logits shape: {output_logits.shape}")
        expected_shape = (batch_size, 20, 100)
        assert output_logits.shape == expected_shape
        print(f"    -> Verified: Output shape matches expected shape {expected_shape}.")
    except Exception as e:
        print(f"An error occurred during training forward pass: {e}")

    print("\n--- Testing Inference Mode (generate method) ---")
    single_image = torch.randn(1, 1, 256, 256)
    SOS_ID = 1
    EOS_ID = 2
    print(f"[4] Input shape for generation: image={single_image.shape}")

    try:
        # 【修正处】在测试时明确传递 max_len，这是个好习惯
        generated_ids = model.generate(
            single_image,
            sos_token_id=SOS_ID,
            eos_token_id=EOS_ID,
            max_len=decoder_config['max_seq_len']  # 告诉它最多生成50个
        )
        print(f"[5] Generation successful!")
        print(f"    -> Generated sequence shape: {generated_ids.shape}")
        print(f"    -> Generated sequence (IDs): {generated_ids.tolist()}")
        assert generated_ids.shape[0] == 1
        assert generated_ids.shape[1] <= decoder_config['max_seq_len']
        print(
            f"    -> Verified: Generated sequence length ({generated_ids.shape[1]}) does not exceed decoder capacity ({decoder_config['max_seq_len']}).")
    except Exception as e:
        print(f"An error occurred during generation: {e}")

    print("\n--- Complete Model Test Complete ---")