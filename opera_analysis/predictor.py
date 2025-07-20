# -*- coding: utf-8 -*-
"""
OPERA/opera_analysis/predictor.py

该模块提供了Predictor类，用于加载训练好的模型并对新图像进行分析。
"""
import torch
from torchvision import transforms
from PIL import Image
import json
from typing import Dict, Any

# 导入必要的模块
from opera_model.architecture.complete_model import ImageToTextModel
from opera_model.tokenizer import Tokenizer


class Predictor:
    """
    一个用于加载模型并执行单张图像推理的类。
    """

    def __init__(self, model: ImageToTextModel, tokenizer: Tokenizer, device: torch.device):
        self.model = model.to(device).eval()  # 确保模型在评估模式
        self.tokenizer = tokenizer
        self.device = device

        # 使用与训练时完全相同的图像变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        加载、预处理并分析一张图像。

        Args:
            image_path (str): 要分析的图像文件的路径。

        Returns:
            Dict[str, Any]: 一个包含模型预测结果的字典。
        """
        # 1. 加载和预处理图像
        image = Image.open(image_path).convert("L")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 2. 使用模型的generate方法进行自回归预测
        generated_ids = self.model.generate(
            image_tensor,
            sos_token_id=self.tokenizer.sos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 3. 将生成的token ID解码为文本
        generated_text = self.tokenizer.decode(generated_ids.squeeze(0).cpu().tolist())

        # 4. 尝试将解码出的文本解析为JSON
        try:
            # 去除可能存在的前后多余字符
            clean_text = generated_text.strip().replace("'", '"')
            predicted_data = json.loads(clean_text)
        except json.JSONDecodeError:
            print("警告: 模型输出的不是一个有效的JSON字符串。")
            print(f"原始输出: '{generated_text}'")
            predicted_data = {"error": "Invalid JSON output", "raw_output": generated_text}

        return predicted_data