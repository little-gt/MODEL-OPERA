# -*- coding: utf-8 -*-
"""
OPERA/opera_model/dataset.py

该模块定义了用于加载、预处理和提供光学数据的PyTorch Dataset类。
它将磁盘上的图像和JSON标签转换为模型可以直接使用的张量。
"""

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Callable
import numpy as np


# 为了使该文件可以独立测试，我们先创建一个虚拟的Tokenizer。
# 在实际项目中，我们会从 `tokenizer.py` 中导入它。
class MockTokenizer:
    def __init__(self, max_len=30):
        self.vocab = {'[PAD]': 0, '[SOS]': 1, '[EOS]': 2, 'a': 3, 'b': 4, 'c': 5}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.sos_token_id = self.vocab['[SOS]']
        self.eos_token_id = self.vocab['[EOS]']
        self.pad_token_id = self.vocab['[PAD]']

    def encode(self, text: str) -> List[int]:
        tokens = [self.sos_token_id]
        for char in text:
            tokens.append(self.vocab.get(char, 0))
        tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        return "".join([self.inv_vocab.get(id, '') for id in token_ids])


class OpticalDataset(Dataset):
    """
    用于加载光学图像和其JSON标签的自定义PyTorch Dataset。
    """

    def __init__(
            self,
            image_dir: str,
            label_dir: str,
            tokenizer: Any,  # 应该是一个Tokenizer对象
            transform: Optional[Callable] = None,
            max_token_len: int = 256
    ):
        """
        初始化数据集。

        Args:
            image_dir (str): 包含图像文件的目录路径。
            label_dir (str): 包含JSON标签文件的目录路径。
            tokenizer (Any): 一个实现了'encode'方法的tokenizer实例。
            transform (Callable, optional): 应用于图像的转换。
            max_token_len (int): 标签token序列的最大允许长度。
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_token_len = max_token_len

        # 获取所有样本的ID (不含扩展名)
        self.sample_ids = [os.path.splitext(f)[0] for f in os.listdir(image_dir)]

        # (可选) 验证每个图像都有一个对应的标签
        for sample_id in self.sample_ids:
            if not os.path.exists(os.path.join(label_dir, f"{sample_id}.json")):
                raise FileNotFoundError(f"标签文件未找到: {sample_id}.json")

    def __len__(self) -> int:
        """返回数据集中的样本总数。"""
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取并预处理一个样本。

        Args:
            idx (int): 样本的索引。

        Returns:
            Dict[str, Any]: 一个包含处理过的图像和标签token的字典。
        """
        # 1. 获取样本ID并构建路径
        sample_id = self.sample_ids[idx]
        img_path = os.path.join(self.image_dir, f"{sample_id}.png")
        lbl_path = os.path.join(self.label_dir, f"{sample_id}.json")

        # 2. 加载图像
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)

        # 3. 加载并处理标签
        with open(lbl_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        label_text = json.dumps(label_data, separators=(',', ':'))
        label_tokens = self.tokenizer.encode(label_text)

        # --- 核心修正 ---
        # 对超长的token序列进行截断
        if len(label_tokens) > self.max_token_len:
            # 截取前面的部分，并确保最后一个token是 [EOS]
            label_tokens = label_tokens[:self.max_token_len - 1] + [self.tokenizer.eos_token_id]

        return {
            'image': image,
            'label_tokens': torch.tensor(label_tokens, dtype=torch.long)
        }


# 当该文件被直接执行时，运行以下测试代码
if __name__ == '__main__':
    from torchvision import transforms

    print("--- Testing OpticalDataset (Corrected) ---")

    TEST_DIR = "temp_test_dataset"
    IMG_DIR = os.path.join(TEST_DIR, "images")
    LBL_DIR = os.path.join(TEST_DIR, "labels")
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LBL_DIR, exist_ok=True)

    # 在测试中也给MockTokenizer一个最大长度
    mock_tokenizer = MockTokenizer(max_len=30)

    # 创建一个正常样本和一个超长样本
    fake_img = Image.fromarray((np.random.rand(64, 64) * 255).astype(np.uint8))
    fake_img.save(os.path.join(IMG_DIR, f"sample_0.png"))
    fake_lbl = {'desc': 'abc'}  # 正常长度
    with open(os.path.join(LBL_DIR, f"sample_0.json"), 'w') as f:
        json.dump(fake_lbl, f)

    fake_img.save(os.path.join(IMG_DIR, f"sample_1.png"))
    fake_lbl_long = {'desc': 'a' * 50}  # 超长
    with open(os.path.join(LBL_DIR, f"sample_1.json"), 'w') as f:
        json.dump(fake_lbl_long, f)

    print(f"\n[1] Created temporary dataset with normal and long samples.")

    img_transform = transforms.Compose([transforms.ToTensor()])

    try:
        # 实例化数据集，并告知它最大长度为30
        dataset = OpticalDataset(
            image_dir=IMG_DIR,
            label_dir=LBL_DIR,
            tokenizer=mock_tokenizer,
            transform=img_transform,
            max_token_len=30
        )
        print("\n[2] OpticalDataset instantiated successfully with max_token_len=30.")

        # 检查正常样本
        normal_sample = dataset[0]
        normal_len = len(normal_sample['label_tokens'])
        print(f"\n[3] Fetched normal sample. Token length: {normal_len}")
        assert normal_len < 30

        # 检查超长样本
        long_sample = dataset[1]
        long_len = len(long_sample['label_tokens'])
        print(f"[4] Fetched long sample. Token length: {long_len}")
        assert long_len == 30
        assert long_sample['label_tokens'][-1] == mock_tokenizer.eos_token_id
        print("    -> SUCCESS: Long sample was correctly truncated to 30 tokens.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        import shutil

        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
            print(f"\n[5] Cleaned up temporary directory.")

    print("\n--- OpticalDataset Test Complete ---")