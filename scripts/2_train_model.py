# -*- coding: utf-8 -*-
"""
OPERA/scripts/2_train_model.py

该脚本是OPERA项目的第二个执行步骤。
它整合了数据、模型和训练引擎，以执行完整的模型训练流程。
"""
import argparse
import yaml
import json
from pathlib import Path
from typing import List, Dict

# 1. 拓宽Python的模块搜索路径
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 2. 导入所有必要的模块
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from opera_model.architecture.complete_model import ImageToTextModel
from opera_model.dataset import OpticalDataset
from opera_model.tokenizer import Tokenizer
from opera_model.engine import Trainer

# 3. 定义不同尺寸的模型配置
MODEL_CONFIGS = {
    'small': {
        'encoder_cfg': {'img_size': 256, 'patch_size': 32, 'in_channels': 1, 'embed_dim': 192, 'depth': 4,
                        'num_heads': 6},
        'decoder_cfg': {'embed_dim': 192, 'depth': 4, 'num_heads': 6, 'max_seq_len': 256}
    },
    'base': {
        'encoder_cfg': {'img_size': 256, 'patch_size': 16, 'in_channels': 1, 'embed_dim': 768, 'depth': 6,
                        'num_heads': 8},
        'decoder_cfg': {'embed_dim': 768, 'depth': 6, 'num_heads': 8, 'max_seq_len': 512}
    }
}


def get_tokenizer(config: Dict) -> Tokenizer:
    """加载或构建并保存一个Tokenizer。"""
    tokenizer_path = Path(config['tokenizer_path'])
    if tokenizer_path.exists():
        print(f"从 '{tokenizer_path}' 加载已存在的Tokenizer...")
        return Tokenizer.load(str(tokenizer_path))
    else:
        print("未找到Tokenizer，从数据集中构建一个新的...")
        label_dir = Path(config['data_path']) / 'labels'

        def label_corpus_generator(directory: Path):
            for filename in directory.glob("*.json"):
                with open(filename, 'r', encoding='utf-8') as f:
                    yield json.dumps(json.load(f), separators=(',', ':'))

        corpus = label_corpus_generator(label_dir)
        special_tokens = ['[PAD]', '[SOS]', '[EOS]', '[UNK]']

        tokenizer = Tokenizer.from_corpus(corpus, special_tokens)
        tokenizer.save(str(tokenizer_path))
        print(f"新Tokenizer构建完成并保存到 '{tokenizer_path}'。词汇表大小: {tokenizer.vocab_size}")
        return tokenizer


def create_dataloaders(config: Dict, tokenizer: Tokenizer) -> tuple:
    """创建训练和验证数据加载器。"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # --- 核心修正处 ---
    # 从模型配置中获取解码器能处理的最大序列长度
    model_cfg = MODEL_CONFIGS[config['model_size']]
    max_len = model_cfg['decoder_cfg']['max_seq_len']

    # 实例化完整的数据集，并将最大长度限制传递给它
    full_dataset = OpticalDataset(
        image_dir=str(Path(config['data_path']) / 'images'),
        label_dir=str(Path(config['data_path']) / 'labels'),
        tokenizer=tokenizer,
        transform=transform,
        max_token_len=max_len  # 确保数据集截断超长序列
    )
    # --------------------

    train_size = int(config['train_val_split_ratio'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"数据集划分完成: {len(train_dataset)}个训练样本, {len(val_dataset)}个验证样本。")

    def collate_fn(batch: List[Dict]):
        images = [item['image'] for item in batch]
        labels = [item['label_tokens'] for item in batch]
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        return {'image': torch.stack(images), 'label_tokens': padded_labels}

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], collate_fn=collate_fn)
    return train_loader, val_loader


def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description="从配置文件训练OPERA模型。")
    parser.add_argument('--config', type=str, required=True, help="指向 training_config_vX.yaml 的路径。")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("--- 训练配置 ---");
    print(yaml.dump(config, indent=2));
    print("--------------------")

    device_str = config['device']
    if device_str == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"\n设备已设置为: {device}")

    tokenizer = get_tokenizer(config)
    train_loader, val_loader = create_dataloaders(config, tokenizer)

    model_size = config['model_size']
    encoder_cfg = MODEL_CONFIGS[model_size]['encoder_cfg']
    decoder_cfg = MODEL_CONFIGS[model_size]['decoder_cfg']
    decoder_cfg['vocab_size'] = tokenizer.vocab_size

    model = ImageToTextModel(encoder_cfg, decoder_cfg)
    print(f"模型 '{model_size}' 已创建。参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=config['checkpoint_dir']
    )

    print("\n--- 开始训练 ---")
    trainer.train(num_epochs=config['num_epochs'])
    print("\n--- 训练完成 ---")


if __name__ == '__main__':
    main()