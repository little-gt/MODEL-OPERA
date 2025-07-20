# -*- coding: utf-8 -*-
"""
OPERA/scripts/3_analyze_image.py
python scripts/3_analyze_image.py --image data/synthetic_dataset_v1/images/syn_xxxxxxxx.png --config configs/training_config_v1.yaml

该脚本是OPERA项目的第三个执行步骤。
它加载一个训练好的模型，并用它来分析一张新的图像，
输出其预测的物理组成。
"""
import argparse
import yaml
import json
from pathlib import Path

# 1. 拓宽Python的模块搜索路径
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 2. 导入所有必要的模块
import torch
from opera_model.architecture.complete_model import ImageToTextModel
from opera_model.tokenizer import Tokenizer
from opera_analysis.predictor import Predictor

# --- 核心修正处 ---
# 将共享的MODEL_CONFIGS字典直接复制到这个文件中，而不是尝试导入它。
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


# --------------------

def main():
    parser = argparse.ArgumentParser(description="使用训练好的OPERA模型分析一张图像。")
    parser.add_argument('--image', type=str, required=True, help="要分析的图像文件的路径。")
    parser.add_argument('--config', type=str, required=True, help="指向用于训练的 training_config_vX.yaml 的路径。")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    checkpoint_path = Path(config['checkpoint_dir']) / 'best_model.pth'
    tokenizer_path = Path(config['tokenizer_path'])
    device = torch.device("cuda" if torch.cuda.is_available() and config['device'] != 'cpu' else 'cpu')

    print(f"--- 分析准备 ---")
    print(f"设备: {device}")
    print(f"加载Tokenizer: {tokenizer_path}")
    print(f"加载模型检查点: {checkpoint_path}")
    print("--------------------")

    if not Path(args.image).exists(): raise FileNotFoundError(f"输入图像未找到: {args.image}")
    if not tokenizer_path.exists(): raise FileNotFoundError(f"Tokenizer文件未找到: {tokenizer_path}")
    if not checkpoint_path.exists(): raise FileNotFoundError(f"模型检查点未找到: {checkpoint_path}")

    tokenizer = Tokenizer.load(str(tokenizer_path))

    model_size = config['model_size']
    encoder_cfg = MODEL_CONFIGS[model_size]['encoder_cfg']
    decoder_cfg = MODEL_CONFIGS[model_size]['decoder_cfg']
    decoder_cfg['vocab_size'] = tokenizer.vocab_size

    model = ImageToTextModel(encoder_cfg, decoder_cfg)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    predictor = Predictor(model=model, tokenizer=tokenizer, device=device)

    print(f"\n>>> 正在分析图像: {args.image}...")
    prediction = predictor.predict(args.image)

    print("\n>>> 模型预测结果:")
    print(json.dumps(prediction, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()