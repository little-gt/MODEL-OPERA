# -*- coding: utf-8 -*-
"""
OPERA/scripts/1_generate_dataset.py
python scripts/1_generate_dataset.py --config configs/data_generation_v1.yaml

该脚本是OPERA项目的第一个执行步骤。
它读取一个配置文件，并调用数据生成引擎来创建
一个完整的数据集，用于后续的模型训练。
"""
import argparse
import yaml
from pathlib import Path

# --- 拓宽Python的模块搜索路径 ---
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# -----------------------------------------

from opera_data.generator import DatasetGenerator


def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description="从配置文件生成光学图像数据集。")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="指向 data_generation_vX.yaml 配置文件的路径。"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    # --- 核心修正处 ---
    # 在打开文件时，明确指定使用 utf-8 编码
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # -------------------

    print("--- 数据生成配置 ---")
    print(yaml.dump(config, indent=2))
    print("--------------------")

    output_path = config['output_path']
    num_samples = config['num_samples']
    resolution = config['image_resolution']

    print(f"\n初始化数据生成器，分辨率为 {resolution}x{resolution}...")
    generator = DatasetGenerator(resolution_px=resolution)

    print(f"开始生成 {num_samples} 个样本到 '{output_path}'...")
    generator.generate_and_save_dataset(
        num_samples=num_samples,
        save_path=output_path
    )

    print("\n数据集生成完成！")


if __name__ == '__main__':
    main()