# -*- coding: utf-8 -*-
"""
OPERA/opera_model/engine.py

该模块包含了核心的训练器(Trainer)类。
它负责将模型、数据、优化器和损失函数整合在一起，
并执行完整的训练和验证循环。
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # 一个漂亮的进度条库
from typing import Dict, Any

# 为了使该文件可以独立测试，我们从其他模块导入必要的类
from opera_model.architecture.complete_model import ImageToTextModel
from opera_model.tokenizer import Tokenizer


class Trainer:
    """
    一个用于训练图像到文本模型的训练器类。
    """

    def __init__(
            self,
            model: ImageToTextModel,
            tokenizer: Tokenizer,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            device: torch.device,
            checkpoint_dir: str = "./cache/checkpoints/"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.epoch = 0
        self.best_val_loss = float('inf')

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """准备一个批次的数据，创建模型的输入和目标。"""
        images = batch['image'].to(self.device)
        label_tokens = batch['label_tokens'].to(self.device)

        # Teacher Forcing:
        # target_input 是解码器的输入 (除了最后一个token)
        # target_output 是我们期望的预测结果 (除了第一个token)
        target_input = label_tokens[:, :-1]
        target_output = label_tokens[:, 1:]

        return images, target_input, target_output

    def _train_one_epoch(self) -> float:
        """执行一个完整的训练轮次。"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1} [Training]")
        for batch in pbar:
            images, tgt_in, tgt_out = self._prepare_batch(batch)

            # 前向传播
            logits = self.model(images, tgt_in)

            # 计算损失
            # CrossEntropyLoss期望 (B*T, V) 和 (B*T)
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]),
                tgt_out.reshape(-1)
            )

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def _validate_one_epoch(self) -> float:
        """执行一个完整的验证轮次。"""
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch + 1} [Validation]")
        with torch.no_grad():
            for batch in pbar:
                images, tgt_in, tgt_out = self._prepare_batch(batch)
                logits = self.model(images, tgt_in)
                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]),
                    tgt_out.reshape(-1)
                )
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, is_best: bool):
        """保存模型检查点。"""
        state = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        filename = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(state, filename)
        if is_best:
            best_filename = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_filename)

    def load_checkpoint(self):
        """加载最新的模型检查点。"""
        filename = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        if os.path.exists(filename):
            print(f"Loading checkpoint from '{filename}'...")
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
        else:
            print("No checkpoint found, starting from scratch.")

    def train(self, num_epochs: int):
        """启动完整的训练流程。"""
        self.load_checkpoint()  # 尝试从检查点恢复

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            train_loss = self._train_one_epoch()
            val_loss = self._validate_one_epoch()

            print(f"Epoch {self.epoch + 1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print("New best model found!")

            self.save_checkpoint(is_best)


# 当该文件被直接执行时，运行一个“烟雾测试” (Smoke Test)
if __name__ == '__main__':
    from torch.utils.data import TensorDataset

    print("--- Smoke Test for Trainer ---")

    # 1. 创建虚拟组件
    print("[1] Creating dummy components...")
    # 虚拟Tokenizer
    dummy_tokenizer = Tokenizer.from_corpus(["abc"], ["[PAD]", "[SOS]", "[EOS]", "[UNK]"])
    # 虚拟模型
    encoder_cfg = {'img_size': 32, 'patch_size': 16, 'embed_dim': 64}
    decoder_cfg = {'vocab_size': dummy_tokenizer.vocab_size, 'embed_dim': 64, 'max_seq_len': 20}
    dummy_model = ImageToTextModel(encoder_cfg, decoder_cfg)
    # 虚拟数据加载器
    dummy_images = torch.randn(10, 1, 32, 32)
    dummy_labels = torch.randint(0, dummy_tokenizer.vocab_size, (10, 15))
    dummy_dataset = TensorDataset(dummy_images, dummy_labels)


    # Collate function to format data like our OpticalDataset
    def collate_fn(batch):
        images, labels = zip(*batch)
        return {'image': torch.stack(images), 'label_tokens': torch.stack(labels)}


    dummy_loader = DataLoader(dummy_dataset, batch_size=2, collate_fn=collate_fn)
    # 虚拟优化器和损失函数
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1e-4)
    dummy_criterion = nn.CrossEntropyLoss(ignore_index=dummy_tokenizer.pad_token_id)

    # 2. 实例化并测试训练器
    print("[2] Instantiating Trainer...")
    trainer = Trainer(
        model=dummy_model,
        tokenizer=dummy_tokenizer,
        train_loader=dummy_loader,
        val_loader=dummy_loader,  # 用同一个loader进行简单测试
        optimizer=dummy_optimizer,
        criterion=dummy_criterion,
        device=torch.device('cpu'),
        checkpoint_dir="./temp_test_checkpoints/"
    )

    # 3. 运行一个短暂的训练
    print("[3] Running a short training loop (2 epochs)...")
    try:
        trainer.train(num_epochs=2)
        print("\nSmoke test passed: Training loop completed without errors.")
    except Exception as e:
        print(f"\nSmoke test failed: {e}")
    finally:
        # 4. 清理临时文件
        import shutil

        if os.path.exists("./temp_test_checkpoints/"):
            shutil.rmtree("./temp_test_checkpoints/")
            print("[4] Cleaned up temporary checkpoint directory.")

    print("\n--- Trainer Smoke Test Complete ---")