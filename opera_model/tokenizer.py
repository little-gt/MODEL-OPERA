# -*- coding: utf-8 -*-
"""
OPERA/opera_model/tokenizer.py

该模块实现了用于将文本标签转换为token ID的Tokenizer。
它基于字符级别，并能从数据集中构建词汇表。
"""
import json
import os
from typing import List, Iterable


class Tokenizer:
    """
    一个简单的字符级Tokenizer，用于将结构化的JSON标签进行编码和解码。
    """

    def __init__(self, vocab: dict, special_tokens: List[str]):
        """
        直接初始化Tokenizer。通常推荐使用 .from_corpus() 或 .load()。
        """
        self.vocab = vocab
        self.inv_vocab = {idx: token for token, idx in vocab.items()}

        for token in special_tokens:
            if token not in self.vocab:
                raise ValueError(f"特殊token '{token}' 未在词汇表中找到。")

        # 将特殊token的ID保存为类的属性，方便外部调用
        self.pad_token_id = self.vocab['[PAD]']
        self.sos_token_id = self.vocab['[SOS]']
        self.eos_token_id = self.vocab['[EOS]']
        self.unk_token_id = self.vocab['[UNK]']
        # 保存special_tokens列表本身，以便后续使用
        self._special_tokens = special_tokens

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def from_corpus(cls, corpus: Iterable[str], special_tokens: List[str]):
        """
        从一个文本语料库构建Tokenizer。
        """
        unique_chars = set()
        for text in corpus:
            unique_chars.update(text)

        vocab = {}
        for token in special_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        for char in sorted(list(unique_chars)):
            if char not in vocab:
                vocab[char] = len(vocab)

        return cls(vocab, special_tokens)

    def save(self, file_path: str):
        """将词汇表和配置保存到JSON文件。"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        config = {
            'vocab': self.vocab,
            'special_tokens': self._special_tokens
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, file_path: str):
        """从文件加载Tokenizer。"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return cls(config['vocab'], config['special_tokens'])

    def encode(self, text: str) -> List[int]:
        """将字符串编码为token ID列表，并添加SOS/EOS。"""
        tokens = [self.sos_token_id]
        for char in text:
            tokens.append(self.vocab.get(char, self.unk_token_id))
        tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """将token ID列表解码为字符串，在EOS处停止。"""
        text = ""
        for token_id in token_ids:
            if token_id in [self.sos_token_id, self.pad_token_id]:
                continue
            if token_id == self.eos_token_id:
                break
            text += self.inv_vocab.get(token_id, '')
        return text


# 当该文件被直接执行时，运行以下测试代码
if __name__ == '__main__':
    print("--- Testing Tokenizer (Corrected) ---")

    corpus = [
        '{"name":"airy","params":{"d":1.2}}',
        '{"name":"slit","params":{"w":0.5}}',
    ]
    special_tokens = ['[PAD]', '[SOS]', '[EOS]', '[UNK]']
    print("\n[1] Sample corpus created.")

    tokenizer = Tokenizer.from_corpus(corpus, special_tokens)
    print("\n[2] Tokenizer built from corpus.")
    print(f"    -> Vocabulary size: {tokenizer.vocab_size}")

    print("\n[3] Testing encode/decode process...")
    original_text = '{"name":"airy"}'
    encoded_ids = tokenizer.encode(original_text)
    decoded_text = tokenizer.decode(encoded_ids)

    print(f"    -> Original text:  '{original_text}'")
    print(f"    -> Encoded IDs:    {encoded_ids}")
    print(f"    -> Decoded text:   '{decoded_text}'")
    assert original_text == decoded_text
    print("    -> SUCCESS: Decode(Encode(text)) == text")

    print("\n[4] Testing save/load process...")
    SAVE_PATH = "temp_test_tokenizer/vocab.json"
    tokenizer.save(SAVE_PATH)
    print(f"    -> Tokenizer saved to '{SAVE_PATH}'")

    loaded_tokenizer = Tokenizer.load(SAVE_PATH)
    print(f"    -> Tokenizer loaded from '{SAVE_PATH}'")

    assert tokenizer.vocab == loaded_tokenizer.vocab
    print("    -> SUCCESS: Loaded tokenizer is identical to the original.")

    import shutil

    if os.path.exists(os.path.dirname(SAVE_PATH)):
        shutil.rmtree(os.path.dirname(SAVE_PATH))
        print(f"\n[5] Cleaned up temporary directory.")

    print("\n--- Tokenizer Test Complete ---")