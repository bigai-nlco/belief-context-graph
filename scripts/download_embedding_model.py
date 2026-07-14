#!/usr/bin/env python3
"""
下载并检查 SFR-Embedding-2_R 模型大小

使用方法：
1. 在 qwen3 环境中安装依赖：pip install sentence-transformers
2. 运行此脚本：python scripts/download_embedding_model.py
"""

import os
import sys
from pathlib import Path

def get_dir_size(path):
    """计算目录大小（递归）"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(entry.path)
    except PermissionError:
        pass
    return total

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("错误：未安装 sentence-transformers")
        print("请先运行：pip install sentence-transformers")
        sys.exit(1)

    model_name = 'Salesforce/SFR-Embedding-2_R'
    print(f"正在下载模型: {model_name}")
    print("=" * 60)

    # 下载模型
    try:
        model = SentenceTransformer(model_name)
        print("✓ 模型下载完成")
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        sys.exit(1)

    # 检查模型信息
    print("\n模型信息:")
    print("-" * 60)
    print(f"模型名称: {model_name}")
    print(f"最大序列长度: {model.max_seq_length}")
    print(f"嵌入维度: {model.get_sentence_embedding_dimension()}")

    # 检查模型大小
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
    print(f"\n模型缓存目录: {cache_dir}")

    # 查找模型目录
    model_dirs = list(cache_dir.glob('models--Salesforce--SFR-Embedding-2_R*'))
    if model_dirs:
        model_dir = model_dirs[0]
        size = get_dir_size(model_dir)
        print(f"模型目录: {model_dir}")
        print(f"模型大小: {format_size(size)}")
    else:
        print("未找到模型目录")

    # 测试推理速度（CPU vs GPU）
    print("\n性能测试:")
    print("-" * 60)

    import torch
    import time

    test_texts = ["This is a test sentence."] * 10

    # CPU 测试
    print("测试 CPU 推理速度...")
    model_cpu = SentenceTransformer(model_name, device='cpu')
    start = time.time()
    _ = model_cpu.encode(test_texts, batch_size=4, show_progress_bar=False)
    cpu_time = time.time() - start
    print(f"  CPU: {cpu_time:.2f} 秒 (10个句子)")

    # GPU 测试（如果可用）
    if torch.cuda.is_available():
        print("测试 GPU 推理速度...")
        model_gpu = SentenceTransformer(model_name, device='cuda')
        torch.cuda.synchronize()
        start = time.time()
        _ = model_gpu.encode(test_texts, batch_size=4, show_progress_bar=False)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"  GPU: {gpu_time:.2f} 秒 (10个句子)")
        print(f"  加速比: {cpu_time/gpu_time:.1f}x")
    else:
        print("  GPU 不可用")

    print("\n建议:")
    print("-" * 60)
    if torch.cuda.is_available():
        print("✓ 推荐使用 GPU (快 5-10 倍)")
        print(f"✓ 可用 GPU: {torch.cuda.device_count()} 个")
        # 检查哪个 GPU 显存最多
        free_mem = []
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory
            free = mem - torch.cuda.memory_allocated(i)
            free_mem.append((i, free / 1024**3))

        best_gpu = max(free_mem, key=lambda x: x[1])
        print(f"✓ 建议使用 GPU {best_gpu[0]} (空闲显存最多: {best_gpu[1]:.1f} GB)")
        print(f"\n配置示例:")
        print(f"  --hero_embedding_device cuda:{best_gpu[0]}")
    else:
        print("⚠ 可使用 CPU，但速度较慢")
        print("⚠ 对于 500 个 claim，CPU 可能需要几小时")
        print("\n配置示例:")
        print("  --hero_embedding_device cpu")

if __name__ == '__main__':
    main()
