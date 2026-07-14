#!/usr/bin/env python3
"""简单测试：下载模型并测试 CPU 推理速度"""

import time

print("=" * 60)
print("SFR-Embedding-2_R 模型下载和测试")
print("=" * 60)

# Step 1: 检查依赖
print("\n[1/4] 检查依赖...")
try:
    from sentence_transformers import SentenceTransformer
    import torch
    print("✓ sentence_transformers 已安装")
    print("✓ torch 已安装")
except ImportError as e:
    print(f"✗ 缺少依赖: {e}")
    print("\n请先安装：pip install sentence-transformers")
    exit(1)

# Step 2: 下载模型
print("\n[2/4] 下载模型...")
model_name = 'Salesforce/SFR-Embedding-2_R'
try:
    model = SentenceTransformer(model_name, device='cpu')
    print(f"✓ 模型下载完成: {model_name}")
    print(f"  - 最大序列长度: {model.max_seq_length}")
    print(f"  - 嵌入维度: {model.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"✗ 下载失败: {e}")
    exit(1)

# Step 3: 测试推理速度（CPU）
print("\n[3/4] 测试 CPU 推理速度...")
test_texts = [
    "This is a test sentence for embedding.",
    "Another example sentence to test the model.",
    "Testing the inference speed on CPU.",
] * 10  # 30 个句子

try:
    # 预热
    _ = model.encode(["warmup"], show_progress_bar=False)

    # 正式测试
    start = time.time()
    embeddings = model.encode(test_texts, batch_size=4, show_progress_bar=False)
    elapsed = time.time() - start

    print(f"✓ CPU 推理完成")
    print(f"  - 句子数量: {len(test_texts)}")
    print(f"  - 总耗时: {elapsed:.2f} 秒")
    print(f"  - 平均速度: {elapsed/len(test_texts)*1000:.1f} ms/句")
    print(f"  - 输出维度: {embeddings.shape}")
except Exception as e:
    print(f"✗ 推理失败: {e}")
    exit(1)

# Step 4: 估算实验耗时
print("\n[4/4] 估算实验耗时...")
time_per_sentence = elapsed / len(test_texts)

# 每个 claim 平均检索 10000 个句子
sentences_per_claim = 10000
time_per_claim = sentences_per_claim * time_per_sentence

print(f"  - 单个 claim 预估: {time_per_claim:.1f} 秒")
print(f"  - 10 个 claim: {time_per_claim * 10 / 60:.1f} 分钟")
print(f"  - 50 个 claim: {time_per_claim * 50 / 60:.1f} 分钟")
print(f"  - 500 个 claim: {time_per_claim * 500 / 60:.1f} 分钟 = {time_per_claim * 500 / 3600:.1f} 小时")

print("\n" + "=" * 60)
print("测试完成！")
print("\n建议:")
if time_per_claim < 5:
    print("✓ CPU 速度可以接受，可以用于小规模实验")
else:
    print("⚠ CPU 速度较慢，推荐使用 GPU:")
    print("  bcg agent run averitec --hero-embedding-device cuda:7")
print("=" * 60)
