#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据下载脚本（整合版）
运行顺序：先跑这个脚本下载全部依赖资源，再跑 generate.py
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Step 2：下载 BGE 检索模型（如果还没下）
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists("./bge-large-en-v1.5"):
    from sentence_transformers import SentenceTransformer
    print("下载 BAAI/bge-large-en-v1.5 ...")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    model.save('./bge-large-en-v1.5')
    print("BGE 下载完毕。")
else:
    print("bge-large-en-v1.5 已存在，跳过。")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3：下载 RoBERTa-large（用于 BERTScore 评测）
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists("./roberta-large"):
    from transformers import AutoModel, AutoTokenizer
    print("下载 roberta-large ...")
    model = AutoModel.from_pretrained("roberta-large")
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    model.save_pretrained("./roberta-large")
    tokenizer.save_pretrained("./roberta-large")
    print("RoBERTa 下载完毕。")
else:
    print("roberta-large 已存在，跳过。")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7：下载 NLTK 资源（srag_selector.py 需要）
# ─────────────────────────────────────────────────────────────────────────────
import nltk
for pkg, path in [
    ('punkt_tab',                        'tokenizers/punkt_tab'),
    ('averaged_perceptron_tagger_eng',   'taggers/averaged_perceptron_tagger_eng'),
    ('wordnet',                          'corpora/wordnet'),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=False)
print("NLTK 资源就绪。")

print("\n✅ 所有资源下载完毕！")