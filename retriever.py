#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import json
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class BGERetriever:
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = "./bge-large-en-v1.5",
        device: str = "cuda",
        cache_dir: str = "./retriever_cache",
        corpus_dir: str = "./wiki_dpr_text_only",
    ):
        self.device = device
        self.cache_dir = cache_dir
        self.corpus_dir = corpus_dir
        os.makedirs(cache_dir, exist_ok=True)

        print(f"[Retriever] 加载模型 {model_name} ...")
        self.model = SentenceTransformer(model_name, device=device)
        self.emb_dim = self.model.get_sentence_embedding_dimension()

        self.corpus = None
        self.index  = None
        self._load_or_build_corpus()

    def _load_or_build_corpus(self):
        corpus_path = os.path.join(self.cache_dir, "corpus.pkl")
        index_path  = os.path.join(self.cache_dir, "faiss_bge.index")

        if os.path.exists(corpus_path) and os.path.exists(index_path):
            print("[Retriever] 从缓存加载语料库与索引 ...")
            with open(corpus_path, "rb") as f:
                self.corpus = pickle.load(f)
            self.index = faiss.read_index(index_path)
            print(f"[Retriever] 语料库大小：{len(self.corpus):,}，索引维度：{self.emb_dim}")
            return

        print("[Retriever] 从 parquet 文件加载语料库 ...")
        parquet_files = sorted(glob.glob(os.path.join(self.corpus_dir, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(
                f"在 {self.corpus_dir} 下找不到 parquet 文件，"
                f"请确认 wiki_dpr 已下载到该目录。"
            )
        print(f"[Retriever] 找到 {len(parquet_files)} 个 parquet 文件，开始读取 ...")

        self.corpus = []
        for pf in tqdm(parquet_files, desc="读取 parquet"):
            df = pd.read_parquet(pf, columns=["id", "title", "text"])
            for row in df.itertuples(index=False):
                self.corpus.append({
                    "id":    str(row.id),
                    "title": str(row.title) if row.title else "",
                    "text":  str(row.text)  if row.text  else "",
                })

        print(f"[Retriever] 语料库加载完毕，共 {len(self.corpus):,} 条。")
        self._build_faiss_index(batch_size=8192, checkpoint_every=20)

        print("[Retriever] 保存语料库缓存 ...")
        with open(corpus_path, "wb") as f:
            pickle.dump(self.corpus, f)
        faiss.write_index(self.index, index_path)
        print("[Retriever] 索引构建完毕并已缓存。")

        for p in [
            os.path.join(self.cache_dir, "build_progress.json"),
            os.path.join(self.cache_dir, "faiss_bge.index.tmp"),
        ]:
            if os.path.exists(p):
                os.remove(p)

    def _build_faiss_index(self, batch_size: int = 8192, checkpoint_every: int = 20):
        progress_path  = os.path.join(self.cache_dir, "build_progress.json")
        index_tmp_path = os.path.join(self.cache_dir, "faiss_bge.index.tmp")

        texts = [f"{item['title']} {item['text']}" for item in self.corpus]
        start_idx = 0

        if os.path.exists(progress_path) and os.path.exists(index_tmp_path):
            try:
                with open(progress_path, "r") as f:
                    progress = json.load(f)
                saved_idx = progress.get("last_processed_idx", 0)
                if 0 < saved_idx < len(texts):
                    self.index = faiss.read_index(index_tmp_path)
                    start_idx = saved_idx
                    print(f"[Checkpoint] 从第 {start_idx:,} / {len(texts):,} 条恢复构建 ...")
                else:
                    self.index = faiss.IndexFlatIP(self.emb_dim)
            except Exception as e:
                print(f"[Checkpoint] 恢复失败（{e}），从头开始。")
                self.index = faiss.IndexFlatIP(self.emb_dim)
        else:
            self.index = faiss.IndexFlatIP(self.emb_dim)

        total_batches = (len(texts) + batch_size - 1) // batch_size
        start_batch   = start_idx // batch_size
        batches_since_ckpt = 0

        pbar = tqdm(
            range(start_idx, len(texts), batch_size),
            desc="构建 FAISS 索引",
            total=total_batches,
            initial=start_batch,
        )

        for i in pbar:
            batch = texts[i : i + batch_size]
            embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device=self.device,
            )
            self.index.add(embeddings.astype("float32"))
            batches_since_ckpt += 1

            if batches_since_ckpt >= checkpoint_every:
                faiss.write_index(self.index, index_tmp_path)
                with open(progress_path, "w") as f:
                    json.dump({"last_processed_idx": i + len(batch)}, f)
                try:
                    os.sync()
                except AttributeError:
                    pass
                batches_since_ckpt = 0
                pbar.set_postfix({"已保存": f"{i + len(batch):,}/{len(texts):,}"})

    def retrieve(self, queries: list, top_k: int = 200, batch_size: int = 32):
        all_results = []
        for i in tqdm(range(0, len(queries), batch_size), desc="检索中", disable=True):
            batch = [
                self.QUERY_INSTRUCTION + q
                for q in queries[i : i + batch_size]
            ]
            embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            scores, indices = self.index.search(embeddings.astype("float32"), top_k)
            for q_scores, q_indices in zip(scores, indices):
                results = []
                for rank, (s, idx) in enumerate(zip(q_scores, q_indices), start=1):
                    item = self.corpus[idx]
                    results.append({
                        "id":    item["id"],
                        "title": item["title"],
                        "text":  item["text"],
                        "score": float(s),
                        "rank":  rank,
                    })
                all_results.append(results)
        return all_results

    def retrieve_with_embeddings(
        self, queries: list, top_k: int = 200, batch_size: int = 32
    ):
        all_results = []
        all_doc_embeddings = []
        for i in tqdm(range(0, len(queries), batch_size), desc="检索中(MMR)", disable=True):
            batch = [
                self.QUERY_INSTRUCTION + q
                for q in queries[i : i + batch_size]
            ]
            embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            scores, indices = self.index.search(embeddings.astype("float32"), top_k)
            for q_scores, q_indices in zip(scores, indices):
                results = []
                doc_embs = []
                for rank, (s, idx) in enumerate(zip(q_scores, q_indices), start=1):
                    item = self.corpus[idx]
                    results.append({
                        "id":    item["id"],
                        "title": item["title"],
                        "text":  item["text"],
                        "score": float(s),
                        "rank":  rank,
                    })
                    doc_embs.append(self.index.reconstruct(int(idx)))
                all_results.append(results)
                all_doc_embeddings.append(doc_embs)
        return all_results, all_doc_embeddings