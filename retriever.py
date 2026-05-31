import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import json
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, load_from_disk


class BGERetriever:
    """
    论文配置:
        - Encoder: BAAI/bge-large-en-v1.5 (1024-dim)
        - Corpus: wiki_dpr / psgs_w100 (DPR Wikipedia passages)
        - top_k: 200
    """

    def __init__(
        self,
        corpus_name: str = "wiki_dpr",
        corpus_config: str = "psgs_w100",
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cuda",
        cache_dir: str = "./retriever_cache",
    ):
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        print(f"[Retriever] Loading model {model_name} ...")
        self.model = SentenceTransformer(model_name, device=device)
        self.emb_dim = self.model.get_sentence_embedding_dimension()  

        self.corpus = None    
        self.index = None    
        self._load_or_build_corpus(corpus_name, corpus_config)

    def _load_or_build_corpus(self, corpus_name: str, corpus_config: str):
        corpus_path = os.path.join(self.cache_dir, "corpus.pkl")
        index_path = os.path.join(self.cache_dir, "faiss_bge.index")
        ids_path = os.path.join(self.cache_dir, "corpus_ids.pkl")
        progress_path = os.path.join(self.cache_dir, "build_progress.json")
        index_tmp_path = os.path.join(self.cache_dir, "faiss_bge.index.tmp")

        # 1. 如果完整缓存已存在，直接加载
        if os.path.exists(corpus_path) and os.path.exists(index_path) and os.path.exists(ids_path):
            with open(corpus_path, "rb") as f:
                self.corpus = pickle.load(f)
            with open(ids_path, "rb") as f:
                self.corpus_ids = pickle.load(f)
            self.index = faiss.read_index(index_path)
            # 清理可能残留的临时文件
            for p in [progress_path, index_tmp_path]:
                if os.path.exists(p):
                    os.remove(p)
            return

        # 2. 加载数据集（这部分通常很快，不需要 checkpoint）
        local_dataset_path = "./wiki_dpr_text_only"
        ds = load_from_disk(local_dataset_path)
        self.corpus = []
        self.corpus_ids = []
        for i, row in enumerate(tqdm(ds, desc="Loading corpus")):
            self.corpus.append({
                "id": row.get("id", str(i)),
                "title": row.get("title", ""),
                "text": row.get("text", ""),
            })
            self.corpus_ids.append(str(i))

        # 3. 构建索引（支持断点续建）
        self._build_faiss_index(batch_size=5120, checkpoint_every=10)

        # 4. 保存最终文件
        with open(corpus_path, "wb") as f:
            pickle.dump(self.corpus, f)
        with open(ids_path, "wb") as f:
            pickle.dump(self.corpus_ids, f)
        faiss.write_index(self.index, index_path)

        # 5. 清理临时文件
        for p in [progress_path, index_tmp_path]:
            if os.path.exists(p):
                os.remove(p)

    def _build_faiss_index(self, batch_size: int = 5120, checkpoint_every: int = 10):
        progress_path = os.path.join(self.cache_dir, "build_progress.json")
        index_tmp_path = os.path.join(self.cache_dir, "faiss_bge.index.tmp")
        
        texts = [f"{item['title']} {item['text']}" for item in self.corpus]
        start_idx = 0
        
        # 尝试恢复之前的进度
        if os.path.exists(progress_path) and os.path.exists(index_tmp_path):
            try:
                with open(progress_path, "r") as f:
                    progress = json.load(f)
                start_idx = progress.get("last_processed_idx", 0)
                if 0 < start_idx < len(texts):
                    self.index = faiss.read_index(index_tmp_path)
                    print(f"[Checkpoint] 从第 {start_idx} / {len(texts)} 条恢复构建...")
                else:
                    start_idx = 0
                    self.index = faiss.IndexFlatIP(self.emb_dim)
            except Exception as e:
                print(f"[Checkpoint] 恢复失败 ({e})，从头开始")
                start_idx = 0
                self.index = faiss.IndexFlatIP(self.emb_dim)
        else:
            self.index = faiss.IndexFlatIP(self.emb_dim)

        total_batches = (len(texts) + batch_size - 1) // batch_size
        start_batch = start_idx // batch_size
        
        pbar = tqdm(
            range(start_idx, len(texts), batch_size),
            desc="Encoding",
            total=total_batches,
            initial=start_batch,
        )
        
        batches_since_ckpt = 0
        
        for i in pbar:
            batch = texts[i:i+batch_size]
            embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.index.add(embeddings.astype("float32"))
            batches_since_ckpt += 1
            
            # 每 N 个 batch 保存一次 checkpoint
            if batches_since_ckpt >= checkpoint_every:
                faiss.write_index(self.index, index_tmp_path)
                with open(progress_path, "w") as f:
                    json.dump({"last_processed_idx": i + len(batch)}, f)
                os.sync()  # 强制刷盘，防止数据丢失
                batches_since_ckpt = 0
                pbar.set_postfix({"saved": f"{i+len(batch)}/{len(texts)}"})
    
    def retrieve(
        self,
        queries: List[str],
        top_k: int = 200,
        batch_size: int = 32,
    ):
        instruction = ""
        all_results = []
        for i in tqdm(range(0, len(queries), batch_size), desc="Retrieving"):
            batch = [instruction + query for query in queries[i:i+batch_size]]
            embeddings = self.model.encode(batch, convert_to_numpy=True, normalize_embeddings=True,show_progress_bar=False,)
            scores, indices = self.index.search(embeddings.astype("float32"), top_k)
            for q_idx, (q_scores, q_indices) in enumerate(zip(scores, indices)):
                results = []
                for rank, (s, i) in enumerate(zip(q_scores, q_indices), start=1):
                    item = self.corpus[i]
                    results.append({
                        "id": item["id"],
                        "title": item["title"],
                        "text": item["text"],
                        "score": float(s),
                        "rank": rank,
                    })
                all_results.append(results) 
        return all_results
    

