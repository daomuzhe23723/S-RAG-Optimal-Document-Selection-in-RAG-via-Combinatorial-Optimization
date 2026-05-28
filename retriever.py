import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

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

        if os.path.exists(corpus_path) and os.path.exists(index_path):
            with open(corpus_path, "rb") as f:
                self.corpus = pickle.load(f)
            with open(ids_path, "rb") as f:
                self.corpus_ids = pickle.load(f)
            self.index = faiss.read_index(index_path)
            return
        
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

        self._build_faiss_index()

        with open(corpus_path, "wb") as f:
            pickle.dump(self.corpus, f)
        with open(ids_path, "wb") as f:
            pickle.dump(self.corpus_ids, f)
        faiss.write_index(self.index, index_path)

    def _build_faiss_index(self, batch_size: int = 2560):
        self.index = faiss.IndexFlatIP(self.emb_dim)
        texts = [f"{item['title']} {item['text']}" for item in self.corpus]
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch=texts[i:i+batch_size]
            embeddings = self.model.encode(batch, convert_to_numpy=True, normalize_embeddings=True,show_progress_bar=False,)
            self.index.add(embeddings.astype("float32"))
    
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
    

