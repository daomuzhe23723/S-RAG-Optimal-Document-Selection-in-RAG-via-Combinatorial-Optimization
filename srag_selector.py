#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S-RAG: Optimal Document Selection in RAG via Combinatorial Optimization
修复版本：
  1. 默认使用 nltk 名词/动词 lemma 提取（与论文 Appendix B.2 一致）
  2. 密度贪心中使用增量 covered 集合（避免每次从头重算，提升速度）
  3. select() 返回 (indices, truncated_docs, costs) 供 generate.py 截断使用
"""

import itertools
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple


class ConceptExtractor:
    """
    从文本中提取概念。
    论文 Appendix B.2：lowercase → tokenize → lemmatize → 去停用词 → 取名词/动词 lemma。
    
    method 优先级：nltk > noun_chunks(spaCy) > simple(正则回退)
    """

    STOPWORDS: Set[str] = {
        "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can", "need",
        "that", "this", "these", "those", "it", "its", "they", "them", "their",
        "we", "us", "our", "you", "your", "he", "she", "his", "her", "i", "my",
        "me", "what", "which", "who", "whom", "whose", "when", "where", "why",
        "how", "not", "no", "nor", "so", "yet", "both", "either", "neither",
        "each", "every", "all", "any", "few", "more", "most", "other", "some",
        "such", "than", "then", "just", "over", "also", "only", "very", "too",
        "into", "about", "after", "before", "between", "through", "during",
        "while", "although", "because", "since", "until", "unless", "whether",
        "as", "up", "out", "there", "here", "now", "one", "two", "first",
        "new", "old", "many", "much", "same", "even", "back", "still",
    }

    def __init__(self, method: str = "nltk"):
        self.method = method
        self.lemmatizer = None
        self.nlp = None

        if method == "nltk":
            try:
                import nltk
                from nltk.stem import WordNetLemmatizer
                for pkg, path in [
                    ('punkt_tab',        'tokenizers/punkt_tab'),
                    ('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng'),
                    ('wordnet',          'corpora/wordnet'),
                    ('stopwords',        'corpora/stopwords'),
                ]:
                    try:
                        nltk.data.find(path)
                    except LookupError:
                        nltk.download(pkg, quiet=True)
                self.lemmatizer = WordNetLemmatizer()
            except Exception as e:
                print(f"[Warning] NLTK 不可用（{e}），退回 simple 模式。")
                self.method = "simple"

        elif method == "noun_chunks":
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm",
                                      disable=["parser", "ner"])
            except Exception as e:
                print(f"[Warning] spaCy 不可用（{e}），退回 simple 模式。")
                self.method = "simple"

    def extract(self, text: str) -> Dict[str, float]:
        if self.method == "nltk" and self.lemmatizer is not None:
            result = self._extract_nltk(text)
        elif self.method == "noun_chunks" and self.nlp is not None:
            result = self._extract_spacy(text)
        else:
            result = self._extract_simple(text)
        return result if result else self._extract_simple(text)

    def _extract_nltk(self, text: str) -> Dict[str, float]:
        import nltk
        tokens = nltk.word_tokenize(text.lower())
        tagged = nltk.pos_tag(tokens)

        concepts = []
        for word, tag in tagged:
            word_clean = re.sub(r"[^a-z0-9]", "", word)
            if len(word_clean) <= 2 or word_clean in self.STOPWORDS:
                continue
            if tag.startswith("NN"):
                lemma = self.lemmatizer.lemmatize(word_clean, pos="n")
                concepts.append(lemma)
            elif tag.startswith("VB"):
                lemma = self.lemmatizer.lemmatize(word_clean, pos="v")
                if lemma not in {"be", "have", "do", "say", "get", "make",
                                  "go", "know", "take", "see", "come", "use"}:
                    concepts.append(lemma)

        if not concepts:
            return {}
        counts = Counter(concepts)
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    def _extract_spacy(self, text: str) -> Dict[str, float]:
        doc = self.nlp(text)
        concepts = []
        for token in doc:
            if (token.pos_ in ("NOUN", "VERB", "PROPN")
                    and not token.is_stop
                    and len(token.lemma_) > 2
                    and token.lemma_ not in self.STOPWORDS):
                concepts.append(token.lemma_.lower())
        if not concepts:
            return {}
        counts = Counter(concepts)
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    def _extract_simple(self, text: str) -> Dict[str, float]:
        words = re.findall(r"\b[a-z]{4,}\b", text.lower())
        words = [w for w in words if w not in self.STOPWORDS]
        if not words:
            return {}
        counts = Counter(words)
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}


class WeightedCoverageFunction:
    def __init__(self,
                 concept_weights: Dict[str, float],
                 doc_concepts: List[Set[str]]):
        self.concept_weights = concept_weights
        self.doc_concepts = doc_concepts

    def evaluate(self, S: Set[int]) -> float:
        if not S:
            return 0.0
        covered = set()
        for i in S:
            covered.update(self.doc_concepts[i])
        return float(sum(self.concept_weights.get(u, 0.0) for u in covered))

    def marginal_gain(self, covered: Set[str], i: int) -> float:
        new = self.doc_concepts[i] - covered
        return float(sum(self.concept_weights.get(u, 0.0) for u in new))


class SRAGSelector:
    def __init__(self,
                 budget: int,
                 concept_extractor: Optional[ConceptExtractor] = None,
                 concept_depth: int = 20,
                 fast_mode: bool = False):
        self.budget = budget
        self.concept_extractor = (concept_extractor
                                  or ConceptExtractor(method="nltk"))
        self.concept_depth = concept_depth
        self.fast_mode = fast_mode

    def _build_concept_universe_and_weights(
        self,
        docs: List[str],
        retriever_scores: List[float],
    ) -> Tuple[Dict[str, float], List[Set[str]]]:
        n = len(docs)
        doc_concepts: List[Set[str]] = [
            set(self.concept_extractor.extract(doc).keys())
            for doc in docs
        ]
        L = min(n, self.concept_depth)
        concept_universe: Set[str] = set()
        for i in range(L):
            concept_universe.update(doc_concepts[i])

        concept_weights: Dict[str, float] = {}
        for c in concept_universe:
            w = 0.0
            for i in range(n):
                if c in doc_concepts[i]:
                    w = max(w, retriever_scores[i])
            concept_weights[c] = w

        return concept_weights, doc_concepts

    def _best_small_solution(
        self,
        candidates: List[int],
        costs: List[int],
        objective: WeightedCoverageFunction,
    ) -> Set[int]:
        best_S: Set[int] = set()
        best_val = 0.0

        for i in candidates:
            if costs[i] <= self.budget:
                val = objective.evaluate({i})
                if val > best_val:
                    best_val, best_S = val, {i}

        for i, j in itertools.combinations(candidates, 2):
            if costs[i] + costs[j] <= self.budget:
                val = objective.evaluate({i, j})
                if val > best_val:
                    best_val, best_S = val, {i, j}

        return best_S

    def _density_greedy_completion(
        self,
        seed: Set[int],
        candidates: List[int],
        costs: List[int],
        objective: WeightedCoverageFunction,
    ) -> Set[int]:
        S = set(seed)
        remaining = self.budget - sum(costs[i] for i in S)

        covered: Set[str] = set()
        for i in S:
            covered.update(objective.doc_concepts[i])

        pool = [i for i in candidates if i not in S]

        while True:
            best_idx = None
            best_ratio = -1.0

            for i in pool:
                if costs[i] > remaining:
                    continue
                gain = objective.marginal_gain(covered, i)
                if gain <= 1e-12:
                    continue
                ratio = gain / costs[i]
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i

            if best_idx is None:
                break

            S.add(best_idx)
            covered.update(objective.doc_concepts[best_idx])
            pool.remove(best_idx)
            remaining -= costs[best_idx]

        return S

    def select(
        self,
        query: str,
        docs: List[str],
        costs: List[int],
        retriever_scores: Optional[List[float]] = None,
    ) -> List[int]:
        n = len(docs)
        assert len(costs) == n

        if retriever_scores is None:
            retriever_scores = [1.0 / (i + 1) for i in range(n)]
        assert len(retriever_scores) == n

        concept_weights, doc_concepts = self._build_concept_universe_and_weights(
            docs, retriever_scores
        )
        objective = WeightedCoverageFunction(concept_weights, doc_concepts)

        if not concept_weights:
            print("[Warning] 未提取到概念，退化为 top-k 选择。")
            selected, used = [], 0
            for idx in sorted(range(n),
                               key=lambda i: retriever_scores[i],
                               reverse=True):
                if used + costs[idx] <= self.budget:
                    selected.append(idx)
                    used += costs[idx]
            return sorted(selected)

        candidates = list(range(n))
        S_best = self._best_small_solution(candidates, costs, objective)

        if not self.fast_mode:
            for seed_tuple in itertools.combinations(candidates, 3):
                if sum(costs[i] for i in seed_tuple) > self.budget:
                    continue
                S_completed = self._density_greedy_completion(
                    set(seed_tuple), candidates, costs, objective
                )
                if objective.evaluate(S_completed) > objective.evaluate(S_best):
                    S_best = S_completed
        else:
            S_fast = self._density_greedy_completion(
                set(), candidates, costs, objective
            )
            if objective.evaluate(S_fast) > objective.evaluate(S_best):
                S_best = S_fast

        return sorted(list(S_best))

    def get_selected_documents(
        self,
        query: str,
        docs: List[str],
        costs: List[int],
        retriever_scores: Optional[List[float]] = None,
    ) -> Tuple[List[int], List[str], int]:
        indices = self.select(query, docs, costs, retriever_scores)
        selected_docs = [docs[i] for i in indices]
        total_cost = sum(costs[i] for i in indices)
        return indices, selected_docs, total_cost


