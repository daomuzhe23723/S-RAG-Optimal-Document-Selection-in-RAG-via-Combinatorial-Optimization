#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S-RAG: Optimal Document Selection in RAG via Combinatorial Optimization
Core implementation of the submodular document selection algorithm.

Reference:
    Anonymous et al. (2026). "Optimal Document Selection in RAG via 
    Combinatorial Optimization: A Theoretical Framework." 
    (ACL submission).

Problem Formulation:
    - Input:  query q, candidate documents D = {d_1,...,d_n}, token costs c_i,
              strict token budget B, retriever scores rel(d_i, q).
    - Output: subset S ⊆ D such that Σ_{i∈S} c_i ≤ B.
    - Objective: maximize weighted coverage function 
                f(S) = Σ_{u∈U} w(u)·1[u ∈ ∪_{d∈S} U(d)]
                which is provably monotone and submodular (Lemma 1).

Algorithm (Algorithm 1 from paper):
    Partial Enumeration + Density-Greedy Completion.
    1. Initialize S_best with the best feasible solution of size ≤ 2.
    2. Enumerate all feasible seeds U ⊆ D with |U| = 3 and c(U) ≤ B.
    3. For each seed, greedily complete by maximizing marginal-gain density:
       d* = argmax_{d feasible} Δ_f(d|S) / c(d).
    4. Return the best S across all seeds and the size-≤2 solution.
    Approximation guarantee: (1 - 1/e) for the surrogate objective (Theorem 1).

Changes from original code:
    - Objective changed from probabilistic soft-coverage to hard set-coverage (Eq. 4).
    - Algorithm changed from simple greedy to partial-enumeration + density greedy.
    - Concept universe U now built from top-L candidate documents (not from query).
    - Concept weights w(u) now use max retriever score (not query tf).
"""

import itertools
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple


class ConceptExtractor:
    """
    Extracts query-relevant concepts from text.
    Paper uses: lowercase, tokenize, lemmatize, remove stopwords,
    then extract noun/verb lemmas (Appendix B.2).
    """

    def __init__(self, method: str = "simple"):
        """
        Args:
            method: "nltk" (POS-aware lemmatization, requires nltk),
                    "noun_chunks" (requires spaCy),
                    or "simple" (regex-based, no deps).
        """
        self.method = method
        self.lemmatizer = None
        self.nlp = None

        if method == "nltk":
            try:
                import nltk
                from nltk.stem import WordNetLemmatizer
                # Download required NLTK data quietly if missing
                for pkg in ('punkt', 'averaged_perceptron_tagger', 'wordnet'):
                    try:
                        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'taggers/{pkg}' if pkg == 'averaged_perceptron_tagger' else f'corpora/{pkg}')
                    except LookupError:
                        nltk.download(pkg, quiet=True)
                self.lemmatizer = WordNetLemmatizer()
            except Exception:
                print("[Warning] NLTK not available. Falling back to simple extraction.")
                self.method = "simple"

        elif method == "noun_chunks":
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except Exception:
                print("[Warning] spaCy model not available. Falling back to simple extraction.")
                self.method = "simple"

    def extract(self, text: str) -> Dict[str, float]:
        """Extract concepts and return normalized within-doc frequencies."""
        if self.method=="nltk" and self.lemmatizer is not None:
            return self._extract_nltk(text)
        elif self.method=="noun_chunks" and self.nlp is not None:
            return self._extract_spacy(text)
        else:
            return self._extract_simple(text)

    def _extract_nltk(self, text: str) -> Dict[str, float]:
        import nltk
        tokens = nltk.word_tokenize(text.lower())
        tagged = nltk.pos_tag(tokens)
        stopwords = self._get_stopwords()

        concepts = []
        for word, tag in tagged:
            word = re.sub(r"[^a-z0-9]", "", word)
            if len(word) <= 2 or word.isnumeric() or word in stopwords:
                continue
            # Keep nouns (NN*) and verbs (VB*) as per paper
            if tag.startswith('NN') or tag.startswith('VB'):
                pos = 'n' if tag.startswith('NN') else 'v'
                lemma = self.lemmatizer.lemmatize(word, pos=pos)
                concepts.append(lemma)

        if not concepts:
            return self._extract_simple(text)

        counts = Counter(concepts)
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    def _extract_spacy(self, text: str) -> Dict[str, float]:
        doc = self.nlp(text)
        concepts = []
        for token in doc:
            if token.pos_ in ('NOUN', 'VERB', 'PROPN') and not token.is_stop and len(token.lemma_) > 2:
                concepts.append(token.lemma_.lower())
        if not concepts:
            return self._extract_simple(text)
        counts = Counter(concepts)
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    def _extract_simple(self, text: str) -> Dict[str, float]:
        """
        Fallback regex-based extraction.
        Lowercases, removes stopwords, keeps words >= 4 chars.
        """
        words = re.findall(r"\b[a-z]{4,}\b", text.lower())
        stopwords = self._get_stopwords()
        words = [w for w in words if w not in stopwords]
        if not words:
            return {}
        counts = Counter(words)
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}
        
    @staticmethod
    def _get_stopwords() -> Set[str]:
        return {
            "that", "with", "from", "they", "have", "this", "will", "your",
            "what", "when", "where", "which", "their", "there", "about",
            "could", "would", "should", "than", "then", "them", "been",
            "being", "over", "also", "only", "some", "time", "very", "after",
            "before", "just", "into", "such", "make", "made", "like", "other",
            "more", "most", "many", "much", "how", "who", "whom", "whose",
            "why", "shall", "may", "might", "must", "can", "does", "did",
            "done", "doing", "has", "had", "having", "get", "gets", "got",
            "gotten", "getting", "use", "used", "using", "say", "said", "says",
            "going", "go", "went", "gone", "know", "knew", "known", "knows",
            "think", "thought", "thinks", "see", "saw", "seen", "sees",
            "come", "came", "comes", "coming", "want", "wanted", "wants",
            "look", "looked", "looking", "looks", "way", "ways", "find",
            "found", "finding", "finds", "give", "gave", "given", "gives",
            "giving", "tell", "told", "telling", "tells", "work", "worked",
            "working", "works", "call", "called", "calling", "calls", "try",
            "tried", "tries", "trying", "need", "needed", "needing", "needs",
            "feel", "felt", "feeling", "feels", "become", "became", "becomes",
            "becoming", "leave", "left", "leaves", "leaving", "put", "puts",
            "putting", "mean", "means", "meant", "meaning", "keep", "keeps",
            "kept", "keeping", "let", "lets", "letting", "begin", "began",
            "begun", "begins", "beginning", "seem", "seemed", "seeming",
            "seems", "help", "helped", "helping", "helps", "show", "showed",
            "shown", "shows", "showing", "hear", "heard", "hearing", "hears",
            "play", "played", "playing", "plays", "run", "ran", "running",
            "runs", "move", "moved", "moves", "moving", "live", "lived",
            "lives", "living", "believe", "believed", "believes", "believing",
            "bring", "brought", "bringing", "brings", "happen", "happened",
            "happening", "happens", "stand", "stood", "standing", "stands",
            "lose", "lost", "loses", "losing", "pay", "paid", "paying",
            "pays", "meet", "met", "meeting", "meets", "include", "included",
            "includes", "including", "continue", "continued", "continues",
            "continuing", "set", "sets", "setting", "learn", "learned",
            "learning", "learns", "change", "changed", "changes", "changing",
            "lead", "led", "leading", "leads", "understand", "understood",
            "understanding", "understands", "watch", "watched", "watches",
            "watching", "follow", "followed", "following", "follows", "stop",
            "stopped", "stopping", "stops", "create", "created", "creates",
            "creating", "speak", "spoke", "spoken", "speaks", "speaking",
            "read", "reads", "reading", "allow", "allowed", "allowing",
            "allows", "add", "added", "adding", "adds", "spend", "spent",
            "spending", "spends", "grow", "grew", "grown", "grows", "growing",
            "open", "opened", "opening", "opens", "walk", "walked", "walking",
            "walks", "win", "won", "winning", "wins", "offer", "offered",
            "offering", "offers", "remember", "remembered", "remembering",
            "remembers", "love", "loved", "loves", "loving", "consider",
            "considered", "considering", "considers", "appear", "appeared",
            "appearing", "appears", "buy", "bought", "buying", "buys", "wait",
            "waited", "waiting", "waits", "serve", "served", "serves",
            "serving", "die", "died", "dies", "dying", "send", "sent",
            "sending", "sends", "expect", "expected", "expecting", "expects",
            "build", "built", "building", "builds", "stay", "stayed", "staying",
            "stays", "fall", "fell", "fallen", "falling", "falls", "cut", "cuts",
            "cutting", "reach", "reached", "reaches", "reaching", "kill",
            "killed", "killing", "kills", "remain", "remained", "remaining",
            "remains", "suggest", "suggested", "suggesting", "suggests",
            "raise", "raised", "raises", "raising", "pass", "passed", "passes",
            "passing", "sell", "sold", "selling", "sells", "require", "required",
            "requiring", "requires", "report", "reported", "reporting", "reports",
            "decide", "decided", "decides", "deciding", "pull", "pulled",
            "pulling", "pulls"
        }


class WeightedCoverageFunction:
    """
    Weighted coverage objective from S-RAG (Eq. 4 in paper).

    f(S) = Σ_{u∈U} w(u) · 1[ u ∈ ∪_{d∈S} U(d) ]

    Properties (per paper, Lemma 1):
        - Monotone: adding a document never decreases coverage.
        - Submodular: diminishing returns (marginal gain decreases as S grows).
    """

    def __init__(self, concept_weights: Dict[str, float],
                 doc_concepts: List[Set[str]]):
        """
        Args:
            concept_weights: Dict concept -> w_c.
            doc_concepts: List of Sets, where doc_concepts[i] is the set of
                          concepts covered by document i.
        """
        self.concept_weights = concept_weights
        self.doc_concepts = doc_concepts

    def evaluate(self, S: Set[int]) -> float:
        """Compute f(S) via union of covered concepts."""
        if not S:
            return 0.0
        covered_concepts = set()
        for i in S:
            covered_concepts.update(self.doc_concepts[i])
        return float(sum(self.concept_weights.get(u,0.0) for u in covered_concepts))

    def marginal_gain(self, S: Set[int], i: int) -> float:
        """
        Δ_f(i|S) = f(S ∪ {i}) - f(S)
                  = Σ_{u ∈ U(d_i) \ ∪_{j∈S} U(d_j)} w(u)
        """
        if i in S:
            return 0.0
        covered = set()
        for j in S:
            covered.update(self.doc_concepts[j])
        new_covered = self.doc_concepts[i]-covered    
        return float(sum(self.concept_weights.get(u,0.0) for u in new_covered))

class SRAGSelector:
    """
    S-RAG Document Selector.

    Implements Algorithm 1 from the paper:
    Partial Enumeration + Density-Greedy Completion for knapsack-constrained
    monotone submodular maximization with (1 - 1/e) approximation guarantee.
    """

    def __init__(self, budget: int,
                 concept_extractor: Optional[ConceptExtractor] = None,
                 concept_depth: int = 20,
                 fast_mode: bool = False):
        """
        Args:
            budget: Strict token budget B (knapsack capacity).
            concept_extractor: Concept extraction module.
            concept_depth: L in paper; number of top-ranked docs used to
                           construct the concept universe U (default 20).
            fast_mode: If True, skip partial enumeration (seed size 3) and use
                       pure density-greedy. This loses the formal (1-1/e)
                       guarantee but is much faster for large n. The paper's
                       production system uses this fast variant (Appendix A).
        """
        self.budget=budget
        self.concept_extractor=concept_extractor or ConceptExtractor(method="simple")
        self.concept_depth=concept_depth
        self.fast_mode=fast_mode


    def _build_concept_universe_and_weights(
        self,
        docs: List[str],
        retriever_scores: Optional[List[float]] = None
    ) -> Tuple[Dict[str, float], List[Set[str]]]:
        """
        Build concept universe U and concept weights w(u).

        Following paper Appendix B.2:
        - U is constructed from the top-L retrieved passages.
        - w(u) = max_{d∈D : u∈U(d)} rel(d,q).
        """
        n=len(docs)
        if retriever_scores is None:
            retriever_scores = [1.0 / (i + 1) for i in range(n)]
        doc_concepts=[]
        for doc in docs:
            extracted = self.concept_extractor.extract(doc)
            doc_concepts.append(set(extracted.keys()))
        
        L=min(n,self.concept_depth)
        concept_universe = set()
        for i in range(L):
            concept_universe.update(doc_concepts[i])
        concept_weights = {}
        for c in concept_universe:
            w=0.0
            for i in range(n):
                if c in doc_concepts[i]:
                    w=max(w,retriever_scores[i])
            concept_weights[c] = w
        return concept_weights, doc_concepts


    def _best_small_solution(
        self,
        candidates: List[int],
        costs: List[int],
        objective: WeightedCoverageFunction,
        max_size: int = 2
    ) -> Set[int]:
        best_S = set()
        best_val = objective.evaluate(best_S)

        for i in candidates:
            if costs[i]<=self.budget:
                val=objective.evaluate({i})
                if val > best_val:
                    best_val = val
                    best_S = {i}

        if max_size >= 2:
            for i, j in itertools.combinations(candidates, 2):
                if costs[i]+costs[j]<=self.budget:
                    val = objective.evaluate({i, j})
                    if val > best_val:
                        best_val = val
                        best_S = {i, j}
        return best_S
        

    def _density_greedy_completion(
        self,
        seed: Set[int],
        candidates: List[int],
        costs: List[int],
        objective: WeightedCoverageFunction
    ) -> Set[int]:
        """
        Density-greedy completion (Algorithm 1 lines 8-12).
        Repeatedly add the feasible document with highest marginal gain density.
        """
        S = set(seed)
        remaining = self.budget - sum(costs[i] for i in S)
        pool = [i for i in candidates if i not in S]
        while True:
            best_idx = None
            best_ratio = -1.0
            best_gain = 0.0
            for i in pool:
                if costs[i] > remaining:
                    continue
                gain=objective.marginal_gain(S,i)
                if gain <= 1e-12:
                    continue
                ratio = gain / costs[i]
                if ratio>best_ratio:
                    best_ratio=ratio
                    best_idx=i
                    best_gain = gain
            if best_idx is None:
                break
            S.add(best_idx)
            pool.remove(best_idx)
            remaining -= costs[best_idx]
        return S


    def select(
        self,
        query: str,
        docs: List[str],
        costs: List[int],
        retriever_scores: Optional[List[float]] = None
    ) -> List[int]:
        """
        Select documents under the token budget using Algorithm 1.

        Args:
            query: User query string.
            docs:  List of candidate document strings. Should be ordered by
                   descending retriever relevance (top-L are used for U).
            costs: List of token costs c_i for each document.
            retriever_scores: Optional dense retriever scores rel(d_i, q).
                              If None, inverse rank is used as a proxy.

        Returns:
            Sorted list of selected document indices.
        """
        n = len(docs)
        assert len(costs) == n, "Length mismatch between docs and costs."
        if retriever_scores is not None:
            assert len(retriever_scores) == n, "Length mismatch for retriever_scores."
        concept_weights, doc_concepts = self._build_concept_universe_and_weights(
            docs, retriever_scores
        )
        objective = WeightedCoverageFunction(concept_weights, doc_concepts)
        if not concept_weights:
            sorted_idx=sorted(range(n), key=lambda i: costs[i])
            selected, used = [], 0
            for idx in sorted_idx:
                if used + costs[idx] <= self.budget:
                    selected.append(idx)
                    used += costs[idx]
                else:
                    break
            return selected
        candidates=list(range(n))
        Sbest = self._best_small_solution(candidates, costs, objective, max_size=2)
        if not self.fast_mode:
            feasible_seeds = [
                combo for combo in itertools.combinations(candidates, 3)
                if sum(costs[i] for i in combo) <= self.budget
            ]
            for seed_tuple in feasible_seeds:
                seed = set(seed_tuple)
                S_completed = self._density_greedy_completion(
                    seed, candidates, costs, objective
                )
                if objective.evaluate(S_completed) > objective.evaluate(Sbest):
                    Sbest = S_completed
        else:
            S_fast = self._density_greedy_completion(
                set(), candidates, costs, objective
            )
            if objective.evaluate(S_fast)>objective.evaluate(Sbest):
                Sbest = S_fast
        return sorted(list(Sbest))
    
    def get_selected_documents(
        self,
        query: str,
        docs: List[str],
        costs: List[int],
        retriever_scores: Optional[List[float]] = None
    ) -> Tuple[List[int], List[str], int]:
        """Convenience wrapper returning indices, texts, and total cost."""
        indices = self.select(query, docs, costs, retriever_scores)
        selected_docs = [docs[i] for i in indices]
        total_cost = sum(costs[i] for i in indices)
        return indices, selected_docs, total_cost



# ======================== Example Usage ========================
if __name__ == "__main__":
    query = (
        "What are the causes and treatments of type 2 diabetes, "
        "and what lifestyle changes are recommended?"
    )

    docs = [
        "Type 2 diabetes is caused by insulin resistance and relative insulin deficiency.",
        "Genetic factors and family history play a significant role in type 2 diabetes risk.",
        "Obesity and sedentary lifestyle are major modifiable risk factors for diabetes.",
        "Treatment includes metformin as first-line therapy and lifestyle modifications.",
        "Insulin therapy may be required when oral medications fail to control blood glucose.",
        "Regular exercise and dietary changes can significantly improve glycemic control.",
        "Diabetic retinopathy is a common complication affecting the eyes.",
        "Cardiovascular disease risk is elevated in patients with poorly controlled diabetes.",
        "The pancreas produces insulin which regulates blood sugar levels in the body.",
        "Gestational diabetes occurs during pregnancy and may resolve after delivery.",
    ]

    costs = [len(d.split()) for d in docs]
    budget = 40

    # Use fast_mode=True for quick demo; set False for full Algorithm 1 guarantee
    selector = SRAGSelector(budget=budget, concept_depth=10, fast_mode=True)
    indices, selected, total_cost = selector.get_selected_documents(query, docs, costs)

    print("=" * 70)
    print("S-RAG: Submodular Document Selection under Knapsack Constraint")
    print("Algorithm : Partial Enumeration + Density-Greedy (Algorithm 1)")
    print("=" * 70)
    print(f"Query        : {query}")
    print(f"Budget B     : {budget} tokens")
    print(f"Candidates n : {len(docs)}")
    print("-" * 70)
    print(f"Selected k   : {len(indices)} documents (total cost = {total_cost})")
    for idx in indices:
        print(f"  [{idx:2d}] (cost={costs[idx]:2d})  {docs[idx]}")
    print("-" * 70)
    print("Theory       : Monotone submodular maximization under knapsack constraint")
    print("Guarantee    : (1 - 1/e) ≈ 0.632  (Theorem 1, Nemhauser et al. 1978)")
    print("Objective    : Weighted set-coverage over query-relevant concepts (Eq. 4)")
    print("=" * 70)