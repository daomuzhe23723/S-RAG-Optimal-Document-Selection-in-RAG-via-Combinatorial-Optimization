#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S-RAG 生成与评测主脚本
修复点：
  1. load_hotpotqa 改为读 jsonl（与 NQ/ELI5 格式统一）
  2. Prompt 模板与论文 Appendix B.1 一致
  3. passage 截断逻辑（论文 §4.1）
  4. topk / mmr baseline 实现
"""

import json
import os
import re
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import trange

from retriever import BGERetriever
from srag_selector import SRAGSelector
from utils import evaluate
import argparse

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────
def load_nq(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            answers = [o["answer"] for o in raw.get("output", []) if o.get("answer")]
            data.append({
                "question": raw["input"],
                "answers":  answers,
            })
    return data


def load_eli5(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            answers = [o["answer"] for o in raw.get("output", []) if o.get("answer")]
            data.append({
                "question": raw.get("input", ""),
                "answers":  answers,
            })
    return data


def load_hotpotqa(path: str):
    # ── 修复：改为读 jsonl，与 NQ/ELI5 格式统一 ──
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            # KILT 格式：output 字段包含 answer
            if raw.get("output"):
                answers = [o["answer"] for o in raw["output"] if o.get("answer")]
            else:
                answers = []
            data.append({
                "question": raw.get("input", ""),
                "answers":  answers,
            })
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 构建 srag_data
# ─────────────────────────────────────────────────────────────────────────────
def build_srag_inputs(inputs, retrieved_results, tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    srag_inputs = []
    for item, results in zip(inputs, retrieved_results):
        docs  = [d["text"]  for d in results]
        costs = [len(tokenizer.encode(d, add_special_tokens=False)) for d in docs]
        srag_inputs.append({
            "question":         item["question"],
            "answers":          item["answers"],
            "docs":             docs,
            "retriever_scores": [d["score"] for d in results],
            "costs":            costs,
        })
    return srag_inputs


# ─────────────────────────────────────────────────────────────────────────────
# Passage 截断（论文 §4.1）
# ─────────────────────────────────────────────────────────────────────────────
def truncate_passages_to_budget(
    passages: List[str],
    costs: List[int],
    budget: int,
    tokenizer,
) -> List[str]:
    result = []
    remaining = budget
    for text, cost in zip(passages, costs):
        if cost <= remaining:
            result.append(text)
            remaining -= cost
        else:
            if remaining > 0:
                token_ids = tokenizer.encode(
                    text, add_special_tokens=False
                )[:remaining]
                truncated = tokenizer.decode(token_ids, skip_special_tokens=True)
                result.append(truncated)
            break
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Prompt 模板（论文 Appendix B.1）
# ─────────────────────────────────────────────────────────────────────────────
INSTRUCTIONS = (
    "Instructions: Answer the question using the passages. "
    "Be concise and factual. If multiple passages support the answer, "
    "synthesize them. Do not fabricate unsupported claims. "
    "Output ONLY the answer phrase, NOT a full sentence."
)


def build_prompt(question: str, passages: List[str], tokenizer) -> str:
    content = f"Question: {question.strip()}\n\nPassages:\n"
    for idx, p in enumerate(passages, start=1):
        content += f"{idx}. {p}\n"
    content += f"\n{INSTRUCTIONS}"
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Top-k baseline
# ─────────────────────────────────────────────────────────────────────────────
def select_topk(docs, costs, scores, budget):
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    selected_indices, used = [], 0
    for i in order:
        if used + costs[i] <= budget:
            selected_indices.append(i)
            used += costs[i]
        elif used < budget:
            selected_indices.append(i)
            break
    return selected_indices


# ─────────────────────────────────────────────────────────────────────────────
# MMR baseline
# ─────────────────────────────────────────────────────────────────────────────
def _bow_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(re.findall(r"\b[a-z]+\b", text_a.lower()))
    tokens_b = set(re.findall(r"\b[a-z]+\b", text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def select_mmr(docs, costs, scores, budget, lambda_mmr=0.6):
    n = len(docs)
    selected_indices = []
    used = 0
    remaining_pool = list(range(n))
    max_score = max(scores) if scores else 1.0
    norm_scores = [s / (max_score + 1e-12) for s in scores]

    while remaining_pool:
        best_idx, best_val = None, -float("inf")
        for i in remaining_pool:
            relevance = norm_scores[i]
            if selected_indices:
                redundancy = max(
                    _bow_similarity(docs[i], docs[j]) for j in selected_indices
                )
            else:
                redundancy = 0.0
            mmr_val = lambda_mmr * relevance - (1 - lambda_mmr) * redundancy
            if mmr_val > best_val:
                best_val, best_idx = mmr_val, i

        if best_idx is None:
            break

        if used + costs[best_idx] <= budget:
            selected_indices.append(best_idx)
            used += costs[best_idx]
        elif used < budget:
            selected_indices.append(best_idx)
            break
        else:
            break

        remaining_pool.remove(best_idx)

    return selected_indices

# ─────────────────────────────────────────────────────────────────────────────
# Greedy (Rel/Cost) baseline
# ─────────────────────────────────────────────────────────────────────────────
def select_greedy_rel_cost(docs, costs, scores, budget):
    n = len(docs)
    candidates = [(i, scores[i] / costs[i]) for i in range(n)]
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    selected_indices = []
    used = 0
    for idx, density in candidates:
        if used + costs[idx] <= budget:
            selected_indices.append(idx)
            used += costs[idx]
        elif used < budget:
            selected_indices.append(idx)
            break
    
    return selected_indices


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    if args.dataset == "nq":
        data = load_nq(args.dataset_path)
        max_new_tokens = 128
    elif args.dataset == "eli5":
        data = load_eli5(args.dataset_path)
        max_new_tokens = 512
    elif args.dataset == "hotpotqa":
        data = load_hotpotqa(args.dataset_path)
        max_new_tokens = 192
    else:
        raise ValueError(f"未知数据集：{args.dataset}")

    retriever = BGERetriever(
        model_name=args.retriever_name,
        device="cuda",
        corpus_dir=args.corpus_dir,
    )
    questions = [d["question"] for d in data]
    retrieved_results = retriever.retrieve(questions, top_k=args.k)

    srag_data = build_srag_inputs(data, retrieved_results, args.tokenizer_name)
    del retriever
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    MMR_LAMBDA = {"nq": 0.6, "eli5": 0.7, "hotpotqa": 0.5}

    prompts = []

    if args.method == "srag":
        from srag_selector import ConceptExtractor
        selector = SRAGSelector(
            budget=args.budget,
            concept_depth=args.concept_depth,
            fast_mode=True,
            concept_extractor=ConceptExtractor(method="nltk"),
        )
        for i in trange(len(srag_data), desc="S-RAG 选择文档"):
            item = srag_data[i]
            indices, selected, _ = selector.get_selected_documents(
                item["question"], item["docs"],
                item["costs"], item["retriever_scores"]
            )
            sel_costs = [item["costs"][j] for j in indices]
            selected = truncate_passages_to_budget(
                selected, sel_costs, args.budget, tokenizer
            )
            prompts.append(build_prompt(item["question"], selected, tokenizer))

    elif args.method == "topk":
        for i in trange(len(srag_data), desc="Top-k 选择文档"):
            item = srag_data[i]
            indices = select_topk(
                item["docs"], item["costs"],
                item["retriever_scores"], args.budget
            )
            selected = [item["docs"][j] for j in indices]
            sel_costs = [item["costs"][j] for j in indices]
            selected = truncate_passages_to_budget(
                selected, sel_costs, args.budget, tokenizer
            )
            prompts.append(build_prompt(item["question"], selected, tokenizer))

    elif args.method == "mmr":
        lam = MMR_LAMBDA.get(args.dataset, 0.6)
        for i in trange(len(srag_data), desc=f"MMR(λ={lam}) 选择文档"):
            item = srag_data[i]
            indices = select_mmr(
                item["docs"], item["costs"],
                item["retriever_scores"], args.budget,
                lambda_mmr=lam
            )
            selected = [item["docs"][j] for j in indices]
            sel_costs = [item["costs"][j] for j in indices]
            selected = truncate_passages_to_budget(
                selected, sel_costs, args.budget, tokenizer
            )
            prompts.append(build_prompt(item["question"], selected, tokenizer))

    elif args.method == "greedy":
        for i in trange(len(srag_data), desc="Greedy(Rel/Cost) 选择文档"):
            item = srag_data[i]
            indices = select_greedy_rel_cost(
                item["docs"], item["costs"],
                item["retriever_scores"], args.budget
            )
            selected = [item["docs"][j] for j in indices]
            sel_costs = [item["costs"][j] for j in indices]
            selected = truncate_passages_to_budget(
                selected, sel_costs, args.budget, tokenizer
            )
            prompts.append(build_prompt(item["question"], selected, tokenizer))

    else:
        raise ValueError(f"未知方法：{args.method}")

    outputs = []
    model.eval()
    for i in trange(0, len(prompts), args.batch_size, desc="生成答案"):
        batch = prompts[i : i + args.batch_size]
        tokenized = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
        with torch.no_grad():
            out = model.generate(
                **tokenized,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        prompt_len = tokenized["input_ids"].shape[1]
        decoded = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
            for o in out
        ]
        outputs.extend(decoded)

    results = evaluate(outputs, [d["answers"] for d in srag_data], args.dataset)
    print(json.dumps(results, indent=2))

    result_file = f"{args.dataset}_{args.method}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    pred_file = f"{args.dataset}_{args.method}_predictions.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f:
        for item, out in zip(srag_data, outputs):
            f.write(json.dumps({
                "question":   item["question"],
                "prediction": out,
                "answers":    item["answers"],
            }, ensure_ascii=False) + "\n")

    print(f"结果已保存至 {result_file}")
    print(f"预测已保存至 {pred_file}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S-RAG 生成 & 评测")
    parser.add_argument("--dataset",       required=True,
                        choices=["nq", "eli5", "hotpotqa"])
    parser.add_argument("--dataset_path",  required=True)
    parser.add_argument("--model_name",    required=True)
    parser.add_argument("--tokenizer_name",required=True)
    parser.add_argument("--method",        default="topk",
                        choices=["srag", "topk", "mmr", "greedy"])
    parser.add_argument("--k",             type=int, default=200)
    parser.add_argument("--budget",        type=int, default=4096)
    parser.add_argument("--concept_depth", type=int, default=20)
    parser.add_argument("--batch_size",    type=int, default=1)
    parser.add_argument("--retriever_name",type=str, default="./bge-large-en-v1.5")
    parser.add_argument("--corpus_dir",    type=str, default="./wiki_dpr_text_only",
                        help="wiki_dpr parquet 文件所在目录")
    args = parser.parse_args()
    main(args)