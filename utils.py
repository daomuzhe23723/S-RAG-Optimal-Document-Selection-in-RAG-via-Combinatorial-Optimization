import re
import string
from typing import List, Dict, Union, Optional
from collections import Counter
from bert_score import score as bert_score_fn
import torch

import numpy as np
import os


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    for w in ("a", "an", "the"):
        text = re.sub(rf"\b{w}\b", " ", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = " ".join(text.split())
    return text

def exact_match(prediction: str, answer: str) -> int:
    prediction = normalize_text(prediction)
    if not prediction:
        return 0
    for ans in answer:
        if normalize_text(ans) == prediction:
            return 1
    return 0

def evaluate_em(predictions: List[str], answers: List[str]) -> float:
    assert len(predictions) == len(answers)
    em = [exact_match(p, a) for p, a in zip(predictions, answers)]
    return np.mean(em)*100

def lcs(text1: List[str], text2: List[str]) -> int:
    m = len(text1)
    n = len(text2)
    prev_row = np.zeros(n+1, dtype=int)
    for i in range(1, m+1):
        current_row = np.zeros(n+1, dtype=int)
        prev_row[0] = 0
        for j in range(1, n+1):
            if text1[i-1] == text2[j-1]:
                current_row[j] = prev_row[j-1] + 1
            else:
                current_row[j] = max(prev_row[j], current_row[j-1])
        prev_row = current_row
    return prev_row[n]

def compute_rouge_l(prediction: str, reference: str) -> int:
    prediction = normalize_text(prediction)
    reference = normalize_text(reference)
    lcs_len = lcs(prediction.split(), reference.split())
    precision = lcs_len / len(prediction.split())
    recall = lcs_len / len(reference.split())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def rouge_l_best_ref(prediction: str, references: List[str]) -> float:
    """与多个参考分别计算，取最高 F1。"""
    if not references:
        return 0.0
    return max(compute_rouge_l(prediction, ref) for ref in references)
def evaluate_rouge_l(predictions: List[str], references: List[str]) -> float:
    assert len(predictions) == len(references)
    rouge_l = [rouge_l_best_ref(p, r) for p, r in zip(predictions, references)]
    return np.mean(rouge_l)*100

def evaluate_bert_score(predictions: List[str], references: List[str]) -> float:
    assert len(predictions) == len(references)
    flat_preds, flat_refs = [], []
    pred_idx_map = []  # 记录每个 flat 项对应第几个问题
    for idx, (pred, refs) in enumerate(zip(predictions, references)):
        if not refs:
            continue
        for ref in refs:
            flat_preds.append(pred if pred.strip() else ".")
            flat_refs.append(ref if ref.strip() else ".")
            pred_idx_map.append(idx)

    if not flat_preds:
        return 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    lang = "en"
    model_type = "roberta-large"
    model_dir = "./roberta-large"
    P, R, F1 = bert_score_fn(
            flat_preds,
            flat_refs,
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            lang=lang,
            verbose=False,
            rescale_with_baseline=False  
        )
    best_f1_per_sample = [0.0] * len(predictions)
    f1_values = F1.cpu().numpy()
    for sample_idx, f1_val in zip(pred_idx_map, f1_values):
        best_f1_per_sample[sample_idx] = max(best_f1_per_sample[sample_idx], float(f1_val))

    return np.mean(best_f1_per_sample) * 100

def evaluate(
    predictions,
    references,
    dataset
):
    results = {}
    if dataset == "eli5":
        results["rouge_l"] = evaluate_rouge_l(predictions, references)
        results["bert_score"] = evaluate_bert_score(predictions, references)
    else:
        results["em"] = evaluate_em(predictions, references)
        results["rouge_l"] = evaluate_rouge_l(predictions, references)
        results["bert_score"] = evaluate_bert_score(predictions, references)
    return results  
