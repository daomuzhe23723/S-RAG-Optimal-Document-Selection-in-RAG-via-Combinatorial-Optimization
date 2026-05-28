import json
import os
from dataclasses import dataclass
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from retriever import BGERetriever
from srag_selector import SRAGSelector
from tqdm import trange
import torch
from utils import evaluate
import argparse

os.environ["TRANSFORMERS_VERBOSITY"] = "error" 


def load_nq(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            answers = raw.get("answer")
            question = raw.get("question")
            data.append({
                "question": question,
                "answers": answers
            })
    return data

def load_eli5(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            answers = raw["answers"]["text"] 
            question = raw["title"]
            if raw["selftext"]:
                question = raw["selftext"]
            data.append({
                "question": question,
                "answers": answers
            })
    return data

def load_hotpotqa(path):
    df = pd.read_parquet(path)
    data = []
    for _, row in df.iterrows():
        passages = []
        contexts = row.get("context", []) if pd.notna(row.get("context")) else []
        for ctx in contexts:
            if isinstance(ctx, (list, tuple)) and len(ctx) == 2:
                title, sentences = ctx
                passages.append({
                    "title": str(title),
                    "text": " ".join(sentences) if isinstance(sentences, list) else str(sentences)
                })

        data.append({
            "question": row.get("question", ""),
            "answers": [row.get("answer")] if pd.notna(row.get("answer")) else [],
        })
    return data


def build_srag_inputs(
    inputs,
    retrieved_results,
    tokenizer_name,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    srag_inputs = []
    for input, results in zip(inputs, retrieved_results):
        docs = [d["text"] for d in results]
        costs = [len(tokenizer.encode(d, add_special_tokens=False)) for d in docs]
        srag_inputs.append({
            "question": input["question"],
            "answers": input["answers"],
            "docs": docs,
            "retriever_scores": [d["score"] for d in results],
            "costs": costs,
        })
    return srag_inputs


def main(args):
    max_new_tokens = 0

    retriever = BGERetriever(model_name=args.retriever_name, device="cuda")

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
        raise ValueError(f"Unknown dataset {args.dataset}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        trust_remote_code=True,
        use_fast=False, 
    )
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype="auto",     
        device_map="auto",         
    )
    
    
    questions = [d["question"] for d in data]
    retrieved_results = retriever.retrieve(questions, top_k=args.k)
    print("1111")
    srag_data = build_srag_inputs(data, retrieved_results, args.tokenizer_name)
    print("2222")
    prompts = []
    prefix = ("Answer the question using the passages. Be concise and factual. If multiple passages support the answer, synthesize them. Do not fabricate unsupported claims. Output ONLY the answer phrase, NOT a full sentence. ")
    if args.method == "srag":
        selector = SRAGSelector(budget=args.budget, concept_depth=args.concept_depth, fast_mode=True)
        for i in trange(len(srag_data)):
            indices, selected, total_cost = selector.get_selected_documents(srag_data[i]["question"], srag_data[i]["docs"], srag_data[i]["costs"], srag_data[i]["retriever_scores"])
            prompt = "Question: " + srag_data[i]["question"].strip() + "\n\nPassages:\n"
            for idx, passage in enumerate(selected, start=1):
                prompt += f"{idx}. {passage}\n"
            messages = [{"role": "system", "content": prefix}, {"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

    outputs = []
    model.eval()
    del retriever
    torch.cuda.empty_cache()
    for i in trange(0, len(prompts), args.batch_size):
        batch = prompts[i:i+args.batch_size]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with torch.no_grad():
            output = model.generate(**tokenized_batch, do_sample=False, max_new_tokens=max_new_tokens,use_cache=True)
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]
        outputs.extend(output)

    results = evaluate(outputs, [d["answers"] for d in srag_data], args.dataset)
    with open(f"{args.dataset}_{args.method}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    

    
        pred_file = f"{args.dataset}_{args.method}_predictions.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f:
        for item, out in zip(srag_data, outputs):
            f.write(json.dumps({
                "question": item["question"],
                "prediction": out,
                "answers": item["answers"],
            }, ensure_ascii=False) + "\n")

    print(f"Results saved to {args.dataset}_{args.method}.json") 
    print(f"Predictions saved to {pred_file}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S-RAG Generation & Evaluation")
    parser.add_argument("--dataset", required=True, choices=["nq", "eli5", "hotpotqa"])
    parser.add_argument("--dataset_path", required=True, help="本地测试集路径")
    parser.add_argument("--model_name", required=True, help="生成模型路径或 HF ID")
    parser.add_argument("--tokenizer_name", required=True, help="Tokenizer 路径或 HF ID")
    parser.add_argument("--method", default="srag", choices=["srag", "topk", "mmr"])
    parser.add_argument("--k", type=int, default=200, help="检索候选池大小")
    parser.add_argument("--budget", type=int, default=4096, help="prompt token 预算")
    parser.add_argument("--concept_depth", type=int, default=20, help="概念宇宙深度 L")
    parser.add_argument("--batch_size", type=int, default=1, help="生成 batch size")
    parser.add_argument("--retriever_name", type=str, default="./bge-large-en-v1.5", help="检索模型路径")
    args = parser.parse_args()
    main(args)