import os
import json
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Step 1：下载 NQ (Natural Questions)
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists("./nq_test.jsonl"):
    print("下载 NQ 测试集 ...")
    nq = load_dataset("facebook/kilt_tasks", "nq", split="validation")
    with open("./nq_test.jsonl", "w", encoding="utf-8") as f:
        for item in nq:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"NQ 保存完毕，共 {len(nq)} 条。")
else:
    print("nq_test.jsonl 已存在，跳过。")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2：下载 ELI5
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists("./eli5_test.jsonl"):
    print("下载 ELI5 测试集 ...")
    eli5 = load_dataset("facebook/kilt_tasks", "eli5", split="validation")
    with open("./eli5_test.jsonl", "w", encoding="utf-8") as f:
        for item in eli5:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"ELI5 保存完毕，共 {len(eli5)} 条。")
else:
    print("eli5_test.jsonl 已存在，跳过。")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3：下载 HotpotQA
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists("./hotpotqa_test.jsonl"):
    print("下载 HotpotQA 测试集 ...")
    hotpot = load_dataset("facebook/kilt_tasks", "hotpotqa", split="validation")
    with open("./hotpotqa_test.jsonl", "w", encoding="utf-8") as f:
        for item in hotpot:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"HotpotQA 保存完毕，共 {len(hotpot)} 条。")
else:
    print("hotpotqa_test.jsonl 已存在，跳过。")

print("\n✅ 所有数据集下载完毕！")