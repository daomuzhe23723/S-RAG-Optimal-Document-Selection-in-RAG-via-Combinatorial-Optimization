### Installation of Dependencies

```bash
conda create -n cs240 python=3.11
conda activate cs240
pip install -r requirements.txt
```

```bash
python tmp.py
```

### Usage
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py --dataset nq --dataset_path data/NQ-open.dev.jsonl --model_name ./Qwen2.5-1.5B-Instruct --tokenizer_name ./Qwen2.5-1.5B-Instruct --method srag
```

