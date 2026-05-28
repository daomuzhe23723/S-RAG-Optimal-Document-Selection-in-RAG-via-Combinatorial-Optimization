import os
from datasets import load_dataset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_ENDPOINT'] = 'https://hf-mirror.com' 

dataset = load_dataset(
    "wiki_dpr",
    "psgs_w100.multiset.no_index.no_embeddings",  
    split="train",
    trust_remote_code=True
)
dataset.save_to_disk("./wiki_dpr_text_only")

from transformers import AutoModel, AutoTokenizer

model_name = "roberta-large"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./roberta-large")
tokenizer.save_pretrained("./roberta-large")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
model.save('./bge-large-en-v1.5')
