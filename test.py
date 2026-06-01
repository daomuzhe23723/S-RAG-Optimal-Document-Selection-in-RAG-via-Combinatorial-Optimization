from huggingface_hub import list_datasets
results = list(list_datasets(search="eli5", limit=20))
for r in results:
    print(r.id)