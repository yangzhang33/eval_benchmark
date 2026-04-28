"""
For each config in yangzhang33/culture-eval-benchmark, adds two fields:
  - CEB_index: integer index starting from 0 within that config
  - CEB_config: the config name the example belongs to
Then re-uploads each config back to the same repo.
"""

from datasets import load_dataset, DatasetDict, get_dataset_config_names

HF_REPO_ID = "yangzhang33/culture-eval-benchmark"

configs = [
    # "arabic_ca", "arabic_cs",
    # "chinese_ca", "chinese_cs",
    # "english_ca",
    # "greek_ca", "greek_cs",
    # "hindi_ca", "hindi_cs",
    # "indonesian_ca", "indonesian_cs",
    # "korean_ca", "korean_cs",
    "italic_ca", "italic_cs",
]
print(f"Processing {len(configs)} configs: {configs}")

for config_name in configs:
    print(f"\n=== {config_name} ===")
    ds = load_dataset(HF_REPO_ID, config_name)

    new_splits = {}
    for split_name, split_data in ds.items():
        def add_fields(example, idx, config=config_name):
            return {"CEB_index": idx, "CEB_config": config}

        split_data = split_data.map(add_fields, with_indices=True)
        new_splits[split_name] = split_data
        print(f"  {split_name}: {len(split_data)} examples")

    DatasetDict(new_splits).push_to_hub(HF_REPO_ID, config_name=config_name)
    print(f"  Uploaded {config_name}")

print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{HF_REPO_ID}")
