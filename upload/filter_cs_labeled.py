"""
1. Loads yangzhang33/culture-eval-benchmark-labeled, filters each config to keep
   only samples where claude_cs == 1, and uploads to yangzhang33/culture-eval-benchmark-cs-filtered.
2. Also loads the CA configs from yangzhang33/culture-eval-benchmark (unlabeled)
   and uploads them as-is to the same destination dataset.
"""

from datasets import load_dataset, DatasetDict, get_dataset_config_names
from huggingface_hub import HfApi

LABELED_REPO_ID = "yangzhang33/culture-eval-benchmark-labeled"
RAW_REPO_ID = "yangzhang33/culture-eval-benchmark"
DST_REPO_ID = "yangzhang33/culture-eval-benchmark-cs-filtered"

RAW_CONFIGS = [
    'arabic_ca',
    'chinese_ca',
    'english_ca',
    'greek_ca',
    'hindi_ca',
    'indonesian_ca',
    'korean_ca',
]

api = HfApi()
api.create_repo(repo_id=DST_REPO_ID, repo_type="dataset", exist_ok=True)

# --- Part 1: labeled CS configs filtered to claude_cs == 1 ---
labeled_configs = get_dataset_config_names(LABELED_REPO_ID)
print(f"Found {len(labeled_configs)} labeled configs: {labeled_configs}")

for config_name in labeled_configs:
    print(f"\n=== {config_name} (labeled) ===")
    ds = load_dataset(LABELED_REPO_ID, config_name)

    filtered_splits = {}
    for split_name, split_data in ds.items():
        original_count = len(split_data)
        filtered = split_data.filter(lambda x: x["claude_cs"] == 1)
        filtered_count = len(filtered)
        print(f"  {split_name}: {original_count} -> {filtered_count} (kept {filtered_count/original_count*100:.1f}%)")
        filtered_splits[split_name] = filtered

    DatasetDict(filtered_splits).push_to_hub(DST_REPO_ID, config_name=config_name)
    print(f"  Uploaded {config_name}")

# --- Part 2: raw CA configs uploaded as-is ---
print(f"\nLoading {len(RAW_CONFIGS)} raw CA configs from {RAW_REPO_ID}")

for config_name in RAW_CONFIGS:
    print(f"\n=== {config_name} (raw) ===")
    ds = load_dataset(RAW_REPO_ID, config_name)

    for split_name, split_data in ds.items():
        print(f"  {split_name}: {len(split_data)} examples")

    ds.push_to_hub(DST_REPO_ID, config_name=config_name)
    print(f"  Uploaded {config_name}")

print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{DST_REPO_ID}")
