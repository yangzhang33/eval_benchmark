
from collections import Counter
from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import HfApi

HF_REPO_ID = "yangzhang33/culture-eval-benchmark"
LITE_REPO_ID = "yangzhang33/cultural_eval_lite"
MAX_SAMPLES = 1000
SEED = 42

CS_CONFIGS = [
    "chinese_cs",
    "arabic_cs",
    "greek_cs",
    "hindi_cs",
    "indonesian_cs",
    "korean_cs",
]


def sample_proportionally(data, max_samples, seed=SEED):
    total = len(data)
    if total <= max_samples:
        return data

    counts = Counter(data["config"])
    parts = []
    for config_val, count in counts.items():
        quota = max(1, round(count / total * max_samples))
        subset = data.filter(lambda x, v=config_val: x["config"] == v)
        subset = subset.shuffle(seed=seed).select(range(min(quota, len(subset))))
        parts.append(subset)
    return concatenate_datasets(parts)


lite_configs = {}

# CS configs — cap at 1000, proportional by config value
for lang in CS_CONFIGS:
    print(f"\n=== {lang} ===")
    data = load_dataset(HF_REPO_ID, lang)["test"]
    counts = Counter(data["config"])
    print(f"Total: {len(data)}")
    for config_val, count in sorted(counts.items()):
        print(f"  {config_val}: {count}")

    sampled = sample_proportionally(data, MAX_SAMPLES)
    sampled_counts = Counter(sampled["config"])
    print(f"After sampling: {len(sampled)}")
    for config_val, count in sorted(sampled_counts.items()):
        print(f"  {config_val}: {count}")

    lite_configs[lang] = sampled

# english_ca — full data
print("\n=== english_ca ===")
english_ca = load_dataset(HF_REPO_ID, "english_ca")["test"]
print(f"Total: {len(english_ca)}")
lite_configs["english_ca"] = english_ca

# greek_cs_en — full data
print("\n=== greek_cs_en ===")
greek_cs_en = load_dataset(HF_REPO_ID, "greek_cs_en")["test"]
print(f"Total: {len(greek_cs_en)}")
lite_configs["greek_cs_en"] = greek_cs_en

# Upload
print(f"\n=== Uploading to {LITE_REPO_ID} ===")
api = HfApi()
api.create_repo(repo_id=LITE_REPO_ID, repo_type="dataset", exist_ok=True)

for config_name, data in lite_configs.items():
    DatasetDict({"test": data}).push_to_hub(LITE_REPO_ID, config_name=config_name)
    print(f"Uploaded {config_name}: {len(data)} examples")

print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{LITE_REPO_ID}")
