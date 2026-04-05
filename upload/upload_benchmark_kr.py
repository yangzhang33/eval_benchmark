"""
Creates and uploads a HuggingFace dataset subset:
  - korean_cs: Korean-specific tasks from HAERAE-HUB/KMMLU
"""

from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import HfApi

HF_REPO_ID = "yangzhang33/culture-eval-benchmark"

# ---------------------------------------------------------------------------
# 1. korean_cs — Korean configs from HAERAE-HUB/KMMLU
# ---------------------------------------------------------------------------
print("=== Loading KMMLU configs ===")

kmmlu_configs = [
    "Korean-History",
    "Law",
    "Criminal-Law",
    "Taxation",
    "Real-Estate",
    "Social-Welfare",
    "Public-Safety",
    "Patent",
    "Geomatics",
    "Maritime-Engineering",
    "Ecology",
    "Accounting",
    "Education",
    "Health",
]

ANSWER_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}

all_parts = []

for config in kmmlu_configs:
    ds = load_dataset("HAERAE-HUB/KMMLU", config, trust_remote_code=True)
    if "test" not in ds:
        print(f"  {config}: no test split, skipping")
        continue
    data = ds["test"]

    def transform(example, cfg=config):
        return {
            "question": example["question"],
            "option_a": example["A"],
            "option_b": example["B"],
            "option_c": example["C"],
            "option_d": example["D"],
            "answer": ANSWER_MAP[str(example["answer"])],
            "config": cfg,
        }

    data = data.map(transform, remove_columns=["question", "A", "B", "C", "D", "answer"])
    all_parts.append(data)
    print(f"  {config} [test]: {len(data)} examples")

korean_cs = DatasetDict({"test": concatenate_datasets(all_parts)})
print(f"\nkorean_cs test split: {len(korean_cs['test'])} examples")

# ---------------------------------------------------------------------------
# 2. Upload to HuggingFace Hub
# ---------------------------------------------------------------------------
print(f"\n=== Uploading to {HF_REPO_ID} ===")

api = HfApi()
api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

korean_cs.push_to_hub(HF_REPO_ID, config_name="korean_cs")
print("Uploaded korean_cs")

print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{HF_REPO_ID}")
