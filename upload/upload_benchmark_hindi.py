"""
Creates and uploads a HuggingFace dataset subset:
  - hindi_cs: Hindi-specific tasks from ai4bharat/MILU
               (Hindi config, selected domains, non-translated only)
"""

from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import HfApi

HF_REPO_ID = "yangzhang33/culture-eval-benchmark"

ANSWER_MAP = {
    "option1": "A",
    "option2": "B",
    "option3": "C",
    "option4": "D",
}

DOMAINS = [
    'Arts & Humanities',
    'Law & Governance',
    'Social Sciences',
    'Health & Medicine',
]

# ---------------------------------------------------------------------------
# 1. hindi_cs — Hindi config from ai4bharat/MILU
# ---------------------------------------------------------------------------
print("=== Loading MILU Hindi config ===")

ds = load_dataset("ai4bharat/MILU", "Hindi", trust_remote_code=True)

all_parts = []

for split, data in ds.items():
    if split != "test":
        continue

    data = data.filter(lambda x: x["domain"] in DOMAINS and x["is_translated"] == False)

    def transform(example):
        return {
            "question": example["question"],
            "option_a": example["option1"],
            "option_b": example["option2"],
            "option_c": example["option3"],
            "option_d": example["option4"],
            "answer": ANSWER_MAP[example["target"]],
            "config": example["domain"],
        }

    data = data.map(transform, remove_columns=["question", "option1", "option2", "option3", "option4", "target", "domain"])
    all_parts.append(data)
    print(f"  [{split}]: {len(data)} examples")

hindi_cs = DatasetDict({"test": concatenate_datasets(all_parts)})
print(f"\nhindi_cs test split: {len(hindi_cs['test'])} examples")

# ---------------------------------------------------------------------------
# 2. Upload to HuggingFace Hub
# ---------------------------------------------------------------------------
print(f"\n=== Uploading to {HF_REPO_ID} ===")

api = HfApi()
api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

hindi_cs.push_to_hub(HF_REPO_ID, config_name="hindi_cs")
print("Uploaded hindi_cs")

print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{HF_REPO_ID}")
