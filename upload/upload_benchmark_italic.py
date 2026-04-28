"""
Creates and uploads a HuggingFace dataset subset:
  - italic_cs: Culture and commonsense tasks from yangzhang33/italic-qa
               (only questions with macro_category == "culture and commonsense")
"""

from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi

HF_REPO_ID = "yangzhang33/culture-eval-benchmark"

# ---------------------------------------------------------------------------
# 1. italic_cs — culture and commonsense split from yangzhang33/italic-qa
# ---------------------------------------------------------------------------
print("=== Loading italic-qa ===")

ds = load_dataset("yangzhang33/italic-qa", trust_remote_code=True)

TARGET_CATEGORY = "culture and commonsense"

all_parts = []

for split, data in ds.items():
    filtered = data.filter(
        lambda x: x["macro_category"].strip().lower() == TARGET_CATEGORY
    )

    all_parts.append(filtered)
    print(f"  [{split}]: {len(filtered)} examples (macro_category='{TARGET_CATEGORY}')")

from datasets import concatenate_datasets
italic_cs = DatasetDict({"test": concatenate_datasets(all_parts)})
print(f"\nitalic_cs test split: {len(italic_cs['test'])} examples")

# ---------------------------------------------------------------------------
# 2. Upload to HuggingFace Hub
# ---------------------------------------------------------------------------
print(f"\n=== Uploading to {HF_REPO_ID} ===")

api = HfApi()
api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

italic_cs.push_to_hub(HF_REPO_ID, config_name="italic_cs")
print("Uploaded italic_cs")

print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{HF_REPO_ID}")
