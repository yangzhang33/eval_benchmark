"""
Creates and uploads a HuggingFace dataset subset:
  - chinese_cs: all Chinese-specific tasks from CMMLU
"""

from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import HfApi

HF_REPO_ID = "yangzhang33/culture-eval-benchmark"

# ---------------------------------------------------------------------------
# 1. chinese_cs — all Chinese-specific configs from CMMLU
# ---------------------------------------------------------------------------
print("=== Loading CMMLU chinese configs ===")

chinese_configs = [
    'chinese_civil_service_exam',
    'chinese_driving_rule',
    'chinese_food_culture',
    'chinese_foreign_policy',
    'chinese_history',
    'chinese_literature',
    'chinese_teacher_qualification',
    'elementary_chinese',
    'modern_chinese',
    'traditional_chinese_medicine',
]

test_parts = []
for config in chinese_configs:
    ds = load_dataset("lmlmcat/cmmlu", config, trust_remote_code=True)
    test_data = ds["test"]
    test_data = test_data.rename_columns({
        "Question": "question",
        "A": "option_a",
        "B": "option_b",
        "C": "option_c",
        "D": "option_d",
        "Answer": "answer",
    })
    test_data = test_data.add_column("config", [config] * len(test_data))
    test_parts.append(test_data)
    print(f"  Loaded {config}: test={len(test_data)}")

chinese_cs = DatasetDict({"test": concatenate_datasets(test_parts)})
print(f"\nchinese_cs test split: {len(chinese_cs['test'])} examples\n")

# ---------------------------------------------------------------------------
# 2. Upload to HuggingFace Hub
# ---------------------------------------------------------------------------
print(f"=== Uploading to {HF_REPO_ID} ===")

api = HfApi()
api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

chinese_cs.push_to_hub(HF_REPO_ID, config_name="chinese_cs")
print("Uploaded chinese_cs")

print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{HF_REPO_ID}")
