"""
Creates and uploads a HuggingFace dataset subset:
  - arabic_cs: Arabic-specific tasks from MBZUAI/ArabicMMLU
               (only questions with exactly 4 non-null choices, answers kept as-is)
"""

from datasets import load_dataset, DatasetDict, concatenate_datasets, Value
from huggingface_hub import HfApi

HF_REPO_ID = "yangzhang33/culture-eval-benchmark"

# ---------------------------------------------------------------------------
# 1. arabic_cs — Arabic configs from MBZUAI/ArabicMMLU
# ---------------------------------------------------------------------------
print("=== Loading ArabicMMLU configs ===")

arabic_configs = [
    'Islamic Studies',
    'Islamic Studies (Middle School)',
    'Islamic Studies (Primary School)',
    'Islamic Studies (High School)',
    'Arabic Language (Middle School)',
    'Arabic Language (Primary School)',
    'Arabic Language (High School)',
    'Arabic Language (General)',
    'Arabic Language (Grammar)',
    'Driving Test',
    'History (Middle School)',
    'History (Primary School)',
    'History (High School)',
    'Geography (Middle School)',
    'Geography (Primary School)',
    'Geography (High School)',
    'Civics (Middle School)',
    'Civics (High School)',
    'Law (Professional)',
    'Political Science (University)',
]

all_parts = []

for config in arabic_configs:
    ds = load_dataset("MBZUAI/ArabicMMLU", config, trust_remote_code=True)
    for split, data in ds.items():
        if split != "test":
            continue
        # Keep only rows where Option 1-4 are all non-null and Option 5 is null
        def keep_row(x):
            return (
                x["Option 1"] is not None and
                x["Option 2"] is not None and
                x["Option 3"] is not None and
                x["Option 4"] is not None and
                x["Option 5"] is None
            )

        data = data.filter(keep_row)

        def transform(example):
            return {
                "question": example["Question"],
                "option_a": example["Option 1"],
                "option_b": example["Option 2"],
                "option_c": example["Option 3"],
                "option_d": example["Option 4"],
                "answer": example["Answer Key"],
                "config": config,
            }

        data = data.map(transform, remove_columns=["Question", "Option 1", "Option 2", "Option 3", "Option 4", "Option 5", "Answer Key"])
        for col in ["Country", "Level", "Group", "Source", "Context"]:
            if col in data.features and data.features[col].dtype != "string":
                data = data.cast_column(col, Value("string"))
        all_parts.append(data)
        print(f"  {config} [{split}]: {len(data)} examples (4-choice)")

arabic_cs = DatasetDict({"test": concatenate_datasets(all_parts)})
print(f"\narabic_cs test split: {len(arabic_cs['test'])} examples")

# ---------------------------------------------------------------------------
# 2. Upload to HuggingFace Hub
# ---------------------------------------------------------------------------
print(f"\n=== Uploading to {HF_REPO_ID} ===")

api = HfApi()
api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

arabic_cs.push_to_hub(HF_REPO_ID, config_name="arabic_cs")
print("Uploaded arabic_cs")

print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{HF_REPO_ID}")
