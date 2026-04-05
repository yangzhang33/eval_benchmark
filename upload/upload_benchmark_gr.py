"""
Creates and uploads a HuggingFace dataset subset:
  - greek_cs: Greek-specific tasks from dascim/GreekMMLU
               (only questions with exactly 4 choices, answers converted to A/B/C/D)
"""

from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import HfApi

HF_REPO_ID = "yangzhang33/culture-eval-benchmark"

# ---------------------------------------------------------------------------
# 1. greek_cs — Greek configs from dascim/GreekMMLU
# ---------------------------------------------------------------------------
print("=== Loading GreekMMLU configs ===")

greek_configs = [
    'Greek_History_Primary_School',
    'Greek_History_Professional',
    'Greek_History_Secondary_School',
    'Greek_Literature',
    'Greek_Mythology',
    'Greek_Traditions',
]

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

all_parts = []

for config in greek_configs:
    ds = load_dataset("dascim/GreekMMLU", config, trust_remote_code=True)
    for split, data in ds.items():
        # Keep only rows with exactly 4 choices
        data = data.filter(lambda x: len(x["choices"]) == 4)

        # Expand choices list into option_a/b/c/d and convert answer to letter
        def transform(example):
            choices = example["choices"]
            return {
                "question": example["question"],
                "option_a": choices[0],
                "option_b": choices[1],
                "option_c": choices[2],
                "option_d": choices[3],
                "answer": ANSWER_MAP[example["answer"]],
                "config": config,
            }

        data = data.map(transform, remove_columns=["question", "choices", "answer"])
        all_parts.append(data)
        print(f"  {config} [{split}]: {len(data)} examples (4-choice)")

greek_cs = DatasetDict({"test": concatenate_datasets(all_parts)})
print(f"\ngreek_cs test split: {len(greek_cs['test'])} examples")

# ---------------------------------------------------------------------------
# 2. Upload to HuggingFace Hub
# ---------------------------------------------------------------------------
print(f"\n=== Uploading to {HF_REPO_ID} ===")

api = HfApi()
api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

greek_cs.push_to_hub(HF_REPO_ID, config_name="greek_cs")
print("Uploaded greek_cs")

print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{HF_REPO_ID}")
