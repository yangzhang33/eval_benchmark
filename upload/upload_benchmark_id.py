"""
Creates and uploads a HuggingFace dataset subset:
  - indonesian_cs: Indonesian-specific tasks from indolem/IndoMMLU
               (only questions with exactly 4 options, selected subjects)
"""

import ast
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi


def string_to_list(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

HF_REPO_ID = "yangzhang33/culture-eval-benchmark"

# ---------------------------------------------------------------------------
# 1. indonesian_cs — from indolem/IndoMMLU
# ---------------------------------------------------------------------------
print("=== Loading IndoMMLU ===")

selected_subjects = [
    "Indonesian language",
    "Civic education",
    "History",
    "Islam religion",
    "Christian religion",
    "Hindu religion",
    "Geography",
    "Sociology",
    "Minangkabau culture",
    "Social science",
    "Art",
]

ds = load_dataset("indolem/IndoMMLU", "default", trust_remote_code=True)

if "test" not in ds:
    raise ValueError("No test split found in indolem/IndoMMLU")

data = ds["test"]

# Filter by subject and exactly 4 options
def keep_row(x):
    opts = x["options"]
    if isinstance(opts, str):
        opts = string_to_list(opts)
    return (
        x["subject"] in selected_subjects and
        isinstance(opts, list) and
        len(opts) == 4
    )

data = data.filter(keep_row)
print(f"After filtering: {len(data)} examples")

def transform(example):
    opts = example["options"]
    if isinstance(opts, str):
        opts = string_to_list(opts)
    # Strip leading "A. ", "B. ", etc. prefixes if present
    def strip_prefix(s):
        if len(s) >= 3 and s[1] == '.' and s[2] == ' ':
            return s[3:]
        return s

    return {
        "question": example["question"],
        "option_a": strip_prefix(opts[0]),
        "option_b": strip_prefix(opts[1]),
        "option_c": strip_prefix(opts[2]),
        "option_d": strip_prefix(opts[3]),
        "answer": example["answer"],
        "config": example["subject"],
    }

data = data.map(transform, remove_columns=["question", "options", "answer", "subject"])

indonesian_cs = DatasetDict({"test": data})
print(f"\nindonesian_cs test split: {len(indonesian_cs['test'])} examples")

# ---------------------------------------------------------------------------
# 2. Upload to HuggingFace Hub
# ---------------------------------------------------------------------------
print(f"\n=== Uploading to {HF_REPO_ID} ===")

api = HfApi()
api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

indonesian_cs.push_to_hub(HF_REPO_ID, config_name="indonesian_cs")
print("Uploaded indonesian_cs")

print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{HF_REPO_ID}")
