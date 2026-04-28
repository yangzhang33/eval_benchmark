"""
Creates and uploads CA (Culturally Aware) items from CohereLabs/Global-MMLU
for multiple languages to HuggingFace Hub.

Each language is uploaded as a separate config: english_ca, chinese_ca,
arabic_ca, greek_ca, etc.
"""

from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi

HF_REPO_ID = "yangzhang33/culture-eval-benchmark"

# Language code -> config name
LANGUAGES = {
    # "en": "english_ca",
    # "zh": "chinese_ca",
    # "ar": "arabic_ca",
    # "el": "greek_ca",
    # "hi": "hindi_ca",
    # "id": "indonesian_ca",
    # "ko": "korean_ca",
    "it": "italic_ca",
}

api = HfApi()
api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

for lang_code, config_name in LANGUAGES.items():
    print(f"=== Loading Global-MMLU '{lang_code}' (CA items) ===")
    ds = load_dataset("CohereLabs/Global-MMLU", lang_code)
    ca_splits = {}
    for split, data in ds.items():
        if split != "test":
            continue
        filtered = data.filter(lambda x: x["cultural_sensitivity_label"] == "CA")
        ca_splits[split] = filtered
        print(f"  Split '{split}': {len(filtered)} CA examples")

    ca_dataset = DatasetDict(ca_splits)
    ca_dataset.push_to_hub(HF_REPO_ID, config_name=config_name)
    print(f"Uploaded {config_name}\n")

print(f"Done! Dataset available at: https://huggingface.co/datasets/{HF_REPO_ID}")
