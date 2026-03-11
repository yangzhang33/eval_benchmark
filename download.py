#!/usr/bin/env python3
"""Download models and datasets from HuggingFace.

Edit the lists below then run:  python download.py
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "1"
from huggingface_hub import snapshot_download

MAX_WORKERS = 1  # concurrent file downloads per repo

REPOS = [
    # Chinese models
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-ai/deepseek-llm-7b-chat",
    # English models
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    # Greek models
    "ilsp/Llama-Krikri-8B-Instruct",
    "ilsp/Meltemi-7B-Instruct-v1.5",
    # Arabic models
    "inceptionai/jais-13b",
    "inceptionai/jais-13b-chat",
    "inceptionai/Jais-2-8B-Chat",
    "FreedomIntelligence/AceGPT-v2-8B",
    "FreedomIntelligence/AceGPT-v2-8B-Chat",
    # Hindi models
    "sarvamai/OpenHathi-7B-Hi-v0.1-Base",
    "krutrim-ai-labs/Krutrim-1-instruct",
    # Southeast Asian models
    "aisingapore/Llama-SEA-LION-v3-8B",
    "aisingapore/Llama-SEA-LION-v3-8B-IT",
    "SeaLLMs/SeaLLM-7B-v2.5",
    "SeaLLMs/SeaLLMs-v3-7B",
    "SeaLLMs/SeaLLMs-v3-7B-Chat",
    # Korean models
    "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B",
    "beomi/Llama-3-Open-Ko-8B",
    "EleutherAI/polyglot-ko-12.8b",
    "EleutherAI/polyglot-ko-5.8b",
    # Multilingual models
    "CohereLabs/aya-expanse-8b",
]

if __name__ == "__main__":
    for repo_id in REPOS:
        repo_type = "dataset" if repo_id.startswith("datasets/") else "model"
        name = repo_id.removeprefix("datasets/")
        print(f"\n>>> [{repo_type}] {name}")
        try:
            snapshot_download(name, repo_type=repo_type, max_workers=MAX_WORKERS, resume_download=True)
            print(f"    Done: {name}")
        except Exception as e:
            print(f"    Failed: {name} — {e}")

    print("\nAll downloads complete.")
