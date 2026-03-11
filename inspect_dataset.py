#!/usr/bin/env python3
"""
Inspect yangzhang33/cultural_eval_lite offline.
Prints all configs and the number of samples per config.

Usage:
  python inspect_dataset.py
"""
from datasets import get_dataset_config_names, load_dataset

DATASET = "yangzhang33/cultural_eval_lite"

configs = get_dataset_config_names(DATASET, trust_remote_code=True)

print(f"Dataset: {DATASET}")
print("Configs:")
for config in configs:
    try:
        ds = load_dataset(DATASET, config, split="test", trust_remote_code=True, num_proc=1)
        num_samples = len(ds)
    except Exception as e:
        num_samples = f"Error: {e}"
    print(f"  - {config} ({num_samples} samples)")
