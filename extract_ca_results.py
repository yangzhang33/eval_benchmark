"""
Extract all _ca sub-results from accuracy and predictions JSON files
in lite_eval_loglik_v1, and save filtered versions to ca_results/.
"""

import json
import os
from pathlib import Path

SRC_DIR = Path("/datalake/datastore1/yang/eval_benchmark/results/lite_eval_loglik_v1")
OUT_DIR = Path("/datalake/datastore1/yang/eval_benchmark/results/ca_results")

OUT_DIR.mkdir(parents=True, exist_ok=True)

accuracy_files = sorted(SRC_DIR.glob("*_accuracy.json"))
prediction_files = sorted(SRC_DIR.glob("*_predictions.json"))

print(f"Found {len(accuracy_files)} accuracy files, {len(prediction_files)} prediction files")

for path in accuracy_files:
    with open(path) as f:
        data = json.load(f)

    ca_accuracy = {k: v for k, v in data["accuracy"].items() if k.endswith("_ca")}

    out = {"model": data["model"], "accuracy": ca_accuracy}
    out_path = OUT_DIR / path.name
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  [accuracy] {path.name} -> {len(ca_accuracy)} _ca keys")

for path in prediction_files:
    with open(path) as f:
        data = json.load(f)

    ca_predictions = [p for p in data["predictions"] if p.get("subset", "").endswith("_ca")]

    out = {"model": data["model"], "predictions": ca_predictions}
    out_path = OUT_DIR / path.name
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  [predictions] {path.name} -> {len(ca_predictions)} _ca entries")

print(f"\nDone. Results saved to: {OUT_DIR}")
