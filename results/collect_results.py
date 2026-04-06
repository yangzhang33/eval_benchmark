import json
import csv
import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from util.results_constants import MODEL_COUNTRY

# RESULTS_DIR = os.path.join(os.path.dirname(__file__), "lite_eval_loglik_v1_5")
# OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "lite_eval_results_loglik_v1_5.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "lite_eval_loglik_v1/ca_results")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "lite_eval_loglik_v1/ca_results/ca_results_results.csv")

json_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_accuracy.json")))

rows = []
all_accuracy_keys = []

for filepath in json_files:
    with open(filepath) as f:
        data = json.load(f)
    model = data["model"]
    accuracy = data.get("accuracy", {})
    for key in accuracy:
        if key not in all_accuracy_keys:
            all_accuracy_keys.append(key)
    rows.append((model, accuracy))

all_accuracy_keys = sorted(all_accuracy_keys)

with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["model", "country"] + all_accuracy_keys)
    for model, accuracy in rows:
        country = MODEL_COUNTRY.get(model, "Unknown")
        writer.writerow([model, country] + [accuracy.get(k, "") for k in all_accuracy_keys])

print(f"Saved {len(rows)} models to {OUTPUT_CSV}")
