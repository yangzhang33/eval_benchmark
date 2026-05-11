"""
Build accuracy.csv for any CS eval directory, merged with CA results.

Usage:
    python process_results_v2_chinese.py --cs-dir <CS_DIR> [--ca-dir <CA_DIR>]

Defaults:
    --cs-dir  results/cs_filtered_lite_eval_loglik_v2
    --ca-dir  results/ca_loglik_v1/ca_results

The script discovers all subset keys and CA keys automatically from the data,
so it works for any language without code changes.

Output:
    <CS_DIR>/all_results/accuracy.csv
    columns: model, country, english_ca, <lang>_ca …, <lang>_cs, <lang>_cs_en …
"""

import argparse
import glob
import json
import os

import pandas as pd

from util.results_constants import MODEL_COUNTRY, MODELS

METRIC = "accuracy"
DEFAULT_CS_DIR = "results/cs_filtered_lite_eval_loglik_v2"
DEFAULT_CA_DIR = "results/ca_loglik_v1/ca_results"


def load_dir(directory: str) -> dict[str, dict]:
    """Return {filename: {subset: value}} for all *_accuracy.json in directory."""
    result = {}
    for filepath in glob.glob(os.path.join(directory, f"*_{METRIC}.json")):
        with open(filepath) as f:
            d = json.load(f)
        result[os.path.basename(filepath)] = d.get(METRIC, {})
    return result


def ca_keys_for(cs_keys: set[str], ca_sample: dict) -> list[str]:
    """english_ca first, then <lang>_ca for each language present in cs_keys."""
    languages = {k.split("_")[0] for k in cs_keys if k.endswith("_cs")}
    extra = sorted(f"{lang}_ca" for lang in languages if f"{lang}_ca" in ca_sample)
    return ["english_ca"] + extra


def subset_order(cs_keys: set[str], ca_keys: list[str]) -> list[str]:
    """CA keys first, then cs/cs_en pairs grouped by language."""
    languages = sorted({k.split("_")[0] for k in cs_keys if k.endswith("_cs")})
    ordered = list(ca_keys)
    for lang in languages:
        if f"{lang}_cs" in cs_keys:
            ordered.append(f"{lang}_cs")
        if f"{lang}_cs_en" in cs_keys:
            ordered.append(f"{lang}_cs_en")
    return ordered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs-dir", default=DEFAULT_CS_DIR)
    parser.add_argument("--ca-dir", default=DEFAULT_CA_DIR)
    args = parser.parse_args()

    output_dir = os.path.join(args.cs_dir, "all_results")
    os.makedirs(output_dir, exist_ok=True)

    cs_data = load_dir(args.cs_dir)
    ca_data = load_dir(args.ca_dir)

    if not cs_data:
        raise FileNotFoundError(f"No *_{METRIC}.json files found in {args.cs_dir}")

    all_cs_keys: set[str] = set()
    for vals in cs_data.values():
        all_cs_keys.update(vals.keys())

    ca_sample = next(iter(ca_data.values()), {}) if ca_data else {}
    ca_keys = ca_keys_for(all_cs_keys, ca_sample)
    ordered_subsets = subset_order(all_cs_keys, ca_keys)

    records = []
    for filename, cs_vals in cs_data.items():
        with open(os.path.join(args.cs_dir, filename)) as f:
            model = json.load(f)["model"]

        merged = dict(cs_vals)
        if filename in ca_data:
            for key in ca_keys:
                if key in ca_data[filename]:
                    merged[key] = ca_data[filename][key]

        rec = {"model": model, "country": MODEL_COUNTRY.get(model, "Unknown")}
        for subset in ordered_subsets:
            if subset in merged:
                rec[subset] = merged[subset]
        records.append(rec)

    df = pd.DataFrame(records)
    subset_cols = [s for s in ordered_subsets if s in df.columns]
    df = df[["model", "country"] + subset_cols]

    df["model"] = pd.Categorical(df["model"], categories=MODELS, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)
    df["model"] = df["model"].astype(str)

    out_path = os.path.join(output_dir, f"{METRIC}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} models → {out_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
