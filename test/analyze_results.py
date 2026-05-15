#!/usr/bin/env python3
"""
Collect and display results from test_top_token.py runs.

New layout (one JSON per model): test/results/<slug>.json with structure:
  {
    "model": ...,
    "prefix_used_for_scoring": ...,
    "option_ids_used": {...},
    "all_letter_variants": {...},
    "subsets": {
        "<subset_name>": {
            "lang": ..., "n_samples": ...,
            "top1_is_answer_letter_variant_rate": ...,
            "top1_matches_used_prefix_rate": ...,
            "records": [...]
        }, ...
    }
  }

Produces:
  1. Per-model tokenizer info (which prefix was chosen).
  2. Matrix: rows=models, cols=subsets, value=top1_is_letter_rate.
  3. Matrix: rows=models, cols=subsets, value=top1_matches_used_prefix_rate.
  4. Worst-N (model, subset) cases with example top-K predictions.
  5. CSV exports of the two matrices.

Usage:
  python analyze_results.py
  python analyze_results.py --resultsdir results --worst 20
"""
import argparse
import contextlib
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


SKIP_NAMES = {"summary.json", "model_prefix_summary.json"}


def load_all(resultsdir: Path):
    """Return dict {model_id: model_results}."""
    models = {}
    for p in sorted(resultsdir.glob("*.json")):
        if p.name in SKIP_NAMES:
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if "subsets" not in d or "model" not in d:
            continue
        models[d["model"]] = d
    return models


def print_section(title: str):
    print(f"\n{'═' * 100}")
    print(f" {title}")
    print(f"{'═' * 100}")


def print_tokenizer_table(models: dict):
    print_section("1. Per-model tokenizer info — which prefix was chosen")
    print(f"{'Model':<58} {'chosen':<8} {'working':<18} {'option_ids (chosen prefix)'}")
    print("-" * 130)
    for model, d in sorted(models.items()):
        chosen = repr(d["prefix_used_for_scoring"])
        working = str(d.get("all_letter_variants_raw_keys", []))
        ids = d["option_ids_used"]
        ids_str = " ".join(f"{l}={i}" for l, i in ids.items())
        print(f"{model:<58} {chosen:<8} {working:<18} {ids_str}")


def build_matrix(models: dict, rate_key: str):
    """Return (models_sorted, subsets_sorted, matrix[m][s] = value)."""
    model_ids = sorted(models.keys())
    subset_names = sorted({s for d in models.values() for s in d["subsets"].keys()})
    matrix = defaultdict(dict)
    for m, d in models.items():
        for s, info in d["subsets"].items():
            matrix[m][s] = info[rate_key]
    return model_ids, subset_names, matrix


def print_matrix(models: dict, rate_key: str, title: str, fmt: str = "{:>5.1%}"):
    print_section(title)
    model_ids, subset_names, matrix = build_matrix(models, rate_key)
    if not model_ids:
        print("(no data)")
        return
    print(f"{'Model':<55} | " + " ".join(f"{s[:8]:>8}" for s in subset_names))
    print("-" * (55 + 3 + 9 * len(subset_names)))
    for m in model_ids:
        row = matrix[m]
        cells = []
        for s in subset_names:
            v = row.get(s)
            cells.append(f"{fmt.format(v):>8}" if v is not None else f"{'·':>8}")
        print(f"{m:<55} | " + " ".join(cells))

    print("-" * (55 + 3 + 9 * len(subset_names)))
    means_per_subset = []
    for s in subset_names:
        vals = [matrix[m][s] for m in model_ids if s in matrix[m]]
        means_per_subset.append(sum(vals) / len(vals) if vals else 0.0)
    print(f"{'mean over models':<55} | " + " ".join(f"{v:>8.1%}" for v in means_per_subset))

    print("\nPer-model mean across subsets (sorted, worst first):")
    means_per_model = []
    for m in model_ids:
        vals = [matrix[m][s] for s in subset_names if s in matrix[m]]
        means_per_model.append((m, sum(vals) / len(vals) if vals else 0.0))
    for m, v in sorted(means_per_model, key=lambda x: x[1]):
        print(f"  {v:>6.1%}  {m}")


def print_worst_cases(models: dict, n_worst: int):
    print_section(f"4. Worst-{n_worst} (model, subset) by top1_is_letter_rate")
    rows = []
    for m, d in models.items():
        for s, info in d["subsets"].items():
            rows.append({
                "model": m, "subset": s,
                "rate_letter": info["top1_is_answer_letter_variant_rate"],
                "rate_match": info["top1_matches_used_prefix_rate"],
                "prefix_used": d["prefix_used_for_scoring"],
                "records": info["records"],
            })
    rows.sort(key=lambda r: r["rate_letter"])
    for i, r in enumerate(rows[:n_worst]):
        print(f"\n[{i+1}] {r['model']}  |  {r['subset']}  |  prefix={r['prefix_used']!r}")
        print(f"    top1_is_letter_rate = {r['rate_letter']:.2%}   "
              f"top1_matches_used_prefix_rate = {r['rate_match']:.2%}")
        non_letter = [rec for rec in r["records"] if not rec["top1_is_answer_letter_variant"]]
        for j, rec in enumerate(non_letter[:3]):
            tail = rec["prompt_tail"].replace("\n", "\\n")
            print(f"    sample {j+1}: gold={rec['gold_answer']}, prompt tail={tail!r}")
            print(f"              top-5 predicted tokens:")
            for tk in rec["top_k_tokens"][:5]:
                mark = "✓" if tk["is_answer_letter_variant"] else " "
                print(f"                {mark} rank={tk['rank']}  "
                      f"id={tk['token_id']:<6}  "
                      f"str={tk['token_repr']:<15}  "
                      f"prob={tk['prob']:.4f}")


def write_matrix_csv(models: dict, rate_key: str, out_path: Path):
    model_ids, subset_names, matrix = build_matrix(models, rate_key)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model"] + subset_names)
        for m in model_ids:
            row = [m] + [matrix[m].get(s, "") for s in subset_names]
            w.writerow(row)
    print(f"\nWrote {out_path}")


def print_overall(models: dict):
    print_section("5. Overall stats")
    n_models = len(models)
    all_subsets = {s for d in models.values() for s in d["subsets"].keys()}
    n_pairs = sum(len(d["subsets"]) for d in models.values())
    print(f"  Models with results        : {n_models}")
    print(f"  Distinct subsets seen      : {len(all_subsets)}")
    print(f"  (model, subset) pairs      : {n_pairs}")

    prefix_count = defaultdict(int)
    for d in models.values():
        prefix_count[repr(d["prefix_used_for_scoring"])] += 1
    print(f"\n  Prefix chosen across {n_models} models:")
    for p, c in sorted(prefix_count.items(), key=lambda x: -x[1]):
        print(f"    {p:<6} : {c} models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultsdir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--worst", type=int, default=15,
                        help="Show this many worst (model,subset) cases by top1_is_letter_rate")
    parser.add_argument("--no-csv", action="store_true", help="Skip writing CSV matrices")
    parser.add_argument("--out", default=None,
                        help="Path to save the full report (default: <resultsdir>/report.txt). "
                             "Use '-' to disable saving.")
    parser.add_argument("--no-stdout", action="store_true",
                        help="Only write to the report file, not the terminal")
    args = parser.parse_args()

    resultsdir = Path(args.resultsdir)
    if not resultsdir.exists():
        print(f"Results directory not found: {resultsdir}")
        return

    models = load_all(resultsdir)
    if not models:
        print(f"No per-model result files found in {resultsdir}")
        return

    out_path = None if args.out == "-" else Path(args.out) if args.out else resultsdir / "report.txt"
    fh = open(out_path, "w", encoding="utf-8") if out_path else None
    if fh and args.no_stdout:
        target = fh
    elif fh:
        target = _Tee(sys.stdout, fh)
    else:
        target = sys.stdout

    with contextlib.redirect_stdout(target):
        print_tokenizer_table(models)
        print_matrix(models, "top1_is_answer_letter_variant_rate",
                     "2. top1 is any A/B/C/D variant (rate, per model × subset)")
        print_matrix(models, "top1_matches_used_prefix_rate",
                     "3. top1 == token id used for scoring (rate, per model × subset)")
        print_worst_cases(models, args.worst)
        print_overall(models)

        if not args.no_csv:
            write_matrix_csv(models, "top1_is_answer_letter_variant_rate",
                             resultsdir / "matrix_top1_is_letter.csv")
            write_matrix_csv(models, "top1_matches_used_prefix_rate",
                             resultsdir / "matrix_top1_matches_used.csv")

    if fh:
        fh.close()
        print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
