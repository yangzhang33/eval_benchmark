"""
Unified results processing pipeline, split into two independent parts.

  Part 1 — Generate CSVs:
    Read *_accuracy.json files, reorder, and save one CSV per metric.
    Also produces accuracy_filtered.csv if filtering is enabled.

  Part 2 — Process CSVs:
    Read the generated CSVs and compute gaps; write results to PROCESS_OUTPUT_DIR.

Toggle RUN_GENERATE / RUN_PROCESS in the CONFIG section to run either or both parts.
"""

import os

import pandas as pd

from util.results_pipeline import add_hall_gaps, collect, compute_gaps, filter_low, reorder

# ===========================================================================
# CONFIG — edit this section
# ===========================================================================

# ---------------------------------------------------------------------------
# Part 1 — Generate CSVs
# ---------------------------------------------------------------------------

RUN_GENERATE = True

# Folder containing *_accuracy.json files
RESULTS_DIR = "results/cs_filtered_lite_eval_loglik_v1"

# Folder to write output CSVs into (will be created if it doesn't exist)
OUTPUT_DIR = "results/cs_filtered_lite_eval_loglik_v1/all_results"

# Folder containing *_accuracy.json files with only _ca keys.
# Set to None if _ca data is already included in RESULTS_DIR files.
CA_DIR = "results/ca_loglik_v1/ca_results"

# Metric(s) to extract — any subset of:
#   "accuracy", "abstain_rate", "conf_err_rate", "cond_acc"
METRICS = ["accuracy", "abstain_rate", "conf_err_rate", "cond_acc"]

# Drop rows where every subset score is below this value (applied per metric)
FILTER_THRESHOLD = 0

# Set to True to skip the filtering step
NO_FILTER = False

# ---------------------------------------------------------------------------
# Part 2 — Process CSVs
# ---------------------------------------------------------------------------

RUN_PROCESS = True

# Direct path to the input accuracy CSV; its parent folder is used as input/output dir.
PROCESS_INPUT_CSV = "results/cs_filtered_lite_eval_loglik_v1/all_results/accuracy.csv"

PROCESS_INPUT_DIR = os.path.dirname(PROCESS_INPUT_CSV)

# Folder to write computed results into (will be created if it doesn't exist)
PROCESS_OUTPUT_DIR = PROCESS_INPUT_DIR

# Set to True to skip gap computation
NO_GAPS = False

# ===========================================================================


def run_generate():
    """Part 1: read JSON files and produce one CSV per metric."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[generate] Results dir : {RESULTS_DIR}")
    print(f"[generate] Output dir  : {OUTPUT_DIR}")
    print(f"[generate] Metrics     : {METRICS}")

    for metric in METRICS:
        print(f"\n--- {metric} ---")

        # 1. Collect
        df = collect(RESULTS_DIR, metric, ca_dir=CA_DIR if os.path.isdir(CA_DIR or "") else None)
        print(f"  collected {len(df)} models, {len(df.columns) - 2} subsets")

        # 2. Reorder
        df = reorder(df)

        # 3. Add HallGap columns for conf_err_rate, then save
        if metric == "conf_err_rate":
            df = add_hall_gaps(df)
        raw_path = os.path.join(OUTPUT_DIR, f"{metric}.csv")
        df.to_csv(raw_path, index=False)
        print(f"  saved {raw_path}")

        # 4. Filter (accuracy only)
        if not NO_FILTER and metric == "accuracy":
            df_filtered = filter_low(pd.read_csv(raw_path), FILTER_THRESHOLD)
            filtered_path = os.path.join(OUTPUT_DIR, f"{metric}_filtered.csv")
            df_filtered.to_csv(filtered_path, index=False)
            print(f"  saved {filtered_path}")


def run_process():
    """Part 2: read generated CSVs and compute gaps."""
    os.makedirs(PROCESS_OUTPUT_DIR, exist_ok=True)
    print(f"[process] Input dir  : {PROCESS_INPUT_DIR}")
    print(f"[process] Output dir : {PROCESS_OUTPUT_DIR}")

    if NO_GAPS:
        print("  gaps: skipped (NO_GAPS=True)")
        return

    # Determine which accuracy CSV to use
    filtered_path = os.path.join(PROCESS_INPUT_DIR, "accuracy_filtered.csv")
    raw_path = PROCESS_INPUT_CSV
    if not NO_FILTER and os.path.isfile(filtered_path):
        input_path = filtered_path
    elif os.path.isfile(raw_path):
        input_path = raw_path
    else:
        print(f"  gaps: skipped (no accuracy CSV found in {PROCESS_INPUT_DIR})")
        return

    print(f"\n--- accuracy gaps ---")
    df_gaps = compute_gaps(pd.read_csv(input_path))
    if len(df_gaps.columns) > 2:
        gaps_path = os.path.join(PROCESS_OUTPUT_DIR, "accuracy_gaps.csv")
        df_gaps.to_csv(gaps_path, index=False)
        print(f"  saved {gaps_path}")
    else:
        print("  gaps: skipped (missing required subset columns)")


def main():
    if RUN_GENERATE:
        run_generate()
    if RUN_PROCESS:
        run_process()
    print("\nDone.")


if __name__ == "__main__":
    main()


# ===========================================================================
# Metric definitions
# ===========================================================================
#
# accuracy
#   Fraction of questions answered correctly out of all questions.
#   accuracy = correct / total
#
# abstain_rate
#   Fraction of questions where the model refused to answer (no valid option
#   selected).
#   abstain_rate = abstained / total
#
# conf_err_rate  (confident error rate)
#   Fraction of questions where the model gave a wrong answer without
#   abstaining, i.e. confidently wrong.
#   conf_err_rate = wrong_and_not_abstained / total
#                 = (1 - accuracy - abstain_rate)
#
# hall_gap  (hallucination gap, per language)
#   Difference in confident error rate between the cross-lingual subset
#   (questions in English about the foreign culture) and the native-language
#   subset (questions in the target language about the foreign culture).
#   A positive value means the model hallucinates more when answering in
#   English than in the native language.
#   hall_gap = conf_err_rate_cs_en - conf_err_rate_cs
