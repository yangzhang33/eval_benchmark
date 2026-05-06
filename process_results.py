"""
Unified results processing pipeline.

Steps:
  1. Collect: read *_accuracy.json files, extract selected metric(s)
  2. Filter:  drop rows where all subset scores < threshold
  3. Reorder: sort columns/rows by canonical subset/model order
  4. Gaps:    compute global/local/knowledge gaps per language

Edit the CONFIG section below to control behaviour.
"""

import os

from util.results_pipeline import add_hall_gaps, collect, compute_gaps, filter_low, reorder

# ===========================================================================
# CONFIG — edit this section
# ===========================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder containing *_accuracy.json files
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "cs_filtered_lite_eval_loglik_v1_5")

# Folder to write output CSVs into (will be created if it doesn't exist)
OUTPUT_DIR = os.path.join(RESULTS_DIR, "all_results")

# Folder containing *_accuracy.json files with only _ca keys.
# Set to None if _ca data is already included in RESULTS_DIR files.
CA_DIR = os.path.join(SCRIPT_DIR, "results", "ca_loglik_v1_5", "ca_results")

# Metric(s) to extract — any subset of:
#   "accuracy", "abstain_rate", "conf_err_rate", "cond_acc"
METRICS = ["accuracy", "abstain_rate", "conf_err_rate", "cond_acc"]

# Drop rows where every subset score is below this value (applied per metric)
FILTER_THRESHOLD = 0

# Set to True to skip the filtering step
NO_FILTER = False

# Set to True to skip gap computation
NO_GAPS = False

# ===========================================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results dir : {RESULTS_DIR}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"Metrics     : {METRICS}")

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
            df_filtered = filter_low(df, FILTER_THRESHOLD)
            filtered_path = os.path.join(OUTPUT_DIR, f"{metric}_filtered.csv")
            df_filtered.to_csv(filtered_path, index=False)
            print(f"  saved {filtered_path}")
        else:
            df_filtered = df

        # 5. Gaps (accuracy only)
        if not NO_GAPS and metric == "accuracy":
            df_gaps = compute_gaps(df_filtered)
            if len(df_gaps.columns) > 2:
                gaps_path = os.path.join(OUTPUT_DIR, f"{metric}_gaps.csv")
                df_gaps.to_csv(gaps_path, index=False)
                print(f"  saved {gaps_path}")
            else:
                print("  gaps: skipped (missing required subset columns)")

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
