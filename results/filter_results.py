import pandas as pd

INPUT_CSV = "lite_eval_results_loglik_v1_5.csv"
OUTPUT_CSV = "lite_eval_results_loglik_v1_5_filtered.csv"
THRESHOLD = 0.25


def filter_low_scores(input_csv: str, output_csv: str, threshold: float = THRESHOLD) -> None:
    df = pd.read_csv(input_csv)

    score_cols = [c for c in df.columns if c not in ("model", "country")]
    mask = (df[score_cols] >= threshold).any(axis=1)
    filtered = df[mask]

    removed = len(df) - len(filtered)
    print(f"Removed {removed} row(s) where all scores < {threshold}")
    print(f"Kept {len(filtered)} row(s)")

    filtered.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")


if __name__ == "__main__":
    filter_low_scores(INPUT_CSV, OUTPUT_CSV)
