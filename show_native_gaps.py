"""
Show accuracy gaps for each model matched to its native language.

For each model, only the gap columns corresponding to its country of origin
are shown:
  China          → chinese_*
  UAE            → arabic_*
  Greece         → greek_*
  India          → hindi_*
  Southeast Asian → indonesian_*
  South Korea    → korean_*
  USA / France / Multilingual → all languages (no single native language)

Usage:
  conda run -n eval_bench python show_native_gaps.py [--gaps-csv PATH]
"""

import argparse
import os

import pandas as pd

# ---------------------------------------------------------------------------
# Country → language prefix mapping
# ---------------------------------------------------------------------------
COUNTRY_TO_LANG = {
    "China": "chinese",
    "UAE": "arabic",
    "Greece": "greek",
    "India": "hindi",
    "Southeast Asian": ["chinese", "hindi", "indonesian", "korean"],
    "South Korea": "korean",
}

DEFAULT_GAPS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results", "lite_eval_loglik_v1", "all_results", "accuracy_gaps.csv",
)


def load_gaps(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def native_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy table: model | country | global_gap | local_gap | knowledge_gap."""
    records = []
    for _, row in df.iterrows():
        country = row["country"]
        lang = COUNTRY_TO_LANG.get(country)

        if lang is None:
            # No single native language — include all language columns
            langs = list(COUNTRY_TO_LANG.values())
            langs = [l for l in langs if isinstance(l, str)]
        elif isinstance(lang, list):
            langs = lang
        else:
            langs = [lang]

        for l in langs:
            records.append({
                "model": row["model"],
                "country": country,
                "language": l,
                "global_gap": row.get(f"{l}_global_gap"),
                "local_gap": row.get(f"{l}_local_gap"),
                "knowledge_gap": row.get(f"{l}_knowledge_gap"),
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gaps-csv",
        default=DEFAULT_GAPS_CSV,
        help="Path to accuracy_gaps.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to save the output CSV",
    )
    args = parser.parse_args()

    df = load_gaps(args.gaps_csv)
    result = native_gaps(df)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.float_format", "{:+.4f}".format)
    print(result.to_string(index=False))

    if args.out:
        result.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
