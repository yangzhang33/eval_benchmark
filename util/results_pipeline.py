import glob
import json
import os

import pandas as pd

from util.results_constants import GAP_LANGUAGES, MODEL_COUNTRY, MODELS, SUBSETS


def collect(results_dir: str, metric: str) -> pd.DataFrame:
    """Read all *_accuracy.json files and extract `metric` into a DataFrame."""
    json_files = sorted(glob.glob(os.path.join(results_dir, "*_accuracy.json")))
    if not json_files:
        raise FileNotFoundError(f"No *_accuracy.json files found in {results_dir}")

    rows = []
    all_keys: list[str] = []
    for filepath in json_files:
        with open(filepath) as f:
            data = json.load(f)
        model = data["model"]
        values = data.get(metric, {})
        for k in values:
            if k not in all_keys:
                all_keys.append(k)
        rows.append((model, values))

    ordered_keys = [k for k in SUBSETS if k in all_keys]
    ordered_keys += [k for k in sorted(all_keys) if k not in ordered_keys]

    records = []
    for model, values in rows:
        rec = {"model": model, "country": MODEL_COUNTRY.get(model, "Unknown")}
        for k in ordered_keys:
            rec[k] = values.get(k, "")
        records.append(rec)

    return pd.DataFrame(records)


def reorder(df: pd.DataFrame) -> pd.DataFrame:
    """Sort columns by canonical subset order and rows by canonical model order."""
    subset_cols = [s for s in SUBSETS if s in df.columns]
    other_cols = [c for c in df.columns if c not in ("model", "country") and c not in subset_cols]
    df = df[["model", "country"] + subset_cols + other_cols]

    df = df.copy()
    df["model"] = pd.Categorical(df["model"], categories=MODELS, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)
    df["model"] = df["model"].astype(str)
    return df


def filter_low(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop rows where every subset score is below `threshold`."""
    score_cols = [c for c in df.columns if c not in ("model", "country")]
    numeric = df[score_cols].apply(pd.to_numeric, errors="coerce")
    mask = (numeric >= threshold).any(axis=1)
    removed = (~mask).sum()
    print(f"  filter: removed {removed} row(s) with all scores < {threshold}, kept {mask.sum()}")
    return df[mask].reset_index(drop=True)


def add_hall_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Add HallGap columns (cs_en - cs) per language to a conf_err_rate DataFrame."""
    df = df.copy()
    for lang in GAP_LANGUAGES:
        cs_col, cs_en_col = f"{lang}_cs", f"{lang}_cs_en"
        if cs_col in df.columns and cs_en_col in df.columns:
            df[f"{lang}_hall_gap"] = (df[cs_en_col] - df[cs_col]).round(4)
    return df


def compute_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Compute global/local/knowledge gaps for each language."""
    rows = []
    for _, row in df.iterrows():
        entry = {"model": row["model"], "country": row["country"]}
        for lang in GAP_LANGUAGES:
            ca_col, cs_col, cs_en_col = f"{lang}_ca", f"{lang}_cs", f"{lang}_cs_en"
            if not all(c in df.columns for c in (ca_col, cs_col, cs_en_col, "english_ca")):
                continue
            global_gap = row[ca_col] - row["english_ca"]
            local_gap = row[cs_col] - row[cs_en_col]
            knowledge_gap = local_gap - global_gap
            entry[f"{lang}_global_gap"] = round(global_gap, 4)
            entry[f"{lang}_local_gap"] = round(local_gap, 4)
            entry[f"{lang}_knowledge_gap"] = round(knowledge_gap, 4)
        rows.append(entry)
    return pd.DataFrame(rows)
