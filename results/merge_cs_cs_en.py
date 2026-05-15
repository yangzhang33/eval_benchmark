"""
Merge two split folders back into one, combining _cs and _cs_en subsets.
  Input:
    - cs_filtered_lite_eval_loglik_v1_cs_only/
    - cs_filtered_lite_eval_loglik_v1_cs_en_only/
  Output:
    - cs_filtered_lite_eval_loglik_v1_merged/
"""

import json
from collections import defaultdict
from pathlib import Path

CS_DIR    = Path(__file__).parent / "cs_filtered_lite_eval_loglik_cs_v3"
EN_DIR    = Path(__file__).parent / "cs_filtered_lite_eval_loglik_v1_cs_en_only"
MERGE_DIR = Path(__file__).parent / "cs_filtered_lite_eval_loglik_cs_v3/merged"
# Set to CS_DIR, EN_DIR, or any other folder to align models and languages.
# None = no alignment (process all files in CS_DIR, include all languages).
REF_DIR   = CS_DIR

MERGE_DIR.mkdir(exist_ok=True)

INDENT = 2  # original files use 2-space indentation


def _strip(key: str) -> str:
    """Strip _cs_en or _cs suffix to get the base language name."""
    if key.endswith("_cs_en"):
        return key[:-len("_cs_en")]
    if key.endswith("_cs"):
        return key[:-len("_cs")]
    return key


# ---------- accuracy ----------
src_acc_files = sorted((REF_DIR if REF_DIR else CS_DIR).glob("*_accuracy.json"))
for ref_path in src_acc_files:
    cs_path = CS_DIR / ref_path.name
    en_path = EN_DIR / ref_path.name

    cs_data = json.loads(cs_path.read_text()) if cs_path.exists() else None
    en_data = json.loads(en_path.read_text()) if en_path.exists() else None

    if cs_data is None and en_data is None:
        print(f"[SKIP accuracy] {ref_path.name} — not found in CS_DIR or EN_DIR")
        continue

    cs_acc = cs_data["accuracy"] if cs_data else {}
    en_acc = en_data["accuracy"] if en_data else {}
    model  = (cs_data or en_data)["model"]

    # when REF_DIR is set, restrict to languages present in the ref file
    if REF_DIR:
        ref_acc     = json.loads(ref_path.read_text())["accuracy"]
        allowed     = {_strip(k) for k in ref_acc}
        cs_acc      = {k: v for k, v in cs_acc.items() if _strip(k) in allowed}
        en_acc      = {k: v for k, v in en_acc.items() if _strip(k) in allowed}

    # interleave keys: chinese_cs, chinese_cs_en, arabic_cs, arabic_cs_en, ...
    merged_acc = {}
    for cs_key, val in cs_acc.items():
        merged_acc[cs_key] = val
        en_key = _strip(cs_key) + "_cs_en"
        if en_key in en_acc:
            merged_acc[en_key] = en_acc[en_key]
    # pick up en-only entries
    covered = {_strip(k) for k in merged_acc}
    for en_key, val in en_acc.items():
        if _strip(en_key) not in covered:
            merged_acc[en_key] = val

    merged = {"model": model, "accuracy": merged_acc}
    (MERGE_DIR / ref_path.name).write_text(
        json.dumps(merged, ensure_ascii=False, indent=INDENT))
    print(f"[accuracy OK] {ref_path.name}")

# ---------- predictions ----------
src_pred_files = sorted((REF_DIR if REF_DIR else CS_DIR).glob("*_predictions.json"))
for ref_path in src_pred_files:
    cs_path = CS_DIR / ref_path.name
    en_path = EN_DIR / ref_path.name

    cs_data = json.loads(cs_path.read_text()) if cs_path.exists() else None
    en_data = json.loads(en_path.read_text()) if en_path.exists() else None

    if cs_data is None and en_data is None:
        print(f"[SKIP predictions] {ref_path.name} — not found in CS_DIR or EN_DIR")
        continue

    cs_preds = cs_data["predictions"] if cs_data else []
    en_preds = en_data["predictions"] if en_data else []
    model    = (cs_data or en_data)["model"]

    # when REF_DIR is set, restrict to languages present in the ref file
    if REF_DIR:
        ref_preds = json.loads(ref_path.read_text())["predictions"]
        allowed   = {_strip(p["subset"]) for p in ref_preds}
        cs_preds  = [p for p in cs_preds if _strip(p["subset"]) in allowed]
        en_preds  = [p for p in en_preds if _strip(p["subset"]) in allowed]

    cs_by_lang = defaultdict(list)
    en_by_lang = defaultdict(list)
    for p in cs_preds:
        cs_by_lang[p["subset"]].append(p)
    for p in en_preds:
        base = _strip(p["subset"]) + "_cs"
        en_by_lang[base].append(p)

    # language order from first appearance in cs predictions
    seen = []
    for p in cs_preds:
        if p["subset"] not in seen:
            seen.append(p["subset"])

    merged_preds = []
    for lang in seen:
        merged_preds.extend(cs_by_lang[lang])
        merged_preds.extend(en_by_lang[lang])

    merged = {"model": model, "predictions": merged_preds}
    (MERGE_DIR / ref_path.name).write_text(
        json.dumps(merged, ensure_ascii=False, indent=INDENT))
    print(f"[predictions OK] {ref_path.name}")

print(f"\nDone. Output -> {MERGE_DIR}")
