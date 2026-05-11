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

CS_DIR    = Path(__file__).parent / "cs_filtered_lite_eval_loglik_v1_cs_only"
EN_DIR    = Path(__file__).parent / "cs_filtered_lite_eval_loglik_v1_cs_en_only"
MERGE_DIR = Path(__file__).parent / "cs_filtered_lite_eval_loglik_v1_merged"

MERGE_DIR.mkdir(exist_ok=True)

INDENT = 2  # original files use 2-space indentation

# ---------- accuracy ----------
cs_acc_files = sorted(CS_DIR.glob("*_accuracy.json"))
for cs_path in cs_acc_files:
    en_path = EN_DIR / cs_path.name
    if not en_path.exists():
        print(f"[SKIP accuracy] {cs_path.name} — no matching _cs_en file")
        continue

    cs_data = json.loads(cs_path.read_text())
    en_data = json.loads(en_path.read_text())
    indent  = INDENT

    # interleave keys: chinese_cs, chinese_cs_en, arabic_cs, arabic_cs_en, ...
    cs_acc = cs_data["accuracy"]
    en_acc = en_data["accuracy"]
    merged_acc = {}
    for cs_key in cs_acc:
        merged_acc[cs_key] = cs_acc[cs_key]
        en_key = cs_key[:-len("_cs")] + "_cs_en"
        if en_key in en_acc:
            merged_acc[en_key] = en_acc[en_key]

    merged = {"model": cs_data["model"], "accuracy": merged_acc}
    (MERGE_DIR / cs_path.name).write_text(
        json.dumps(merged, ensure_ascii=False, indent=indent))
    print(f"[accuracy OK] {cs_path.name}")

# ---------- predictions ----------
cs_pred_files = sorted(CS_DIR.glob("*_predictions.json"))
for cs_path in cs_pred_files:
    en_path = EN_DIR / cs_path.name
    if not en_path.exists():
        print(f"[SKIP predictions] {cs_path.name} — no matching _cs_en file")
        continue

    cs_data = json.loads(cs_path.read_text())
    en_data = json.loads(en_path.read_text())
    indent  = INDENT

    # group by language
    cs_by_lang = defaultdict(list)
    en_by_lang = defaultdict(list)
    for p in cs_data["predictions"]:
        cs_by_lang[p["subset"]].append(p)
    for p in en_data["predictions"]:
        base = p["subset"][:-len("_cs_en")] + "_cs"
        en_by_lang[base].append(p)

    # language order from first appearance in cs predictions
    seen = []
    for p in cs_data["predictions"]:
        if p["subset"] not in seen:
            seen.append(p["subset"])

    merged_preds = []
    for lang in seen:
        merged_preds.extend(cs_by_lang[lang])
        merged_preds.extend(en_by_lang[lang])

    merged = {"model": cs_data["model"], "predictions": merged_preds}
    (MERGE_DIR / cs_path.name).write_text(
        json.dumps(merged, ensure_ascii=False, indent=indent))
    print(f"[predictions OK] {cs_path.name}")

print(f"\nDone. Output -> {MERGE_DIR}")
