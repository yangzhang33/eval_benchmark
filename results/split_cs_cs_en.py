"""
Split accuracy + predictions JSONs in cs_filtered_lite_eval_loglik_v1 into two folders:
  - cs_filtered_lite_eval_loglik_v1_cs_only   : only *_cs subsets
  - cs_filtered_lite_eval_loglik_v1_cs_en_only: only *_cs_en subsets
"""

import json
from pathlib import Path

SRC_DIR = Path(__file__).parent / "cs_filtered_lite_eval_loglik_v1"
CS_DIR  = Path(__file__).parent / "cs_filtered_lite_eval_loglik_v1/cs_only"
EN_DIR  = Path(__file__).parent / "cs_filtered_lite_eval_loglik_v1/cs_en_only"

CS_DIR.mkdir(exist_ok=True)
EN_DIR.mkdir(exist_ok=True)

# ---------- accuracy ----------
accuracy_files = sorted(SRC_DIR.glob("*_accuracy.json"))
for src_path in accuracy_files:
    data  = json.loads(src_path.read_text())
    model = data["model"]
    acc   = data["accuracy"]

    cs_acc    = {k: v for k, v in acc.items() if not k.endswith("_cs_en")}
    cs_en_acc = {k: v for k, v in acc.items() if k.endswith("_cs_en")}

    (CS_DIR / src_path.name).write_text(
        json.dumps({"model": model, "accuracy": cs_acc}, ensure_ascii=False, indent=4))
    (EN_DIR / src_path.name).write_text(
        json.dumps({"model": model, "accuracy": cs_en_acc}, ensure_ascii=False, indent=4))

    print(f"[accuracy OK] {src_path.name}")

# ---------- predictions ----------
pred_files = sorted(SRC_DIR.glob("*_predictions.json"))
for src_path in pred_files:
    data  = json.loads(src_path.read_text())
    model = data["model"]
    preds = data["predictions"]

    cs_preds    = [p for p in preds if not p["subset"].endswith("_cs_en")]
    cs_en_preds = [p for p in preds if p["subset"].endswith("_cs_en")]

    (CS_DIR / src_path.name).write_text(
        json.dumps({"model": model, "predictions": cs_preds}, ensure_ascii=False, indent=4))
    (EN_DIR / src_path.name).write_text(
        json.dumps({"model": model, "predictions": cs_en_preds}, ensure_ascii=False, indent=4))

    print(f"[predictions OK] {src_path.name}")

print(f"\nDone. {len(accuracy_files)} accuracy + {len(pred_files)} predictions files processed.")
print(f"  _cs    -> {CS_DIR}")
print(f"  _cs_en -> {EN_DIR}")
