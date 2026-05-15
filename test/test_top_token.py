#!/usr/bin/env python3
"""
Test: verify whether the model's top-1 predicted token after an MCQ prompt
is actually one of the answer-letter variants (A/B/C/D, with or without
leading space / newline), or some other token entirely.

For each (model, subset, sample) we record:
  - top-K predicted tokens (string + id + log-prob)
  - the log-probs of the A/B/C/D variants we score against
  - whether the top-1 token matches an answer-letter variant
  - which prefix (' ', '', '\n') was used to resolve option_ids

One JSON per model is written to test/results/<slug>.json containing the
tokenizer info plus one entry per subset (with all per-sample records).
File is rewritten after every subset finishes, so a crash mid-run still
leaves valid intermediate results.

Each subset is capped at --max-samples (default 100).
"""
import argparse
import inspect
import json
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Localized prompt templates: {lang: (question_label, answer_label)}
PROMPT_LANG = {
    "zh": ("问题：", "答案："),
    "el": ("Ερώτηση: ", "Απάντηση:"),
    "ar": ("السؤال: ", "الجواب:"),
    "hi": ("प्रश्न: ", "उत्तर:"),
    "id": ("Pertanyaan: ", "Jawaban:"),
    "ko": ("질문: ", "답변:"),
    "it": ("Domanda: ", "Risposta:"),
    "fr": ("Question: ", "Réponse:"),
    "ja": ("質問：", "回答："),
    "es": ("Pregunta: ", "Respuesta:"),
    "en": ("Question: ", "Answer:"),
    "bn": ("প্রশ্ন: ", "উত্তর:"),
    "nl": ("Vraag: ", "Antwoord:"),
    "he": ("שאלה: ", "תשובה:"),
    "ne": ("प्रश्न: ", "उत्तर:"),
    "fa": ("سؤال: ", "جواب:"),
    "pl": ("Pytanie: ", "Odpowiedź:"),
    "ru": ("Вопрос: ", "Ответ:"),
    "te": ("ప్రశ్న: ", "సమాధానం:"),
    "uk": ("Питання: ", "Відповідь:"),
}

SUBSET_LANG_MAP = {
    "chinese": "zh", "arabic": "ar", "greek": "el", "hindi": "hi",
    "indonesian": "id", "korean": "ko", "italic": "it", "french": "fr",
    "japanese": "ja", "spanish": "es", "bengali": "bn", "dutch": "nl",
    "hebrew": "he", "nepali": "ne", "persian": "fa", "polish": "pl",
    "russian": "ru", "telugu": "te", "ukrainian": "uk",
}

ANSWER_LETTERS = ["A", "B", "C", "D"]


def subset_lang(subset: str) -> str:
    if subset.endswith("_cs_en") or subset == "english_ca":
        return "en"
    for prefix, lang in SUBSET_LANG_MAP.items():
        if subset.startswith(prefix) and (subset.endswith("_ca") or subset.endswith("_cs")):
            return lang
    raise ValueError(f"Subset '{subset}' not recognized.")


def build_prompt(row: dict, lang: str = "en") -> str:
    q_label, ans_label = PROMPT_LANG[lang]
    return (
        f"{q_label}{row['question']}\n"
        f"A. {row['option_a']}\n"
        f"B. {row['option_b']}\n"
        f"C. {row['option_c']}\n"
        f"D. {row['option_d']}\n"
        f"{ans_label}"
    )


def resolve_option_ids(tok) -> tuple[dict, str]:
    """Find which prefix yields single-token A/B/C/D; return (id_map, prefix_used)."""
    for prefix in (" ", "", "\n"):
        candidate = {}
        ok = True
        for letter in ANSWER_LETTERS:
            ids = tok.encode(prefix + letter, add_special_tokens=False)
            if len(ids) != 1:
                ok = False
                break
            candidate[letter] = ids[0]
        if ok:
            return candidate, prefix
    raise ValueError("No single-token representation found for answer letters.")


def collect_all_letter_variants(tok) -> dict:
    """For each of (' ', '', '\n') prefix, try to find single-token A/B/C/D ids.
    Returns {prefix: {letter: id}} for prefixes that work."""
    out = {}
    for prefix in (" ", "", "\n"):
        candidate = {}
        ok = True
        for letter in ANSWER_LETTERS:
            ids = tok.encode(prefix + letter, add_special_tokens=False)
            if len(ids) != 1:
                ok = False
                break
            candidate[letter] = ids[0]
        if ok:
            out[prefix] = candidate
    return out


def save_model_results(model_results: dict, out_path: Path) -> None:
    """Atomic-ish write: write to .tmp then rename, so an interrupted run
    can't leave behind a half-written file."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(model_results, f, ensure_ascii=False, indent=2)
    tmp_path.replace(out_path)


@torch.inference_mode()
def evaluate_model(model_id: str, subsets: list[str], batch_size: int, dataset: str,
                   max_samples_per_subset: int, local_only: bool, top_k: int,
                   outdir: Path) -> None:
    print(f"\n=== Loading {model_id} ===", flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tok = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True, local_files_only=local_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="balanced",
        trust_remote_code=True, local_files_only=local_only,
    )

    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()

    use_num_logits_to_keep = "num_logits_to_keep" in inspect.signature(model.forward).parameters
    supports_use_cache = "use_cache" in inspect.signature(model.forward).parameters

    option_ids, prefix_used = resolve_option_ids(tok)
    all_variants = collect_all_letter_variants(tok)
    answer_letter_ids = set()
    for pref_map in all_variants.values():
        answer_letter_ids.update(pref_map.values())

    print(f"  Resolved option_ids (prefix={prefix_used!r}): {option_ids}", flush=True)
    print(f"  All letter variants: { {repr(p): v for p, v in all_variants.items()} }", flush=True)

    slug = model_id.replace("/", "__")
    out_path = outdir / f"{slug}.json"

    model_results = {
        "model": model_id,
        "prefix_used_for_scoring": prefix_used,
        "option_ids_used": option_ids,
        "all_letter_variants": {repr(p): v for p, v in all_variants.items()},
        "all_letter_variants_raw_keys": list(all_variants.keys()),
        "subsets": {},   # subset -> {lang, n_samples, ..., records: [...]}
    }
    # Write tokenizer info immediately, before any subset runs.
    save_model_results(model_results, out_path)

    for subset in subsets:
        print(f"  Subset: {subset}", flush=True)
        ds = load_dataset(dataset, subset, split="test")
        ds = ds.select(range(min(max_samples_per_subset, len(ds))))
        lang = subset_lang(subset)
        prompts = [build_prompt(row, lang=lang) for row in ds]
        gold = [row["answer"].strip().upper() for row in ds]

        records = []
        top1_is_letter_count = 0
        top1_matches_used_prefix_count = 0

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i: i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
            forward_kwargs = {"use_cache": False} if supports_use_cache else {}
            if use_num_logits_to_keep:
                last_logits = model(**enc, num_logits_to_keep=1, **forward_kwargs).logits[:, -1, :]
            else:
                last_logits = model(**enc, **forward_kwargs).logits[:, -1, :]
            log_probs = F.log_softmax(last_logits, dim=-1)

            topk_logp, topk_idx = torch.topk(log_probs, k=top_k, dim=-1)

            for b in range(len(batch)):
                top_tokens = []
                for k in range(top_k):
                    tid = int(topk_idx[b, k].item())
                    lp = float(topk_logp[b, k].item())
                    tstr = tok.decode([tid])
                    top_tokens.append({
                        "rank": k + 1,
                        "token_id": tid,
                        "token_str": tstr,
                        "token_repr": repr(tstr),
                        "log_prob": lp,
                        "prob": float(torch.exp(topk_logp[b, k]).item()),
                        "is_answer_letter_variant": tid in answer_letter_ids,
                    })

                option_logp = {l: float(log_probs[b, tid].item()) for l, tid in option_ids.items()}
                pred_used = max(option_logp, key=option_logp.get)

                variants_logp = {}
                for pref, pmap in all_variants.items():
                    variants_logp[repr(pref)] = {l: float(log_probs[b, tid].item()) for l, tid in pmap.items()}

                top1_id = int(topk_idx[b, 0].item())
                top1_is_letter = top1_id in answer_letter_ids
                top1_matches_used = top1_id in set(option_ids.values())
                if top1_is_letter:
                    top1_is_letter_count += 1
                if top1_matches_used:
                    top1_matches_used_prefix_count += 1

                records.append({
                    "prompt_tail": prompts[i + b][-80:],
                    "gold_answer": gold[i + b],
                    "top_k_tokens": top_tokens,
                    "top1_is_answer_letter_variant": top1_is_letter,
                    "top1_matches_used_prefix": top1_matches_used,
                    "pred_under_used_prefix": pred_used,
                    "option_logp_used_prefix": option_logp,
                    "option_logp_all_variants": variants_logp,
                })

        n = len(records)
        rate_letter = top1_is_letter_count / n if n else 0.0
        rate_used = top1_matches_used_prefix_count / n if n else 0.0
        print(f"    top1 is any A/B/C/D variant: {top1_is_letter_count}/{n} ({rate_letter:.2%})", flush=True)
        print(f"    top1 matches used prefix {prefix_used!r}: {top1_matches_used_prefix_count}/{n} ({rate_used:.2%})", flush=True)

        model_results["subsets"][subset] = {
            "lang": lang,
            "n_samples": n,
            "top1_is_answer_letter_variant_count": top1_is_letter_count,
            "top1_is_answer_letter_variant_rate": round(rate_letter, 4),
            "top1_matches_used_prefix_count": top1_matches_used_prefix_count,
            "top1_matches_used_prefix_rate": round(rate_used, 4),
            "records": records,
        }
        # Rewrite the per-model file after each subset, so partial progress is durable.
        save_model_results(model_results, out_path)

    print(f"  saved -> {out_path}", flush=True)

    del model
    del tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main():
    parser = argparse.ArgumentParser(description="Probe top predicted token after MCQ prompt")
    parser.add_argument("--dataset", default="yangzhang33/culture-eval-benchmark-cs-filtered-lite")
    parser.add_argument("--outdir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--models", nargs="+", required=True, metavar="MODEL")
    parser.add_argument("--subsets", nargs="+", required=True, metavar="SUBSET")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for model_id in args.models:
        try:
            evaluate_model(
                model_id, args.subsets, args.batch_size, args.dataset,
                max_samples_per_subset=args.max_samples,
                local_only=args.local_only,
                top_k=args.top_k,
                outdir=outdir,
            )
        except Exception as e:
            print(f"!! Failed on {model_id}: {e}", flush=True)
            continue

    # Final summary scan across per-model files
    flat_rows = []
    model_prefix_rows = []
    for p in sorted(outdir.glob("*.json")):
        if p.name in ("summary.json", "model_prefix_summary.json"):
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if "subsets" not in d:
            continue
        model_prefix_rows.append({
            "model": d["model"],
            "prefix_used_for_scoring": d["prefix_used_for_scoring"],
            "working_prefixes": d["all_letter_variants_raw_keys"],
            "option_ids_used": d["option_ids_used"],
        })
        for subset, s in d["subsets"].items():
            flat_rows.append({
                "model": d["model"],
                "subset": subset,
                "lang": s["lang"],
                "n": s["n_samples"],
                "prefix_used": d["prefix_used_for_scoring"],
                "top1_is_letter_rate": s["top1_is_answer_letter_variant_rate"],
                "top1_matches_used_prefix_rate": s["top1_matches_used_prefix_rate"],
            })

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(flat_rows, f, ensure_ascii=False, indent=2)
    with open(outdir / "model_prefix_summary.json", "w", encoding="utf-8") as f:
        json.dump(model_prefix_rows, f, ensure_ascii=False, indent=2)

    print(f"\n--- Per-model prefix used ---")
    print(f"{'Model':<55} {'prefix':<8} {'working prefixes':<25}")
    print("-" * 90)
    for r in model_prefix_rows:
        print(f"{r['model']:<55} {r['prefix_used_for_scoring']!r:<8} {str(r['working_prefixes']):<25}")

    print(f"\n{'Model':<55} {'Subset':<22} {'pfx':<5} {'top1∈letters':>14} {'top1=used':>12}")
    print("-" * 110)
    for r in flat_rows:
        print(f"{r['model']:<55} {r['subset']:<22} {r['prefix_used']!r:<5} "
              f"{r['top1_is_letter_rate']:>14.2%} {r['top1_matches_used_prefix_rate']:>12.2%}")


if __name__ == "__main__":
    main()
