#!/usr/bin/env python3
"""
MCQ evaluation on yangzhang33/CEB_code_switched using log-likelihood scoring.

Instead of generating text, computes the log-probability of each answer token
(" A", " B", " C", " D") at the final prompt position and picks the highest.
This works uniformly for base and instruct models without any special formatting.

Subset naming:
  - {language}_cs_mix  -> uses `question_cs_mix`, native-language prompt labels
  - {language}_en_mix  -> uses `question_en_mix`, English prompt labels

Usage:
  python run_mcq_eval_loglik_code_switched.py --outdir results --models Qwen/Qwen2.5-7B --subsets chinese_cs_mix chinese_en_mix
"""
import argparse
import inspect
import json
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Mistral3ForConditionalGeneration, MistralCommonBackend

MISTRAL3_MODELS = {
    "mistralai/Ministral-3-8B-Base-2512",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Base-2512",
    "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Instruct-2512",
}

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

# Maps subset prefix to prompt language (only for native-language subsets)
SUBSET_LANG_MAP = {
    "chinese": "zh",
    "arabic": "ar",
    "greek": "el",
    "hindi": "hi",
    "indonesian": "id",
    "korean": "ko",
    "italic": "it",
    "french": "fr",
    "japanese": "ja",
    "spanish": "es",
    "bengali": "bn",
    "dutch": "nl",
    "hebrew": "he",
    "nepali": "ne",
    "persian": "fa",
    "polish": "pl",
    "russian": "ru",
    "telugu": "te",
    "ukrainian": "uk",
}

# Maps subset prefix to English culture name (used by --v3-en preamble)
SUBSET_CULTURE_MAP = {
    "chinese": "Chinese",
    "arabic": "Arabic",
    "greek": "Greek",
    "hindi": "Hindi",
    "indonesian": "Indonesian",
    "korean": "Korean",
    "italic": "Italian",
    "french": "French",
    "japanese": "Japanese",
    "spanish": "Spanish",
    "bengali": "Bengali",
    "dutch": "Dutch",
    "hebrew": "Hebrew",
    "nepali": "Nepali",
    "persian": "Persian",
    "polish": "Polish",
    "russian": "Russian",
    "telugu": "Telugu",
    "ukrainian": "Ukrainian",
}

# Native-language culture preambles (used by --v3-cs preamble)
SUBSET_CULTURE_NATIVE_MAP = {
    "chinese":    "这是一个关于中国文化的问题。",
    "arabic":     "هذا سؤال عن الثقافة العربية.",
    "greek":      "Αυτή είναι μια ερώτηση σχετικά με την ελληνική κουλτούρα.",
    "hindi":      "यह भारतीय संस्कृति के बारे में एक प्रश्न है।",
    "indonesian": "Ini adalah pertanyaan tentang budaya Indonesia.",
    "korean":     "이것은 한국 문화에 관한 질문입니다.",
    "italic":     "Questa è una domanda sulla cultura italiana.",
    "french":     "C'est une question sur la culture française.",
    "japanese":   "これは日本文化に関する問題です。",
    "spanish":    "Esta es una pregunta sobre la cultura española.",
    "bengali":    "এটি বাংলা সংস্কৃতি সম্পর্কে একটি প্রশ্ন।",
    "dutch":      "Dit is een vraag over de Nederlandse cultuur.",
    "hebrew":     "זוהי שאלה על התרבות העברית.",
    "nepali":     "यो नेपाली संस्कृतिको बारेमा एउटा प्रश्न हो।",
    "persian":    "این یک سؤال درباره فرهنگ ایرانی است.",
    "polish":     "To jest pytanie o kulturę polską.",
    "russian":    "Это вопрос о русской культуре.",
    "telugu":     "ఇది తెలుగు సంస్కృతి గురించిన ప్రశ్న.",
    "ukrainian":  "Це питання про українську культуру.",
}

ANSWER_LETTERS = ["A", "B", "C", "D"]


def subset_lang(subset: str) -> str:
    """Infer prompt language from subset name.

    - {language}_en_mix -> English
    - {language}_cs_mix -> native language (must be in SUBSET_LANG_MAP)
    """
    if subset.endswith("_en_mix"):
        return "en"
    if subset.endswith("_cs_mix"):
        for prefix, lang in SUBSET_LANG_MAP.items():
            if subset.startswith(prefix):
                return lang
    raise ValueError(
        f"Subset '{subset}' does not end with '_cs_mix' or '_en_mix', or its prefix is not in "
        f"SUBSET_LANG_MAP. Add an entry or check the subset name."
    )


def subset_question_field(subset: str) -> str:
    """Return the dataset column name that holds the question for this subset."""
    if subset.endswith("_cs_mix"):
        return "question_cs_mix"
    if subset.endswith("_en_mix"):
        return "question_en_mix"
    raise ValueError(
        f"Subset '{subset}' does not end with '_cs_mix' or '_en_mix'."
    )


def culture_preamble(subset: str) -> str | None:
    """Return a culture context preamble for _en_mix subsets (used with --v3-en)."""
    if not subset.endswith("_en_mix"):
        return None
    for prefix, culture in SUBSET_CULTURE_MAP.items():
        if subset.startswith(prefix):
            return f"This is a question about {culture} culture."
    raise ValueError(
        f"Subset '{subset}' ends with '_en_mix' but its prefix is not in SUBSET_CULTURE_MAP. "
        f"Add an entry for it."
    )


def culture_preamble_native(subset: str) -> str | None:
    """Return a native-language culture context preamble for _cs_mix subsets (used with --v3-cs)."""
    if not subset.endswith("_cs_mix"):
        return None
    for prefix, preamble in SUBSET_CULTURE_NATIVE_MAP.items():
        if subset.startswith(prefix):
            return preamble
    raise ValueError(
        f"Subset '{subset}' ends with '_cs_mix' but its prefix is not in SUBSET_CULTURE_NATIVE_MAP. "
        f"Add an entry for it."
    )


def build_prompt(row: dict, lang: str, question_field: str, preamble: str | None = None) -> str:
    q_label, ans_label = PROMPT_LANG[lang]
    prefix = f"{preamble}\n" if preamble else ""
    return (
        f"{prefix}{q_label}{row[question_field]}\n"
        f"A. {row['option_a']}\n"
        f"B. {row['option_b']}\n"
        f"C. {row['option_c']}\n"
        f"D. {row['option_d']}\n"
        f"{ans_label}"
    )


@torch.inference_mode()
def evaluate_model(model_id: str, subsets: list[str], batch_size: int, dataset: str, max_samples_per_subset: int | None = None, local_only: bool = False, v3_en: bool = False, v3_cs: bool = False) -> tuple[dict, list, dict]:
    """Load model, run on all subsets, return (accuracy_by_subset, raw_records, scoring_info)."""
    print(f"\n=== Loading {model_id} ===")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if model_id in MISTRAL3_MODELS:
        tok = MistralCommonBackend.from_pretrained(model_id, local_files_only=local_only)
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="balanced",
            local_files_only=local_only,
        )
    else:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True, local_files_only=local_only)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="balanced",
            trust_remote_code=True,
            local_files_only=local_only,
        )

    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()

    # gemma-3-27b has vocab_size=262,144; the full (B, seq_len, vocab) logits
    # tensor can be ~512 MB per forward pass and causes OOM.  When the model
    # supports num_logits_to_keep (transformers ≥4.48) we ask it to only
    # materialise logits for the final token position, dropping memory to <1 MB.
    use_num_logits_to_keep = "num_logits_to_keep" in inspect.signature(model.forward).parameters
    supports_use_cache = "use_cache" in inspect.signature(model.forward).parameters

    # Resolve answer token IDs for each prefix variant that works.
    # A prefix "works" if every letter A/B/C/D encodes to a single token under it.
    # Models where both " A" and "A" are single tokens get 8 variants; others get 4.
    option_ids_by_prefix: dict[str, dict[str, int]] = {}
    for prefix in (" ", ""):
        candidate = {}
        ok = True
        for letter in ANSWER_LETTERS:
            ids = tok.encode(prefix + letter, add_special_tokens=False)
            if len(ids) != 1:
                ok = False
                break
            candidate[letter] = ids[0]
        if ok:
            option_ids_by_prefix[prefix] = candidate

    if not option_ids_by_prefix:
        raise ValueError(
            f"No single-token representation found for answer letters for {model_id}."
        )

    # Flat list of (prefix, letter, token_id) — up to 8 entries.
    option_variants = [
        (prefix, letter, tid)
        for prefix, ids in option_ids_by_prefix.items()
        for letter, tid in ids.items()
    ]
    print(f"  Working prefixes: {list(option_ids_by_prefix.keys())}  "
          f"({len(option_variants)} answer tokens)")
    for prefix, ids in option_ids_by_prefix.items():
        print(f"    prefix={prefix!r}: { {l: ids[l] for l in ANSWER_LETTERS} }")

    accuracy = {}
    top1_rates = {}
    top5_rates = {}
    raw_records = []
    for subset in subsets:
        print(f"  Subset: {subset}", flush=True)
        ds = load_dataset(dataset, subset, split="test")
        if max_samples_per_subset is not None:
            ds = ds.select(range(min(max_samples_per_subset, len(ds))))
        lang = subset_lang(subset)
        q_field = subset_question_field(subset)
        preamble = (
            (culture_preamble(subset) if v3_en else None)
            or (culture_preamble_native(subset) if v3_cs else None)
        )
        prompts = [build_prompt(row, lang=lang, question_field=q_field, preamble=preamble) for row in ds]
        gold = [row["answer"].strip().upper() for row in ds]

        preds = []
        all_scores = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i: i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
            forward_kwargs = {"use_cache": False} if supports_use_cache else {}
            if use_num_logits_to_keep:
                last_logits = model(**enc, num_logits_to_keep=1, **forward_kwargs).logits[:, -1, :]
            else:
                last_logits = model(**enc, **forward_kwargs).logits[:, -1, :]
            log_probs = F.log_softmax(last_logits, dim=-1)  # (B, vocab)
            top5 = torch.topk(log_probs, k=5, dim=-1).indices  # (B, 5)

            for b in range(len(batch)):
                # (prefix, letter, tid, lp) for every variant
                variant_data = [
                    (prefix, letter, tid, log_probs[b, tid].item())
                    for prefix, letter, tid in option_variants
                ]
                # Nested dict for JSON output
                variant_logps: dict[str, dict[str, float]] = {}
                for prefix, letter, _, lp in variant_data:
                    variant_logps.setdefault(prefix, {})[letter] = lp
                # argmax over all variants -> picks both prefix and letter
                chosen_prefix, pred_letter, chosen_tid, _ = max(variant_data, key=lambda v: v[3])
                top5_ids = [int(t) for t in top5[b].tolist()]
                preds.append(pred_letter)
                all_scores.append({
                    "log_probs": variant_logps,
                    "argmax_prefix": chosen_prefix,
                    "argmax_token_id": int(chosen_tid),
                    "argmax_is_top1": top5_ids[0] == int(chosen_tid),
                    "argmax_is_top5": int(chosen_tid) in top5_ids,
                })

        for row, prompt, info, pred, gold_ans in zip(ds, prompts, all_scores, preds, gold):
            raw_records.append({
                "subset": subset,
                "prompt": prompt,
                **info,
                "extracted_answer": pred,
                "gold_answer": gold_ans,
                "correct": pred == gold_ans,
                **{k: row[k] for k in row if k != "answer"},
            })

        correct = sum(p == g for p, g in zip(preds, gold))
        acc = correct / len(gold) if gold else 0.0
        n = len(gold) or 1
        top1_rate = sum(s["argmax_is_top1"] for s in all_scores) / n
        top5_rate = sum(s["argmax_is_top5"] for s in all_scores) / n
        accuracy[subset] = round(acc, 4)
        top1_rates[subset] = round(top1_rate, 4)
        top5_rates[subset] = round(top5_rate, 4)
        print(f"    acc = {acc:.4f}  argmax_is_top1 = {top1_rate:.2%}  "
              f"argmax_is_top5 = {top5_rate:.2%}  ({correct}/{len(gold)})")

    del model
    del tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    scoring_info = {
        "option_ids_by_prefix": option_ids_by_prefix,
        "argmax_is_top1_rate": top1_rates,
        "argmax_is_top5_rate": top5_rates,
    }
    return accuracy, raw_records, scoring_info


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def main():
    parser = argparse.ArgumentParser(description="MCQ eval via log-likelihood scoring (code-switched dataset)")
    parser.add_argument("--dataset", default="yangzhang33/CEB_code_switched")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--v3-en", action="store_true", help="Prepend culture context preamble to _en_mix subset prompts")
    parser.add_argument("--v3-cs", action="store_true", help="Prepend native-language culture context preamble to _cs_mix subset prompts")
    parser.add_argument("--models", nargs="*", default=[], metavar="MODEL")
    parser.add_argument("--subsets", nargs="+", required=True, metavar="SUBSET")
    args = parser.parse_args()

    if not args.models:
        print("No models specified, nothing to do.")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for model_id in args.models:
        accuracy, raw_records, scoring_info = evaluate_model(
            model_id, args.subsets, args.batch_size, args.dataset,
            max_samples_per_subset=args.max_samples, local_only=args.local_only,
            v3_en=args.v3_en, v3_cs=args.v3_cs,
        )
        slug = model_slug(model_id)

        # Per-model accuracy JSON (merge per-subset dicts with existing)
        acc_path = outdir / f"{slug}_accuracy.json"
        if acc_path.exists():
            with open(acc_path, encoding="utf-8") as f:
                existing_acc = json.load(f)
            existing_acc["accuracy"].update(accuracy)
            accuracy = existing_acc["accuracy"]
            for key in ("argmax_is_top1_rate", "argmax_is_top5_rate"):
                if key in existing_acc:
                    merged = dict(existing_acc[key])
                    merged.update(scoring_info[key])
                    scoring_info[key] = merged
        with open(acc_path, "w", encoding="utf-8") as f:
            json.dump({"model": model_id, "accuracy": accuracy, **scoring_info},
                      f, ensure_ascii=False, indent=2)
        print(f"Saved accuracy to {acc_path}")

        # Per-model raw predictions JSON (merge with existing, dedup by subset)
        raw_path = outdir / f"{slug}_predictions.json"
        if raw_path.exists():
            with open(raw_path, encoding="utf-8") as f:
                existing_raw = json.load(f)
            new_subsets = {r["subset"] for r in raw_records}
            kept = [r for r in existing_raw["predictions"] if r["subset"] not in new_subsets]
            raw_records = kept + raw_records
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump({"model": model_id, **scoring_info, "predictions": raw_records},
                      f, ensure_ascii=False, indent=2)
        print(f"Saved predictions to {raw_path}")

    # Print table
    print(f"\n{'Model':<45} {'Subset':<20} {'Acc':>6}")
    print("-" * 75)
    for model_id in args.models:
        slug = model_slug(model_id)
        acc_path = outdir / f"{slug}_accuracy.json"
        with open(acc_path, encoding="utf-8") as f:
            data = json.load(f)
        for subset, acc in data["accuracy"].items():
            print(f"{model_id:<45} {subset:<20} {acc:>6.4f}")


if __name__ == "__main__":
    main()
