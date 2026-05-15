#!/usr/bin/env python3
"""
MCQ evaluation on yangzhang33/culture-eval-benchmark using log-likelihood scoring.

Instead of generating text, computes the log-probability of each answer token
(" A", " B", " C", " D") at the final prompt position and picks the highest.
This works uniformly for base and instruct models without any special formatting.

Usage:
  python run_mcq_eval_loglik.py --outdir results --models Qwen/Qwen2.5-7B --subsets english_ca chinese_ca
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

    - english_ca     -> English
    - language_ca    -> native language (must be in SUBSET_LANG_MAP)
    - language_cs    -> native language (must be in SUBSET_LANG_MAP)
    - language_cs_en -> English
    """
    if subset.endswith("_cs_en") or subset == "english_ca":
        return "en"
    for prefix, lang in SUBSET_LANG_MAP.items():
        if subset.startswith(prefix) and (subset.endswith("_ca") or subset.endswith("_cs")):
            return lang
    raise ValueError(
        f"Subset '{subset}' is not '_cs_en' or 'english_ca', and its prefix is not in "
        f"SUBSET_LANG_MAP. Add an entry or check the subset name."
    )


def culture_preamble(subset: str) -> str | None:
    """Return a culture context preamble for _cs_en subsets (used with --v3-en)."""
    if not subset.endswith("_cs_en"):
        return None
    for prefix, culture in SUBSET_CULTURE_MAP.items():
        if subset.startswith(prefix):
            return f"This is a question about {culture} culture."
    raise ValueError(
        f"Subset '{subset}' ends with '_cs_en' but its prefix is not in SUBSET_CULTURE_MAP. "
        f"Add an entry for it."
    )


def culture_preamble_native(subset: str) -> str | None:
    """Return a native-language culture context preamble for _cs subsets (used with --v3-cs)."""
    if not subset.endswith("_cs"):
        return None
    for prefix, preamble in SUBSET_CULTURE_NATIVE_MAP.items():
        if subset.startswith(prefix):
            return preamble
    raise ValueError(
        f"Subset '{subset}' ends with '_cs' but its prefix is not in SUBSET_CULTURE_NATIVE_MAP. "
        f"Add an entry for it."
    )


def build_prompt(row: dict, lang: str = "en", preamble: str | None = None) -> str:
    q_label, ans_label = PROMPT_LANG[lang]
    prefix = f"{preamble}\n" if preamble else ""
    return (
        f"{prefix}{q_label}{row['question']}\n"
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

    # Resolve answer token IDs once.
    # Try multiple prefix formats; use the first where every letter is a single token.
    option_ids = {}
    prefix_used = None
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
            option_ids = candidate
            prefix_used = prefix
            print(f"  Answer token ids (prefix={prefix!r}): { {l: option_ids[l] for l in ANSWER_LETTERS} }")
            break
    else:
        raise ValueError(
            f"No single-token representation found for answer letters for {model_id}."
        )

    accuracy = {}
    raw_records = []
    for subset in subsets:
        print(f"  Subset: {subset}", flush=True)
        ds = load_dataset(dataset, subset, split="test")
        if max_samples_per_subset is not None:
            ds = ds.select(range(min(max_samples_per_subset, len(ds))))
        lang = subset_lang(subset)
        preamble = (
            (culture_preamble(subset) if v3_en else None)
            or (culture_preamble_native(subset) if v3_cs else None)
        )
        prompts = [build_prompt(row, lang=lang, preamble=preamble) for row in ds]
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

            for b in range(len(batch)):
                scores = {l: log_probs[b, tid].item() for l, tid in option_ids.items()}
                pred = max(scores, key=scores.get)
                preds.append(pred)
                all_scores.append(scores)

        for row, prompt, scores, pred, gold_ans in zip(ds, prompts, all_scores, preds, gold):
            raw_records.append({
                "subset": subset,
                "prompt": prompt,
                "log_probs": scores,
                "extracted_answer": pred,
                "gold_answer": gold_ans,
                "correct": pred == gold_ans,
                **{k: row[k] for k in row if k != "answer"},
            })

        correct = sum(p == g for p, g in zip(preds, gold))
        acc = correct / len(gold) if gold else 0.0
        accuracy[subset] = round(acc, 4)
        print(f"    acc = {acc:.4f}  ({correct}/{len(gold)})")

    del model
    del tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    scoring_info = {"prefix_used_for_scoring": prefix_used, "option_ids_used": option_ids}
    return accuracy, raw_records, scoring_info


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def main():
    parser = argparse.ArgumentParser(description="MCQ eval via log-likelihood scoring")
    parser.add_argument("--dataset", default="yangzhang33/culture-eval-benchmark-cs-filtered-lite")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--v3-en", action="store_true", help="Prepend culture context preamble to _cs_en subset prompts")
    parser.add_argument("--v3-cs", action="store_true", help="Prepend native-language culture context preamble to _cs subset prompts")
    parser.add_argument("--models", nargs="*", default=[], metavar="MODEL")
    parser.add_argument("--subsets", nargs="+", required=True, metavar="SUBSET")
    args = parser.parse_args()

    if not args.models:
        print("No models specified, nothing to do.")
        return

    _ds = args.dataset.split("/")[-1]
    if _ds == "culture-eval-benchmark-cs-filtered-lite":
        assert any(v in args.outdir for v in ("v1", "v3")), (
            f"DATASET is '{_ds}' but --outdir '{args.outdir}' does not contain 'v1' or 'v3'."
        )
    elif _ds == "culture-eval-benchmark-cs-filtered-lite-human-filtered":
        assert "v2" in args.outdir, (
            f"DATASET is '{_ds}' but --outdir '{args.outdir}' does not contain 'v2'."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for model_id in args.models:
        accuracy, raw_records, scoring_info = evaluate_model(
            model_id, args.subsets, args.batch_size, args.dataset,
            max_samples_per_subset=args.max_samples, local_only=args.local_only,
            v3_en=args.v3_en, v3_cs=args.v3_cs,
        )
        slug = model_slug(model_id)

        # Per-model accuracy JSON (merge with existing)
        acc_path = outdir / f"{slug}_accuracy.json"
        if acc_path.exists():
            with open(acc_path, encoding="utf-8") as f:
                existing_acc = json.load(f)
            existing_acc["accuracy"].update(accuracy)
            accuracy = existing_acc["accuracy"]
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
