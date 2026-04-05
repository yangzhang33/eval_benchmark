#!/usr/bin/env python3
"""
MCQ evaluation on yangzhang33/culture-eval-benchmark using log-likelihood scoring.

Instead of generating text, computes the log-probability of each answer token
(" A", " B", " C", " D") at the final prompt position and picks the highest.
This works uniformly for base and instruct models without any special formatting.

Usage:
  python run_mcq_eval_loglik.py --outdir results
  python run_mcq_eval_loglik.py --models Qwen/Qwen2.5-7B --outdir results
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Mistral3ForConditionalGeneration, MistralCommonBackend

DATASET = "yangzhang33/culture-eval-benchmark-cs-filtered-lite"
# SUBSETS = [
#     "chinese_cs", "chinese_cs_en",
#     "arabic_cs", "arabic_cs_en",
#     "greek_cs", "greek_cs_en",
#     "hindi_cs", "hindi_cs_en",
#     "indonesian_cs", "indonesian_cs_en",
#     "korean_cs", "korean_cs_en",
# ]

SUBSETS = [
    "english_ca",
    "chinese_ca",
    "arabic_ca",
    "greek_ca",
    "hindi_ca",
    "indonesian_ca",
    "korean_ca",
]
MODELS = [
    #chinese models
    # "Qwen/Qwen2.5-7B",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen2.5-14B",
    # "Qwen/Qwen2.5-14B-Instruct",
    # "Qwen/Qwen2.5-32B",
    # "Qwen/Qwen2.5-32B-Instruct",
    # "deepseek-ai/deepseek-llm-7b-base",
    # "deepseek-ai/deepseek-llm-7b-chat",
    # #english models
    # "meta-llama/Meta-Llama-3.1-8B",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "google/gemma-2-9b",
    # "google/gemma-2-9b-it",
    "google/gemma-2-27b",
    "google/gemma-2-27b-it",
    "google/gemma-3-12b-pt",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-pt",
    "google/gemma-3-27b-it",
    # #greek models
    # "ilsp/Llama-Krikri-8B-Instruct",
    # "ilsp/Meltemi-7B-Instruct-v1.5",
    # # arabic models
    # # "inceptionai/jais-13b", # not done
    # # "inceptionai/jais-13b-chat", # not done
    # "inceptionai/Jais-2-8B-Chat",
    # "FreedomIntelligence/AceGPT-v2-8B",
    # "FreedomIntelligence/AceGPT-v2-8B-Chat",
    # #hindi models
    # "sarvamai/OpenHathi-7B-Hi-v0.1-Base",
    # # "krutrim-ai-labs/Krutrim-1-instruct", # not done
    # # southeast asian models
    # "aisingapore/Llama-SEA-LION-v3-8B",
    # "aisingapore/Llama-SEA-LION-v3-8B-IT",
    # "SeaLLMs/SeaLLM-7B-v2.5", # instruct
    # "SeaLLMs/SeaLLMs-v3-7B",
    # "SeaLLMs/SeaLLMs-v3-7B-Chat",
    # # korean models
    # # "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B" #instruct not done
    # "beomi/Llama-3-Open-Ko-8B",
    # "EleutherAI/polyglot-ko-12.8b",
    # "EleutherAI/polyglot-ko-5.8b",
    # # multilingual models
    # "CohereLabs/aya-expanse-8b",  #instruct
    # #mistral models
    # "mistralai/Ministral-3-8B-Base-2512",
    # "mistralai/Ministral-3-8B-Instruct-2512", # not done
]

MISTRAL3_MODELS = {
    "mistralai/Ministral-3-8B-Base-2512",
    "mistralai/Ministral-3-8B-Instruct-2512",
}

# Localized prompt templates: {lang: (question_label, answer_label)}
PROMPT_LANG = {
    "zh": ("问题：", "答案："),
    "el": ("Ερώτηση: ", "Απάντηση:"),
    "ar": ("السؤال: ", "الجواب:"),
    "hi": ("प्रश्न: ", "उत्तर:"),
    "id": ("Pertanyaan: ", "Jawaban:"),
    "ko": ("질문: ", "답변:"),
    "en": ("Question: ", "Answer:"),
}

# Maps subset prefix to prompt language (only for native-language subsets)
SUBSET_LANG_MAP = {
    "chinese": "zh",
    "arabic": "ar",
    "greek": "el",
    "hindi": "hi",
    "indonesian": "id",
    "korean": "ko",
}

ANSWER_LETTERS = ["A", "B", "C", "D"]


def subset_lang(subset: str) -> str:
    """Infer prompt language from subset name.

    - language_ca  -> native language
    - language_cs  -> native language
    - language_cs_en -> English
    """
    for prefix, lang in SUBSET_LANG_MAP.items():
        if subset.startswith(prefix) and (subset.endswith("_ca") or subset.endswith("_cs")):
            return lang
    return "en"


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


@torch.inference_mode()
def evaluate_model(model_id: str, subsets: list[str], batch_size: int, max_samples_per_subset: int | None = None, local_only: bool = False) -> tuple[dict, list]:
    """Load model, run on all subsets, return (accuracy_by_subset, raw_records)."""
    print(f"\n=== Loading {model_id} ===")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if model_id in MISTRAL3_MODELS:
        tok = MistralCommonBackend.from_pretrained(model_id, local_files_only=local_only)
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=local_only,
        )
    else:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True, local_files_only=local_only)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=local_only,
        )

    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()

    # Resolve answer token IDs once.
    # Try multiple prefix formats; use the first where every letter is a single token.
    option_ids = {}
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
        ds = load_dataset(DATASET, subset, split="test")
        if max_samples_per_subset is not None:
            ds = ds.select(range(min(max_samples_per_subset, len(ds))))
        lang = subset_lang(subset)
        prompts = [build_prompt(row, lang=lang) for row in ds]
        gold = [row["answer"].strip().upper() for row in ds]

        preds = []
        all_scores = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i: i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
            logits = model(**enc).logits          # (B, seq_len, vocab)
            last_logits = logits[:, -1, :]        # (B, vocab) — position that predicts next token
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
    torch.cuda.empty_cache()
    return accuracy, raw_records


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--outdir", default="../results/lite_eval_loglik_v1")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_samples_per_subset", type=int, default=None)
    ap.add_argument("--no-hf-download", action="store_true", default=False,
                    help="Disable HuggingFace downloads; use local cache only (errors if not cached).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    local_only = args.no_hf_download

    for model_id in args.models:
        accuracy, raw_records = evaluate_model(model_id, args.subsets, args.batch_size, args.max_samples_per_subset, local_only=local_only)
        slug = model_slug(model_id)

        # Per-model accuracy JSON
        acc_path = outdir / f"{slug}_accuracy.json"
        with open(acc_path, "w", encoding="utf-8") as f:
            json.dump({"model": model_id, "accuracy": accuracy}, f, ensure_ascii=False, indent=2)
        print(f"Saved accuracy to {acc_path}")

        # Per-model raw predictions JSON
        raw_path = outdir / f"{slug}_predictions.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump({"model": model_id, "predictions": raw_records}, f, ensure_ascii=False, indent=2)
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
