#!/usr/bin/env python3
"""
MCQ evaluation on yangzhang33/culture-eval-benchmark.

Reports accuracy per model per subset, saves results to CSV.

Usage:
  python run_mcq_eval.py --outdir results
  python run_mcq_eval.py --models Qwen/Qwen2.5-7B --outdir results
"""
import argparse
import json
import re
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

DATASET = "yangzhang33/cultural_eval_lite"
SUBSETS = [
    "english_ca",
    "chinese_ca", "chinese_cs", "chinese_cs_en",
    "arabic_ca", "arabic_cs", "arabic_cs_en",
    "greek_ca", "greek_cs", "greek_cs_en",
    "hindi_ca", "hindi_cs", "hindi_cs_en",
    "indonesian_ca", "indonesian_cs", "indonesian_cs_en",
    "korean_ca", "korean_cs", "korean_cs_en",
]
MODELS = [
    #chinese models
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-ai/deepseek-llm-7b-chat",
    #english models
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    #greek models
    "ilsp/Llama-Krikri-8B-Instruct",
    "ilsp/Meltemi-7B-Instruct-v1.5",
    #arabic models
    "inceptionai/jais-13b",
    "inceptionai/jais-13b-chat",
    "inceptionai/Jais-2-8B-Chat",
    "FreedomIntelligence/AceGPT-v2-8B",
    "FreedomIntelligence/AceGPT-v2-8B-Chat",
    #hindi models
    "sarvamai/OpenHathi-7B-Hi-v0.1-Base",
    "krutrim-ai-labs/Krutrim-1-instruct",
    #southeast asian models
    "aisingapore/Llama-SEA-LION-v3-8B",
    "aisingapore/Llama-SEA-LION-v3-8B-IT",
    "SeaLLMs/SeaLLM-7B-v2.5", # instruct
    "SeaLLMs/SeaLLMs-v3-7B",
    "SeaLLMs/SeaLLMs-v3-7B-Chat",
    #korean models
    "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B" #instruct
    "beomi/Llama-3-Open-Ko-8B",
    "EleutherAI/polyglot-ko-12.8b",
    "EleutherAI/polyglot-ko-5.8b",
    #multilingual models
    "CohereLabs/aya-expanse-8b"  #instruct
]


INSTRUCT_KEYWORDS = ("instruct", "-it", "-chat", "-chat-hf")

# Models whose names don't contain instruct keywords but are instruct/chat models
# (annotated with #instruct in the MODELS list above)
INSTRUCT_MODEL_OVERRIDES = {
    "SeaLLMs/SeaLLM-7B-v2.5",
    "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B",
    "CohereLabs/aya-expanse-8b",
}

# Localized prompt templates: {lang: (question_label, instruction, answer_label)}
PROMPT_LANG = {
    "zh": (
        "问题：",
        "仅用字母（A、B、C 或 D）作答。",
        "答案：",
    ),
    "el": (
        "Ερώτηση: ",
        "Απαντήστε μόνο με το γράμμα (A, B, C ή D).",
        "Απάντηση:",
    ),
    "ar": (
        "السؤال: ",
        "أجب بالحرف فقط (A أو B أو C أو D).",
        "الجواب:",
    ),
    "hi": (
        "प्रश्न: ",
        "केवल अक्षर (A, B, C, या D) से उत्तर दें।",
        "उत्तर:",
    ),
    "id": (
        "Pertanyaan: ",
        "Jawab hanya dengan huruf (A, B, C, atau D).",
        "Jawaban:",
    ),
    "ko": (
        "질문: ",
        "알파벳(A, B, C 또는 D)으로만 답하세요.",
        "답변:",
    ),
    "en": (
        "Question: ",
        "Answer with only the letter (A, B, C, or D).",
        "Answer:",
    ),
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


def subset_lang(subset: str) -> str:
    """Infer prompt language from subset name."""
    if not subset.endswith("_en"):
        for prefix, lang in SUBSET_LANG_MAP.items():
            if subset.startswith(prefix) and subset.endswith("_cs"):
                return lang
    return "en"


def is_instruct_model(model_id: str) -> bool:
    return model_id in INSTRUCT_MODEL_OVERRIDES or any(kw in model_id.lower() for kw in INSTRUCT_KEYWORDS)


def build_prompt(row: dict, lang: str = "en", tok=None, instruct: bool = False) -> str:
    q_label, instruction, ans_label = PROMPT_LANG[lang]
    text = (
        f"{q_label}{row['question']}\n"
        f"A. {row['option_a']}\n"
        f"B. {row['option_b']}\n"
        f"C. {row['option_c']}\n"
        f"D. {row['option_d']}\n"
        f"{instruction}\n{ans_label}"
    )
    if instruct and tok is not None:
        messages = [{"role": "user", "content": text}]
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def extract_answer(text: str) -> str:
    """Return the first A/B/C/D found in the generated text, or '' if none."""
    text = text.strip()
    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1)
    # fallback: first character if it's a valid option
    if text and text[0] in "ABCD":
        return text[0]
    return ""


@torch.inference_mode()
def evaluate_model(model_id: str, subsets: list[str], batch_size: int, max_new_tokens: int, max_samples_per_subset: int | None = None, local_only: bool = False) -> tuple[dict, list]:
    """Load model, run on all subsets, return (accuracy_by_subset, raw_records)."""
    print(f"\n=== Loading {model_id} ===")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True, local_files_only=local_only)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_only,
    )
    model.eval()

    instruct = is_instruct_model(model_id)
    print(f"  Instruct model: {instruct}")

    accuracy = {}
    raw_records = []
    for subset in subsets:
        print(f"  Subset: {subset}", flush=True)
        ds = load_dataset(DATASET, subset, split="test", local_files_only=local_only)
        if max_samples_per_subset is not None:
            ds = ds.select(range(min(max_samples_per_subset, len(ds))))
        lang = subset_lang(subset)
        prompts = [build_prompt(row, lang=lang, tok=tok, instruct=instruct) for row in ds]
        gold = [row["answer"].strip().upper() for row in ds]

        raw_outputs = []
        preds = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i: i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            gen_ids = out[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
            raw_outputs.extend(texts)
            preds.extend(extract_answer(t) for t in texts)

        for row, prompt, raw_out, pred, gold_ans in zip(ds, prompts, raw_outputs, preds, gold):
            raw_records.append({
                "subset": subset,
                "prompt": prompt,
                "model_output": raw_out,
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
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--max_samples_per_subset", type=int, default=None)
    ap.add_argument("--no-hf-download", action="store_true", default=False,
                    help="Disable HuggingFace downloads; use local cache only (errors if not cached).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    local_only = args.no_hf_download
    local_only = True
    if local_only:
        print("HF downloads disabled — using local cache only.")

    for model_id in args.models:
        accuracy, raw_records = evaluate_model(model_id, args.subsets, args.batch_size, args.max_new_tokens, args.max_samples_per_subset, local_only=local_only)
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
