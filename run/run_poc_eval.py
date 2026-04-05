#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PoC: Language × Locality evaluation (bilingual 60 QA, 120 prompts)

- Loads a bilingual QA set (JSONL)
- Runs one or more HF models (Transformers)
- Saves raw generations + simple metrics (exact match + "hallucination-like" heuristic)

Usage:
  python run_poc_eval.py \
    --data data/poc_60qa_bilingual.jsonl \
    --models Qwen/Qwen2.5-7B meta-llama/Meta-Llama-3.1-8B \
    --outdir data/results \
    --device auto \
    --batch_size 4 \
    --max_new_tokens 32

Notes:
- This is a minimal-cost PoC. Hallucination-like is a heuristic:
  if the answer is incorrect AND it is not an explicit "I don't know"/"不知道" refusal, mark as hallucination_like=1.
- For a stronger hallucination measure, add reference-grounded verification later.
"""
import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


IDK_PAT = re.compile(r"\b(i\s*don't\s*know|i\s*do\s*not\s*know|not\s*sure|cannot\s*answer|can't\s*answer)\b", re.I)
IDK_ZH_PAT = re.compile(r"(不知道|不确定|无法确定|无法回答|不能确定)")


def normalize_text(s: str) -> str:
    s = s.strip()
    # remove surrounding quotes / punctuation
    s = s.strip(" \t\r\n\"'“”‘’.,;:!?，。；：！？")
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def extract_short_answer(generation: str) -> str:
    """
    Heuristic: take the first line / first sentence-like chunk.
    """
    g = generation.strip()
    # remove leading assistant markers
    g = re.sub(r"^(assistant|answer)\s*[:：]\s*", "", g, flags=re.I).strip()
    # take first line
    first = g.splitlines()[0].strip() if g else ""
    # cut after first sentence end if very long
    first = re.split(r"[。！？.!?]", first)[0].strip()
    return normalize_text(first)


def is_idk(ans: str) -> bool:
    a = ans.strip()
    if not a:
        return True
    return bool(IDK_PAT.search(a) or IDK_ZH_PAT.search(a) or a.lower() in {"idk", "unknown"})


def match_any(ans: str, golds: List[str], answer_type: str) -> bool:
    """
    Minimal matching:
    - Normalize both.
    - For 'set_any': accept if ans contains any gold token.
    - For 'pair': accept if both entities appear (order-insensitive) for a subset of provided gold patterns.
    - Otherwise: exact match or substring match (safe for short answers).
    """
    a = normalize_text(ans).lower()
    golds_norm = [normalize_text(g).lower() for g in golds]

    if answer_type == "set_any":
        # accept if answer contains any gold token (normalized)
        for g in golds_norm:
            if g and (g in a or a in g):
                return True
        return False

    if answer_type == "pair":
        # accept if answer matches any provided pair pattern (already encoded) OR contains two country names separated by conjunction/comma
        for g in golds_norm:
            if g and (a == g or g in a or a in g):
                return True
        # loose: if it contains "china" and "nepal" etc. (works for Everest)
        if ("china" in a and "nepal" in a) or ("中国" in ans and "尼泊尔" in ans):
            return True
        return False

    # default: exact or substring
    for g in golds_norm:
        if not g:
            continue
        if a == g:
            return True
        # allow simple substring containment for variants like "october 1"
        if (g in a) or (a in g):
            return True
    return False


def build_prompt(lang: str, question: str) -> str:
    if lang == "zh":
        instr = "请用简短短语回答。如果不确定，请回答“不知道”，不要猜测。"
        return f"{instr}\n问题：{question}\n答案："
    else:
        instr = "Answer with a short phrase. If you are not sure, say \"I don't know.\" Do not guess."
        return f"{instr}\nQuestion: {question}\nAnswer:"


@dataclass
class Example:
    qid: str
    category: str
    lang: str
    question: str
    answers: List[str]
    answer_type: str


def load_jsonl(path: str) -> List[Example]:
    items: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            items.append(Example(
                qid=row["id"],
                category=row["category"],
                lang=row["lang"],
                question=row["question"],
                answers=row["answers"],
                answer_type=row.get("answer_type", "entity"),
            ))
    return items


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts: List[str], device: torch.device, max_new_tokens: int) -> List[str]:
    tok = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    out = model.generate(
        **tok,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
    )
    # strip prompt tokens
    gen_ids = out[:, tok["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return texts


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute accuracy and hallucination-like rates for each (category, lang) cell.
    """
    from collections import defaultdict
    cell = defaultdict(lambda: {"n":0, "acc":0, "hall":0})

    for r in rows:
        key = (r["category"], r["lang"])
        cell[key]["n"] += 1
        cell[key]["acc"] += int(r["correct"])
        cell[key]["hall"] += int(r["hallucination_like"])

    summary = {}
    for (cat, lang), v in cell.items():
        n = v["n"]
        summary[f"{cat}__{lang}"] = {
            "n": n,
            "accuracy": (v["acc"]/n) if n else 0.0,
            "hallucination_like_rate": (v["hall"]/n) if n else 0.0,
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to JSONL dataset")
    ap.add_argument("--models", required=True, nargs="+", help="HF model ids or local paths")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--dtype", default="auto", choices=["auto","fp16","bf16","fp32"])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    examples = load_jsonl(args.data)
    device = pick_device(args.device)

    for model_id in args.models:
        print(f"\n=== Running model: {model_id} on {len(examples)} prompts ===")
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        torch_dtype = None
        if args.dtype == "auto":
            if device.type == "cuda":
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                torch_dtype = torch.float32
        elif args.dtype == "fp16":
            torch_dtype = torch.float16
        elif args.dtype == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True
        )
        model.eval()
        if device.type != "cuda":
            model.to(device)

        rows: List[Dict[str, Any]] = []
        prompts: List[str] = []
        meta: List[Example] = []

        for ex in examples:
            prompts.append(build_prompt(ex.lang, ex.question))
            meta.append(ex)

            if len(prompts) >= args.batch_size:
                gens = generate_batch(model, tok, prompts, device, args.max_new_tokens)
                for ex2, g in zip(meta, gens):
                    ans = extract_short_answer(g)
                    correct = match_any(ans, ex2.answers, ex2.answer_type)
                    hall_like = (not correct) and (not is_idk(ans))
                    rows.append({
                        "model": model_id,
                        "id": ex2.qid,
                        "category": ex2.category,
                        "lang": ex2.lang,
                        "question": ex2.question,
                        "gold_answers": ex2.answers,
                        "answer_type": ex2.answer_type,
                        "generation_raw": g,
                        "answer_extracted": ans,
                        "correct": bool(correct),
                        "hallucination_like": bool(hall_like),
                    })
                prompts, meta = [], []

        # flush remainder
        if prompts:
            gens = generate_batch(model, tok, prompts, device, args.max_new_tokens)
            for ex2, g in zip(meta, gens):
                ans = extract_short_answer(g)
                correct = match_any(ans, ex2.answers, ex2.answer_type)
                hall_like = (not correct) and (not is_idk(ans))
                rows.append({
                    "model": model_id,
                    "id": ex2.qid,
                    "category": ex2.category,
                    "lang": ex2.lang,
                    "question": ex2.question,
                    "gold_answers": ex2.answers,
                    "answer_type": ex2.answer_type,
                    "generation_raw": g,
                    "answer_extracted": ans,
                    "correct": bool(correct),
                    "hallucination_like": bool(hall_like),
                })

        # save
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id)
        pred_path = outdir / f"{safe_name}.predictions.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summ = summarize(rows)
        summ_path = outdir / f"{safe_name}.summary.json"
        with open(summ_path, "w", encoding="utf-8") as f:
            json.dump(summ, f, ensure_ascii=False, indent=2)

        print(f"Saved predictions: {pred_path}")
        print(f"Saved summary: {summ_path}")
        print("Summary:")
        for k, v in sorted(summ.items()):
            print(f"  {k}: acc={v['accuracy']:.3f}, hall_like={v['hallucination_like_rate']:.3f}, n={v['n']}")

        # cleanup to free VRAM
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
