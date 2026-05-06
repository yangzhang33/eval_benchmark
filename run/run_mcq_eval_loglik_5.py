#!/usr/bin/env python3
"""
MCQ evaluation on yangzhang33/culture-eval-benchmark using log-likelihood scoring,
with an "I don't know" option E added to every prompt.

Instead of generating text, computes the log-probability of each answer token
(" A", " B", " C", " D", " E") at the final prompt position and picks the highest.
This works uniformly for base and instruct models without any special formatting.

Usage:
  python run_mcq_eval_loglik_5.py --outdir results
  python run_mcq_eval_loglik_5.py --models Qwen/Qwen2.5-7B --outdir results
"""
import inspect
import json
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Mistral3ForConditionalGeneration, MistralCommonBackend

DATASET = "yangzhang33/culture-eval-benchmark-cs-filtered-lite"
CA_MODE = False
BATCH_SIZE = 1
MAX_SAMPLES_PER_SUBSET = None
LOCAL_ONLY = True

SUBSETS_cs = [
    "chinese_cs", "chinese_cs_en",
    "arabic_cs", "arabic_cs_en",
    "greek_cs", "greek_cs_en",
    "hindi_cs", "hindi_cs_en",
    "indonesian_cs", "indonesian_cs_en",
    "korean_cs", "korean_cs_en",
    # "italic_cs", "italic_cs_en",
]

SUBSETS_ca = [
    "english_ca",
    "chinese_ca",
    "arabic_ca",
    "greek_ca",
    "hindi_ca",
    "indonesian_ca",
    "korean_ca",
    # "italic_ca",
]


if CA_MODE:
    OUTDIR = "../results/ca_loglik_v1_5/ca_results"
    SUBSETS = SUBSETS_ca
else:
    OUTDIR = "../results/cs_filtered_lite_eval_loglik_v1_5"
    SUBSETS = SUBSETS_cs


MODELS = [
    # chinese models
    # "Qwen/Qwen2.5-7B",
    # "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-32B-Instruct",
    # "deepseek-ai/deepseek-llm-7b-base",
    # "deepseek-ai/deepseek-llm-7b-chat",
    # #english models
    # "meta-llama/Meta-Llama-3.1-8B",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "google/gemma-2-9b",
    # "google/gemma-2-9b-it",




    # "google/gemma-2-27b",

    # "google/gemma-2-27b-it", #oom
    # "google/gemma-3-12b-pt",
    # "google/gemma-3-12b-it",
    # "google/gemma-3-27b-pt", # oom
    # "google/gemma-3-27b-it", # oom



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
    # # mistral models
    # # "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Base-2512",
    # # "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Instruct-2512",
    # # "/datalake/datastore1/yang/_hf_models/Lucie-7B",
]

MISTRAL3_MODELS = {
    "mistralai/Ministral-3-8B-Base-2512",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Base-2512",
    "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Instruct-2512",
}

# Localized prompt templates: {lang: (question_label, answer_label, idk_option)}
PROMPT_LANG = {
    "zh": ("问题：", "答案：", "我不知道"),
    "el": ("Ερώτηση: ", "Απάντηση:", "Δεν ξέρω"),
    "ar": ("السؤال: ", "الجواب:", "لا أعرف"),
    "hi": ("प्रश्न: ", "उत्तर:", "मुझे नहीं पता"),
    "id": ("Pertanyaan: ", "Jawaban:", "Saya tidak tahu"),
    "ko": ("질문: ", "답변:", "모르겠습니다"),
    "it": ("Domanda: ", "Risposta:", "Non lo so"),
    "en": ("Question: ", "Answer:", "I don't know"),
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
}

ANSWER_LETTERS = ["A", "B", "C", "D", "E"]


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
    q_label, ans_label, idk_option = PROMPT_LANG[lang]
    return (
        f"{q_label}{row['question']}\n"
        f"A. {row['option_a']}\n"
        f"B. {row['option_b']}\n"
        f"C. {row['option_c']}\n"
        f"D. {row['option_d']}\n"
        f"E. {idk_option}\n"
        f"{ans_label}"
    )


@torch.inference_mode()
def evaluate_model(model_id: str, subsets: list[str], batch_size: int, max_samples_per_subset: int | None = None, local_only: bool = False) -> tuple[dict, dict, dict, dict, list]:
    """Load model, run on all subsets, return (accuracy, abstain_rates, conf_err_rates, cond_accs, raw_records)."""
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
    abstain_rates = {}
    conf_err_rates = {}
    cond_accs = {}
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

        n = len(gold)
        correct    = sum(p == g for p, g in zip(preds, gold))
        abstained  = sum(p == "E" for p in preds)
        conf_wrong = sum(p != g and p != "E" for p, g in zip(preds, gold))

        acc           = correct / n if n else 0.0
        abstain_rate  = abstained / n if n else 0.0
        conf_err_rate = conf_wrong / n if n else 0.0
        attempted     = correct + conf_wrong
        cond_acc      = correct / attempted if attempted else 0.0

        accuracy[subset]       = round(acc,           4)
        abstain_rates[subset]  = round(abstain_rate,  4)
        conf_err_rates[subset] = round(conf_err_rate, 4)
        cond_accs[subset]      = round(cond_acc,      4)
        print(
            f"    acc={acc:.4f}  abstain={abstain_rate:.4f}  "
            f"conf_err={conf_err_rate:.4f}  cond_acc={cond_acc:.4f}  "
            f"({correct}/{n})"
        )

    del model
    del tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return accuracy, abstain_rates, conf_err_rates, cond_accs, raw_records


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def main():
    _ds = DATASET.split("/")[-1]
    if _ds == "culture-eval-benchmark-cs-filtered-lite":
        assert "v1" in OUTDIR, (
            f"DATASET is '{_ds}' but OUTDIR '{OUTDIR}' does not contain 'v1'. "
            "Please set OUTDIR to a path containing 'v1'."
        )
    elif _ds == "culture-eval-benchmark-cs-filtered-lite-human-filtered":
        assert "v2" in OUTDIR, (
            f"DATASET is '{_ds}' but OUTDIR '{OUTDIR}' does not contain 'v2'. "
            "Please set OUTDIR to a path containing 'v2'."
        )

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    for model_id in MODELS:
        accuracy, abstain_rates, conf_err_rates, cond_accs, raw_records = evaluate_model(
            model_id, SUBSETS, BATCH_SIZE, MAX_SAMPLES_PER_SUBSET, local_only=LOCAL_ONLY
        )
        slug = model_slug(model_id)

        # Per-model accuracy JSON (merge with existing)
        acc_path = outdir / f"{slug}_accuracy.json"
        if acc_path.exists():
            with open(acc_path, encoding="utf-8") as f:
                existing_acc = json.load(f)
            for key, new_vals in [
                ("accuracy",      accuracy),
                ("abstain_rate",  abstain_rates),
                ("conf_err_rate", conf_err_rates),
                ("cond_acc",      cond_accs),
            ]:
                existing_acc[key].update(new_vals)
            accuracy      = existing_acc["accuracy"]
            abstain_rates = existing_acc["abstain_rate"]
            conf_err_rates = existing_acc["conf_err_rate"]
            cond_accs     = existing_acc["cond_acc"]
        with open(acc_path, "w", encoding="utf-8") as f:
            json.dump({
                "model":         model_id,
                "accuracy":      accuracy,
                "abstain_rate":  abstain_rates,
                "conf_err_rate": conf_err_rates,
                "cond_acc":      cond_accs,
            }, f, ensure_ascii=False, indent=2)
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
            json.dump({"model": model_id, "predictions": raw_records}, f, ensure_ascii=False, indent=2)
        print(f"Saved predictions to {raw_path}")

    # Print table
    print(f"\n{'Model':<45} {'Subset':<20} {'Acc':>7} {'Abstain':>8} {'ConfErr':>8} {'CondAcc':>8}")
    print("-" * 100)
    for model_id in MODELS:
        slug = model_slug(model_id)
        acc_path = outdir / f"{slug}_accuracy.json"
        with open(acc_path, encoding="utf-8") as f:
            data = json.load(f)
        for subset in data["accuracy"]:
            print(
                f"{model_id:<45} {subset:<20} "
                f"{data['accuracy'][subset]:>7.4f} "
                f"{data['abstain_rate'][subset]:>8.4f} "
                f"{data['conf_err_rate'][subset]:>8.4f} "
                f"{data['cond_acc'][subset]:>8.4f}"
            )


if __name__ == "__main__":
    main()
