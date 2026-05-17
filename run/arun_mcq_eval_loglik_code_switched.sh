#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET="yangzhang33/CEB_code_switched"

# ── Output directories ────────────────────────────────────────────────────────
OUTDIR_CS_MIX="../results/ceb_code_switched_loglik_v1_cs_mix"
OUTDIR_EN_MIX="../results/ceb_code_switched_loglik_v1_en_mix"

# ── Run settings ──────────────────────────────────────────────────────────────
BATCH_SIZE=4
MAX_SAMPLES=       # leave empty for no limit
LOCAL_ONLY=true

# ── Models to evaluate ────────────────────────────────────────────────────────
MODELS=(
    # chinese models
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B"
    "Qwen/Qwen2.5-14B-Instruct"
    "deepseek-ai/deepseek-llm-7b-base"
    "deepseek-ai/deepseek-llm-7b-chat"
    # english models
    "meta-llama/Meta-Llama-3.1-8B"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-2-9b"
    "google/gemma-2-9b-it"
    "google/gemma-3-12b-pt"
    "google/gemma-3-12b-it"
    # greek models
    "ilsp/Llama-Krikri-8B-Instruct"
    "ilsp/Meltemi-7B-Instruct-v1.5"
    # arabic models
    "inceptionai/Jais-2-8B-Chat"
    "FreedomIntelligence/AceGPT-v2-8B-Chat"
    # hindi models
    "sarvamai/OpenHathi-7B-Hi-v0.1-Base"
    # southeast asian models
    "aisingapore/Llama-SEA-LION-v3-8B"
    "aisingapore/Llama-SEA-LION-v3-8B-IT"
    "SeaLLMs/SeaLLM-7B-v2.5"
    "SeaLLMs/SeaLLMs-v3-7B"
    "SeaLLMs/SeaLLMs-v3-7B-Chat"
    # korean models
    "beomi/Llama-3-Open-Ko-8B"
    "EleutherAI/polyglot-ko-12.8b"
    "EleutherAI/polyglot-ko-5.8b"
    # multilingual models
    "CohereLabs/aya-expanse-8b"
    # mistral models
    "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Base-2512"
    "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Instruct-2512"
    "/datalake/datastore1/yang/_hf_models/Lucie-7B"
    "/datalake/datastore1/yang/_hf_models/EuroLLM-9B-2512"
    "/datalake/datastore1/yang/_hf_models/EuroLLM-9B-Instruct-2512"
    "/datalake/datastore1/yang/_hf_models/EuroLLM-22B-2512"
    "/datalake/datastore1/yang/_hf_models/EuroLLM-22B-Instruct-2512"
    "/datalake/datastore1/yang/_hf_models/Teuken-7B-base-v0.6"
    "/datalake/datastore1/yang/_hf_models/Teuken-7B-instruct-v0.6"
    "/datalake/datastore1/yang/_hf_models/DeepSeek-V2-Lite"
    "/datalake/datastore1/yang/_hf_models/DeepSeek-V2-Lite-Chat"
    "/datalake/datastore1/yang/_hf_models/GLM-4.7-Flash"
    "/datalake/datastore1/yang/_hf_models/glm-4-9b-chat-hf"
    "/datalake/datastore1/yang/_hf_models/glm-4-9b-hf"
    "/datalake/datastore1/yang/_hf_models/Gaperon-1125-8B"
    "/datalake/datastore1/yang/_hf_models/Gaperon-1125-8B-SFT"
    "/datalake/datastore1/yang/_hf_models/Gaperon-1125-24B"
    "/datalake/datastore1/yang/_hf_models/Gaperon-1125-24B-SFT"
    "/datalake/datastore1/yang/_hf_models/Minerva-7B-base-v1.0"
    "google/gemma-2-27b-it"
    "Qwen/Qwen2.5-32B"
    "Qwen/Qwen2.5-32B-Instruct"
    "google/gemma-2-27b"       # oom
    "google/gemma-3-27b-pt"        # oom
    "google/gemma-3-27b-it"        # oom

)

# ── Subsets ───────────────────────────────────────────────────────────────────
SUBSETS_CS_MIX=(
    "arabic_cs_mix"
    "bengali_cs_mix"
    "chinese_cs_mix"
    "dutch_cs_mix"
    "french_cs_mix"
    "greek_cs_mix"
    "hebrew_cs_mix"
    "hindi_cs_mix"
    "indonesian_cs_mix"
    "italic_cs_mix"
    "japanese_cs_mix"
    "korean_cs_mix"
    "nepali_cs_mix"
    "persian_cs_mix"
    "polish_cs_mix"
    "russian_cs_mix"
    "spanish_cs_mix"
    "telugu_cs_mix"
    "ukrainian_cs_mix"
)

SUBSETS_EN_MIX=(
    "arabic_en_mix"
    "bengali_en_mix"
    "chinese_en_mix"
    "dutch_en_mix"
    "french_en_mix"
    "greek_en_mix"
    "hebrew_en_mix"
    "hindi_en_mix"
    "indonesian_en_mix"
    "italic_en_mix"
    "japanese_en_mix"
    "korean_en_mix"
    "nepali_en_mix"
    "persian_en_mix"
    "polish_en_mix"
    "russian_en_mix"
    "spanish_en_mix"
    "telugu_en_mix"
    "ukrainian_en_mix"
)

# ── Shared flags ──────────────────────────────────────────────────────────────
cd "$(dirname "$0")"

EXTRA_ARGS=()
[ "$LOCAL_ONLY" = true ] && EXTRA_ARGS+=(--local-only)
[ -n "$MAX_SAMPLES" ]    && EXTRA_ARGS+=(--max-samples "$MAX_SAMPLES")

# ── Run 1: CS-mix ────────────────────────────────────────────────────────────
echo "======== CS-mix run ========"
python run_mcq_eval_loglik_code_switched.py \
    --dataset    "$DATASET"        \
    --outdir     "$OUTDIR_CS_MIX"  \
    --batch-size "$BATCH_SIZE"     \
    "${EXTRA_ARGS[@]}"             \
    --models  "${MODELS[@]}"       \
    --subsets "${SUBSETS_CS_MIX[@]}"

# ── Run 2: EN-mix ────────────────────────────────────────────────────────────
echo "======== EN-mix run ========"
python run_mcq_eval_loglik_code_switched.py \
    --dataset    "$DATASET"        \
    --outdir     "$OUTDIR_EN_MIX"  \
    --batch-size "$BATCH_SIZE"     \
    "${EXTRA_ARGS[@]}"             \
    --models  "${MODELS[@]}"       \
    --subsets "${SUBSETS_EN_MIX[@]}"
