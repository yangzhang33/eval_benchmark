#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET="yangzhang33/culture-eval-benchmark-cs-filtered-lite"
# DATASET="yangzhang33/culture-eval-benchmark-cs-filtered-lite-human-filtered"

# ── Output directories ────────────────────────────────────────────────────────

OUTDIR_CS="../results/cs_filtered_lite_eval_loglik_v1"
OUTDIR_CS_V3="../results/cs_filtered_lite_eval_loglik_cs_v3"
OUTDIR_CS_EN_V3="../results/cs_filtered_lite_eval_loglik_cs_en_v3"
OUTDIR_CA="../results/ca_loglik_v1/ca_results"

# ── Run settings ──────────────────────────────────────────────────────────────
BATCH_SIZE=4
MAX_SAMPLES=       # leave empty for no limit
LOCAL_ONLY=true
# V3_EN=false        # prepend culture preamble to _cs_en prompts
# V3_CS=false        # prepend native-language culture preamble to _cs prompts

# ── Models to evaluate ────────────────────────────────────────────────────────
MODELS=(
    # # chinese models
    # "Qwen/Qwen2.5-7B"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen2.5-32B"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "deepseek-ai/deepseek-llm-7b-base"
    # "deepseek-ai/deepseek-llm-7b-chat"
    # # english models
    # "meta-llama/Meta-Llama-3.1-8B"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "google/gemma-2-9b"
    # "google/gemma-2-9b-it"
    # "google/gemma-2-27b"       # oom
    # "google/gemma-3-12b-pt"
    # "google/gemma-3-12b-it"
    # "google/gemma-3-27b-pt"        # oom
    # "google/gemma-3-27b-it"        # oom
    # # greek models
    # "ilsp/Llama-Krikri-8B-Instruct"
    # "ilsp/Meltemi-7B-Instruct-v1.5"
    # # arabic models
    # "inceptionai/Jais-2-8B-Chat"
    # "FreedomIntelligence/AceGPT-v2-8B-Chat"
    # # hindi models
    # "sarvamai/OpenHathi-7B-Hi-v0.1-Base"
    # # southeast asian models
    # "aisingapore/Llama-SEA-LION-v3-8B"
    # "aisingapore/Llama-SEA-LION-v3-8B-IT"
    # "SeaLLMs/SeaLLM-7B-v2.5"
    # "SeaLLMs/SeaLLMs-v3-7B"
    # "SeaLLMs/SeaLLMs-v3-7B-Chat"
    # # korean models
    # "beomi/Llama-3-Open-Ko-8B"
    # "EleutherAI/polyglot-ko-12.8b"
    # "EleutherAI/polyglot-ko-5.8b"
    # # multilingual models
    # "CohereLabs/aya-expanse-8b"
    # # mistral models
    # "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Base-2512"
    # "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Instruct-2512"
    # "/datalake/datastore1/yang/_hf_models/Lucie-7B"
    # "/datalake/datastore1/yang/_hf_models/EuroLLM-9B-2512"
    # "/datalake/datastore1/yang/_hf_models/EuroLLM-9B-Instruct-2512"
    # "/datalake/datastore1/yang/_hf_models/EuroLLM-22B-2512"
    # "/datalake/datastore1/yang/_hf_models/EuroLLM-22B-Instruct-2512"
    # "/datalake/datastore1/yang/_hf_models/Teuken-7B-base-v0.6"
    # "/datalake/datastore1/yang/_hf_models/Teuken-7B-instruct-v0.6"
    # "/datalake/datastore1/yang/_hf_models/DeepSeek-V2-Lite"
    # "/datalake/datastore1/yang/_hf_models/DeepSeek-V2-Lite-Chat"
    # "/datalake/datastore1/yang/_hf_models/GLM-4.7-Flash"
    # "/datalake/datastore1/yang/_hf_models/glm-4-9b-chat-hf"
    # "/datalake/datastore1/yang/_hf_models/glm-4-9b-hf"
    "/datalake/datastore1/yang/_hf_models/Gaperon-1125-8B"
    "/datalake/datastore1/yang/_hf_models/Gaperon-1125-8B-SFT"
    "/datalake/datastore1/yang/_hf_models/Gaperon-1125-24B"
    "/datalake/datastore1/yang/_hf_models/Gaperon-1125-24B-SFT"
    "/datalake/datastore1/yang/_hf_models/Minerva-7B-base-v1.0"
    # "google/gemma-2-27b-it" 
)

# ── Subsets ───────────────────────────────────────────────────────────────────
SUBSETS_CS_ALL=(
    "chinese_cs"      "chinese_cs_en"
    "arabic_cs"       "arabic_cs_en"
    "greek_cs"        "greek_cs_en"
    "hindi_cs"        "hindi_cs_en"
    "indonesian_cs"   "indonesian_cs_en"
    "korean_cs"       "korean_cs_en"
    "italic_cs"       "italic_cs_en"
    "french_cs"       "french_cs_en"
    "japanese_cs"     "japanese_cs_en"
    "spanish_cs"      "spanish_cs_en"
    "bengali_cs"      "bengali_cs_en"
    "dutch_cs"        "dutch_cs_en"
    "hebrew_cs"       "hebrew_cs_en"
    "nepali_cs"       "nepali_cs_en"
    "persian_cs"      "persian_cs_en"
    "polish_cs"       "polish_cs_en"
    "russian_cs"      "russian_cs_en"
    "telugu_cs"       "telugu_cs_en"
    "ukrainian_cs"    "ukrainian_cs_en"
)

SUBSETS_CS=(
    "chinese_cs"
    "arabic_cs"
    "greek_cs"
    "hindi_cs"
    "indonesian_cs"
    "korean_cs"
    "italic_cs"
    "french_cs"
    "japanese_cs"
    "spanish_cs"
    "bengali_cs"
    "dutch_cs"
    "hebrew_cs"
    "nepali_cs"
    "persian_cs"
    "polish_cs"
    "russian_cs"
    "telugu_cs"
    "ukrainian_cs"
)

SUBSETS_CS_EN=(
    "chinese_cs_en"
    "arabic_cs_en"
    "greek_cs_en"
    "hindi_cs_en"
    "indonesian_cs_en"
    "korean_cs_en"
    "italic_cs_en"
    "french_cs_en"
    "japanese_cs_en"
    "spanish_cs_en"
    "bengali_cs_en"
    "dutch_cs_en"
    "hebrew_cs_en"
    "nepali_cs_en"
    "persian_cs_en"
    "polish_cs_en"
    "russian_cs_en"
    "telugu_cs_en"
    "ukrainian_cs_en"
)

SUBSETS_CA=(
    "english_ca"
    "chinese_ca"
    "arabic_ca"
    "greek_ca"
    "hindi_ca"
    "indonesian_ca"
    "korean_ca"
    "italic_ca"
    "french_ca"
    "japanese_ca"
    "spanish_ca"
    "bengali_ca"
    "dutch_ca"
    "hebrew_ca"
    "nepali_ca"
    "persian_ca"
    "polish_ca"
    "russian_ca"
    "telugu_ca"
    "ukrainian_ca"
)

# ── Shared flags ──────────────────────────────────────────────────────────────
cd "$(dirname "$0")"

EXTRA_ARGS=()
[ "$LOCAL_ONLY" = true ] && EXTRA_ARGS+=(--local-only)
[ -n "$MAX_SAMPLES" ]    && EXTRA_ARGS+=(--max-samples "$MAX_SAMPLES")
# [ "$V3_EN" = true ]      && EXTRA_ARGS+=(--v3-en)
# [ "$V3_CS" = true ]      && EXTRA_ARGS+=(--v3-cs)

# ── Run 1: CS mode ───────────────────────────────────────────────────────────
echo "======== CS run ========"
python run_mcq_eval_loglik.py \
    --dataset    "$DATASET"    \
    --outdir     "$OUTDIR_CS"  \
    --batch-size "$BATCH_SIZE" \
    "${EXTRA_ARGS[@]}"         \
    --models  "${MODELS[@]}"   \
    --subsets "${SUBSETS_CS_ALL[@]}"

# ── Run 3: CA mode ────────────────────────────────────────────────────────────
echo "======== CA run ========"
python run_mcq_eval_loglik.py \
    --dataset    "$DATASET"    \
    --outdir     "$OUTDIR_CA"  \
    --batch-size "$BATCH_SIZE" \
    "${EXTRA_ARGS[@]}"         \
    --models  "${MODELS[@]}"   \
    --subsets "${SUBSETS_CA[@]}"


# ── Run 2: CS v3 (native-language culture preamble) ──────────────────────────
echo "======== CS v3 run ========"
python run_mcq_eval_loglik.py \
    --dataset    "$DATASET"         \
    --outdir     "$OUTDIR_CS_V3"    \
    --batch-size "$BATCH_SIZE"      \
    "${EXTRA_ARGS[@]}"              \
    --v3-cs                         \
    --models  "${MODELS[@]}"        \
    --subsets "${SUBSETS_CS[@]}"

# ── Run 3: CS_EN v3 (English culture preamble) ───────────────────────────────
echo "======== CS_EN v3 run ========"
python run_mcq_eval_loglik.py \
    --dataset    "$DATASET"         \
    --outdir     "$OUTDIR_CS_EN_V3" \
    --batch-size "$BATCH_SIZE"      \
    "${EXTRA_ARGS[@]}"              \
    --v3-en                         \
    --models  "${MODELS[@]}"        \
    --subsets "${SUBSETS_CS_EN[@]}"

