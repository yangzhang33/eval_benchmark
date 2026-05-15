#!/usr/bin/env bash
set -euo pipefail

# Probe: is the top-1 predicted token after the MCQ prompt actually an
# A/B/C/D variant (with or without leading space/newline), or something else?
#
# Runs each (model, subset) pair, max 100 samples per subset.
# Output: test/results/<model_slug>__<subset>.json + test/results/summary.json

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")"

DATASET="yangzhang33/culture-eval-benchmark-cs-filtered-lite"
OUTDIR="results"
BATCH_SIZE=32
MAX_SAMPLES=100
TOP_K=10
LOCAL_ONLY=true

# ── Models (mirrors run/arun_mcq_eval_loglik.sh) ──────────────────────────────
MODELS=(
    # chinese models
    # "Qwen/Qwen2.5-7B"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen2.5-32B"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "deepseek-ai/deepseek-llm-7b-base"
    # "deepseek-ai/deepseek-llm-7b-chat"
    # english models
    "meta-llama/Meta-Llama-3.1-8B"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-2-9b"
    "google/gemma-2-9b-it"
    # "google/gemma-2-27b"       # oom
    "google/gemma-3-12b-pt"
    "google/gemma-3-12b-it"
    # "google/gemma-3-27b-pt"        # oom
    # "google/gemma-3-27b-it"        # oom
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
    # "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Base-2512"
    # "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Instruct-2512"
    # "/datalake/datastore1/yang/_hf_models/Lucie-7B"
    "/datalake/datastore1/yang/_hf_models/EuroLLM-9B-2512"
    "/datalake/datastore1/yang/_hf_models/EuroLLM-9B-Instruct-2512"
    # "/datalake/datastore1/yang/_hf_models/EuroLLM-22B-2512"
    # "/datalake/datastore1/yang/_hf_models/EuroLLM-22B-Instruct-2512"
    "/datalake/datastore1/yang/_hf_models/Teuken-7B-base-v0.6"
    "/datalake/datastore1/yang/_hf_models/Teuken-7B-instruct-v0.6"
    # "/datalake/datastore1/yang/_hf_models/DeepSeek-V2-Lite"
    # "/datalake/datastore1/yang/_hf_models/DeepSeek-V2-Lite-Chat"
    "/datalake/datastore1/yang/_hf_models/GLM-4.7-Flash"
    "/datalake/datastore1/yang/_hf_models/glm-4-9b-chat-hf"
    "/datalake/datastore1/yang/_hf_models/glm-4-9b-hf"
    # "google/gemma-2-27b-it"
)

# ── Subsets: every language × {_ca, _cs, _cs_en} ─────────────────────────────
SUBSETS=(
    "english_ca"
    "chinese_ca"     "chinese_cs"     "chinese_cs_en"
    "arabic_ca"      "arabic_cs"      "arabic_cs_en"
    "greek_ca"       "greek_cs"       "greek_cs_en"
    "hindi_ca"       "hindi_cs"       "hindi_cs_en"
    "indonesian_ca"  "indonesian_cs"  "indonesian_cs_en"
    "korean_ca"      "korean_cs"      "korean_cs_en"
    "italic_ca"      "italic_cs"      "italic_cs_en"
    "french_ca"      "french_cs"      "french_cs_en"
    "japanese_ca"    "japanese_cs"    "japanese_cs_en"
    "spanish_ca"     "spanish_cs"     "spanish_cs_en"
    "bengali_ca"     "bengali_cs"     "bengali_cs_en"
    "dutch_ca"       "dutch_cs"       "dutch_cs_en"
    "hebrew_ca"      "hebrew_cs"      "hebrew_cs_en"
    "nepali_ca"      "nepali_cs"      "nepali_cs_en"
    "persian_ca"     "persian_cs"     "persian_cs_en"
    "polish_ca"      "polish_cs"      "polish_cs_en"
    "russian_ca"     "russian_cs"     "russian_cs_en"
    "telugu_ca"      "telugu_cs"      "telugu_cs_en"
    "ukrainian_ca"   "ukrainian_cs"   "ukrainian_cs_en"
)

EXTRA_ARGS=()
[ "$LOCAL_ONLY" = true ] && EXTRA_ARGS+=(--local-only)

python -u test_top_token.py \
    --dataset     "$DATASET"     \
    --outdir      "$OUTDIR"      \
    --batch-size  "$BATCH_SIZE"  \
    --max-samples "$MAX_SAMPLES" \
    --top-k       "$TOP_K"       \
    "${EXTRA_ARGS[@]}"           \
    --models  "${MODELS[@]}"     \
    --subsets "${SUBSETS[@]}"
