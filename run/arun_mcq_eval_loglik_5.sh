#!/usr/bin/env bash
set -euo pipefail

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET="yangzhang33/culture-eval-benchmark-cs-filtered-lite"
# DATASET="yangzhang33/culture-eval-benchmark-cs-filtered-lite-human-filtered"

# ── Output directories ────────────────────────────────────────────────────────
OUTDIR_CS="../results/cs_filtered_lite_eval_loglik_v1_5"
OUTDIR_CA="../results/ca_loglik_v1_5/ca_results"

# ── Run settings ──────────────────────────────────────────────────────────────
BATCH_SIZE=8
MAX_SAMPLES=       # leave empty for no limit
LOCAL_ONLY=true

# ── Models to evaluate ────────────────────────────────────────────────────────
# Group 1 = official Qwen / Llama / Gemma / Jais / Mistral (family order, size ↑)
# Group 2 = all other models (size ↑). Regional derivatives of Llama/Qwen
#           (Krikri, SEA-LION, Open-Ko, LLaMAntino, Swallow, AceGPT) live here.
MODELS=(
    # ══════════════════════════════════════════════════════════════════════════
    # Group 1: Qwen / Llama / Gemma / Jais / Mistral
    # ══════════════════════════════════════════════════════════════════════════
    # ── qwen ──
    # "/datalake/datastore1/yang/_hf_models/Qwen3-0.6B"                   # 0.6B
    # "/datalake/datastore1/yang/_hf_models/Qwen3.5-0.8B-Base"           # 0.8B
    # "/datalake/datastore1/yang/_hf_models/Qwen3.5-0.8B"
    # "/datalake/datastore1/yang/_hf_models/Qwen3-1.7B"                  # 1.7B
    # "/datalake/datastore1/yang/_hf_models/Qwen3.5-2B-Base"             # 2B
    # "/datalake/datastore1/yang/_hf_models/Qwen3.5-2B"
    # "/datalake/datastore1/yang/_hf_models/Qwen3-4B"                    # 4B
    # "/datalake/datastore1/yang/_hf_models/Qwen3-4B-Instruct-2507"
    # "/datalake/datastore1/yang/_hf_models/Qwen3.5-4B-Base"
    # "/datalake/datastore1/yang/_hf_models/Qwen3.5-4B"
    # "Qwen/Qwen2.5-7B"                                                  # 7B
    # "Qwen/Qwen2.5-7B-Instruct"
    # "/datalake/datastore1/yang/_hf_models/Qwen3.5-9B-Base"            # 9B
    # "/datalake/datastore1/yang/_hf_models/Qwen3.5-9B"


    # "Qwen/Qwen2.5-14B"                                                 # 14B
    # "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen2.5-32B"                                                 # 32B
    # "Qwen/Qwen2.5-32B-Instruct"
    # ── llama ──
    # "/datalake/datastore1/yang/_hf_models/Llama-3.2-1B"                # 1B
    # "/datalake/datastore1/yang/_hf_models/Llama-3.2-1B-Instruct"
    # "/datalake/datastore1/yang/_hf_models/Llama-3.2-3B"                # 3B
    # "/datalake/datastore1/yang/_hf_models/Llama-3.2-3B-Instruct"
    # "meta-llama/Meta-Llama-3.1-8B"                                     # 8B
    # "meta-llama/Llama-3.1-8B-Instruct"

    
    # # ── gemma ──
    # "/datalake/datastore1/yang/_hf_models/gemma-3-270m"                # 0.27B
    # "/datalake/datastore1/yang/_hf_models/gemma-3-270m-it"
    # "/datalake/datastore1/yang/_hf_models/gemma-3-1b-pt"               # 1B
    # "/datalake/datastore1/yang/_hf_models/gemma-3-1b-it"
    # "/datalake/datastore1/yang/_hf_models/gemma-4-E2B"                 # ~2B (eff.)
    # "/datalake/datastore1/yang/_hf_models/gemma-4-E2B-it"
    # "/datalake/datastore1/yang/_hf_models/gemma-3-4b-pt"               # 4B
    "/datalake/datastore1/yang/_hf_models/gemma-3-4b-it"
    "/datalake/datastore1/yang/_hf_models/gemma-4-E4B"                 # ~4B (eff.)
    "/datalake/datastore1/yang/_hf_models/gemma-4-E4B-it"
    "google/gemma-2-9b"                                                # 9B
    "google/gemma-2-9b-it"
    # "google/gemma-3-12b-pt"                                            # 12B
    # "google/gemma-3-12b-it"
    # "google/gemma-2-27b"                                               # 27B  # oom
    # "google/gemma-2-27b-it"
    # "google/gemma-3-27b-pt"                                            # oom
    # "google/gemma-3-27b-it"                                            # oom
    # ── jais ──
    "inceptionai/Jais-2-8B-Chat"                                       # 8B
    # ── mistral ──
    "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Base-2512"       # 8B
    # "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Instruct-2512" # not done

    # ══════════════════════════════════════════════════════════════════════════
    # Group 2: other models (size ↑)
    # ══════════════════════════════════════════════════════════════════════════
    # "EleutherAI/polyglot-ko-5.8b"                                      # 5.8B
    # "deepseek-ai/deepseek-llm-7b-base"                                 # 7B
    # "deepseek-ai/deepseek-llm-7b-chat"
    # "ilsp/Meltemi-7B-Instruct-v1.5"
    # "sarvamai/OpenHathi-7B-Hi-v0.1-Base"
    # "SeaLLMs/SeaLLM-7B-v2.5"
    # "SeaLLMs/SeaLLMs-v3-7B"
    # "SeaLLMs/SeaLLMs-v3-7B-Chat"
    # "/datalake/datastore1/yang/_hf_models/Lucie-7B"
    # "/datalake/datastore1/yang/_hf_models/Teuken-7B-base-v0.6"
    # "/datalake/datastore1/yang/_hf_models/Teuken-7B-instruct-v0.6"
    # "/datalake/datastore1/yang/_hf_models/Minerva-7B-base-v1.0"
    # "/datalake/datastore1/yang/_hf_models/Bielik-Minitron-7B-v3.0-Instruct"
    # "/datalake/datastore1/yang/_hf_models/PersianMind-v1.0"
    # "ilsp/Llama-Krikri-8B-Instruct"                                    # 8B
    # "FreedomIntelligence/AceGPT-v2-8B-Chat"
    # "aisingapore/Llama-SEA-LION-v3-8B"
    # "aisingapore/Llama-SEA-LION-v3-8B-IT"
    # "beomi/Llama-3-Open-Ko-8B"
    # "CohereLabs/aya-expanse-8b"
    # "/datalake/datastore1/yang/_hf_models/Gaperon-1125-8B"
    # "/datalake/datastore1/yang/_hf_models/Gaperon-1125-8B-SFT"
    # "/datalake/datastore1/yang/_hf_models/Qwen3-Swallow-8B-CPT-v0.2"
    # "/datalake/datastore1/yang/_hf_models/Qwen3-Swallow-8B-SFT-v0.2"
    # "/datalake/datastore1/yang/_hf_models/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
    # "/datalake/datastore1/yang/_hf_models/EuroLLM-9B-2512"             # 9B
    # "/datalake/datastore1/yang/_hf_models/EuroLLM-9B-Instruct-2512"
    # "/datalake/datastore1/yang/_hf_models/glm-4-9b-hf"
    # "/datalake/datastore1/yang/_hf_models/glm-4-9b-chat-hf"
    # "/datalake/datastore1/yang/_hf_models/GLM-4.7-Flash"               # size? (~9B guess)
    # "/datalake/datastore1/yang/_hf_models/KORMo-10B-base"              # 10B
    # "/datalake/datastore1/yang/_hf_models/KORMo-10B-sft"
    # "/datalake/datastore1/yang/_hf_models/Bielik-11B-v3-Base-20250730"   # 11B
    # "EleutherAI/polyglot-ko-12.8b"                                     # 12.8B
    # "/datalake/datastore1/yang/_hf_models/DeepSeek-V2-Lite"           # ~16B MoE (2.4B act.)
    # "/datalake/datastore1/yang/_hf_models/DeepSeek-V2-Lite-Chat"
    # "/datalake/datastore1/yang/_hf_models/EuroLLM-22B-2512"           # 22B
    # "/datalake/datastore1/yang/_hf_models/EuroLLM-22B-Instruct-2512"
    # "/datalake/datastore1/yang/_hf_models/Gaperon-1125-24B"           # 24B
    # "/datalake/datastore1/yang/_hf_models/Gaperon-1125-24B-SFT"

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
    "spanish_cs"      "spanish_cs_en"
    "bengali_cs"      "bengali_cs_en"
    "dutch_cs"        "dutch_cs_en"
    # "hebrew_cs"       "hebrew_cs_en"
    "nepali_cs"       "nepali_cs_en"
    "persian_cs"      "persian_cs_en"
    "polish_cs"       "polish_cs_en"
    "russian_cs"      "russian_cs_en"
    "telugu_cs"       "telugu_cs_en"
    "ukrainian_cs"    "ukrainian_cs_en"
    "japanese_cs"     "japanese_cs_en"
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
    # "hebrew_ca"
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

# ── Run 1: CS mode ───────────────────────────────────────────────────────────
# echo "======== CS run ========"
# python run_mcq_eval_loglik_5.py \
#     --dataset    "$DATASET"    \
#     --outdir     "$OUTDIR_CS"  \
#     --batch-size "$BATCH_SIZE" \
#     "${EXTRA_ARGS[@]}"         \
#     --models  "${MODELS[@]}"   \
#     --subsets "${SUBSETS_CS_ALL[@]}"

# ── Run 2: CA mode ────────────────────────────────────────────────────────────
echo "======== CA run ========"
python run_mcq_eval_loglik_5.py \
    --dataset    "$DATASET"    \
    --outdir     "$OUTDIR_CA"  \
    --batch-size "$BATCH_SIZE" \
    "${EXTRA_ARGS[@]}"         \
    --models  "${MODELS[@]}"   \
    --subsets "${SUBSETS_CA[@]}"
