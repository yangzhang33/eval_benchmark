"""
Classifies whether each question in the culture-eval-benchmark dataset is
"culture sensitive" — i.e. rooted in local cultural knowledge and NOT a
language-independent question (like maths or general science).

For every config whose name ends with '_cs', the script:
  1. Loads the split from HuggingFace.
  2. Asks Claude (via AWS Bedrock) to judge each question.
  3. Adds a new integer column "claude_cs":
        1 → culturally dependent
        0 → NOT culturally dependent (language-independent)
  4. Saves progress to a local JSONL file after every 100 rows so the run
     can be resumed if interrupted.
  5. Pushes the annotated split back to the same HF repo under the same
     config name (overwriting the existing split).
"""

import json
import re
import boto3
from botocore.config import Config
from datasets import load_dataset, DatasetDict, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_REPO_ID = "yangzhang33/culture-eval-benchmark"

# If set, annotated splits are pushed here instead of back to HF_REPO_ID.
# Set to None to overwrite the source dataset in place.
# Example: HF_OUTPUT_REPO_ID = "yangzhang33/culture-eval-benchmark-labeled"
HF_OUTPUT_REPO_ID = "yangzhang33/culture-eval-benchmark-labeled"

# BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"

# Each entry is (config_name, language, country/region) so the classifier
# prompt can explicitly name the culture being evaluated.
CONFIGS = [
    # ("chinese_cs",    "Chinese",    "China"),
    # ("arabic_cs",     "Arabic",     "Arab world"),
    # ("greek_cs",      "Greek",      "Greece"),
    # ("hindi_cs",      "Hindi",      "India"),
    # ("indonesian_cs", "Indonesian", "Indonesia"),
    # ("korean_cs",     "Korean",     "Korea"),
    ("italic_cs",     "Italian",    "Italy"),
]

# Set to a positive integer to process only the first N samples (for testing).
# None = process every sample.
MAX_SAMPLES = None

# ---------------------------------------------------------------------------
# AWS Bedrock client
# ---------------------------------------------------------------------------

config = Config(read_timeout=1000)
brt = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
    config=config,
)


# ---------------------------------------------------------------------------
# Claude helpers
# ---------------------------------------------------------------------------

def invoke_claude(prompt: str) -> str:
    """Send a single-turn prompt to Claude and return the text response."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 128,          # classification only needs a short reply
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 0.9,
    })
    response = brt.invoke_model(
        body=body,
        modelId=BEDROCK_MODEL_ID,
        accept="application/json",
        contentType="application/json",
    )
    return json.loads(response.get("body").read()).get("content")[0]["text"]


def classify_row(row: dict, language: str, country: str) -> int:
    """
    Ask Claude whether the question is culturally dependent for the given
    language/country context.

    Returns 1 if culturally dependent, 0 if language-independent.
    Falls back to 0 if the response cannot be parsed.
    """
    # Build a short context from the question and answer options so Claude
    # can judge the topic even without translating it.
    question_text = row.get("question", "")
    options = " | ".join(
        str(row.get(k, ""))
        for k in ("option_a", "option_b", "option_c", "option_d")
        if row.get(k)
    )

    prompt = (
        "You are classifying exam questions by cultural specificity.\n\n"
        f"The question is written in {language} and targets knowledge about {country}.\n\n"
        "Label the question as follows:\n"
        f"  1 (CULTURALLY DEPENDENT)  — the question requires knowledge specific to "
        f"{country}, such as its history, traditions, cuisine, religion, geography, "
        "literature, public figures, or social customs. A person unfamiliar with "
        f"{country} culture would not be able to answer it.\n"
        "  0 (NOT CULTURALLY DEPENDENT) — the question tests universal knowledge "
        "that does not depend on any particular culture, such as mathematics, "
        "logic, or widely-known global science facts.\n\n"
        f"Question: {question_text}\n"
        f"Options: {options}\n\n"
        "Respond with ONLY this JSON, no explanation, no markdown:\n"
        "{\"culturally_dependent\": <0 or 1>}"
    )

    raw = invoke_claude(prompt)

    # Parse the JSON answer; fall back gracefully on any error.
    try:
        # Accept both bare JSON and markdown-fenced JSON
        text = raw.strip()
        if "```" in text:
            m = re.search(r'\{.*?\}', text, re.DOTALL)
            text = m.group(0) if m else text
        parsed = json.loads(text)
        return int(bool(parsed.get("culturally_dependent", 0)))
    except Exception:
        # If Claude's reply is unexpected, treat the question as non-cultural
        # so we don't silently inflate the cultural count.
        print(f"\nWARNING: Could not parse Claude response: {raw!r}. Defaulting to 0.")
        return 0


# ---------------------------------------------------------------------------
# Main loop — one config at a time
# ---------------------------------------------------------------------------

for SOURCE_CONFIG, SOURCE_LANGUAGE, SOURCE_COUNTRY in CONFIGS:
    LOCAL_PATH = f"{SOURCE_CONFIG}_labeled.jsonl"

    print(f"\n{'='*60}")
    print(f"Classifying cultural sensitivity: {SOURCE_CONFIG} "
          f"({SOURCE_LANGUAGE} / {SOURCE_COUNTRY})")
    print(f"{'='*60}")

    # Load the dataset split from HuggingFace
    print(f"Loading {SOURCE_CONFIG} from HuggingFace ({HF_REPO_ID})...")
    ds = load_dataset(HF_REPO_ID, SOURCE_CONFIG)

    # Use the 'test' split; fall back to whichever split exists
    split_name = "test" if "test" in ds else list(ds.keys())[0]
    data = ds[split_name]

    if MAX_SAMPLES is not None:
        data = data.select(range(min(MAX_SAMPLES, len(data))))
    print(f"Loaded {len(data)} examples from split '{split_name}'")

    # Resume from previously saved progress if the local file exists
    labeled_rows: list[dict] = []
    try:
        with open(LOCAL_PATH, "r", encoding="utf-8") as f:
            labeled_rows = [json.loads(line) for line in f if line.strip()]
        print(f"Resumed from {len(labeled_rows)} previously labeled examples")
    except FileNotFoundError:
        pass

    # Classify remaining rows (skip already-processed ones)
    start_idx = len(labeled_rows)
    for i, row in enumerate(
        tqdm(
            data.select(range(start_idx, len(data))),
            desc="Classifying",
            unit="row",
            initial=start_idx,
            total=len(data),
        )
    ):
        try:
            label = classify_row(row, SOURCE_LANGUAGE, SOURCE_COUNTRY)
            annotated = dict(row)
            annotated["claude_cs"] = label  # 1 = cultural, 0 = not cultural
            labeled_rows.append(annotated)

            # Checkpoint every 100 rows to guard against interruptions
            if (i + 1) % 100 == 0:
                with open(LOCAL_PATH, "w", encoding="utf-8") as f:
                    for r in labeled_rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"\nError at row {start_idx + i}: {e}")
            # Save progress before re-raising so the run can be resumed
            with open(LOCAL_PATH, "w", encoding="utf-8") as f:
                for r in labeled_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            raise

    # Final local save
    with open(LOCAL_PATH, "w", encoding="utf-8") as f:
        for row in labeled_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved locally to {LOCAL_PATH}")

    # Count how many questions were flagged as culturally dependent
    cs_count = sum(1 for r in labeled_rows if r.get("claude_cs") == 1)
    print(f"Cultural questions: {cs_count}/{len(labeled_rows)} "
          f"({100*cs_count/len(labeled_rows):.1f}%)")

    # Upload the annotated split — to the output repo if specified, else back
    # to the source repo under the same config name.
    output_repo = HF_OUTPUT_REPO_ID if HF_OUTPUT_REPO_ID else HF_REPO_ID
    print(f"Uploading annotated {SOURCE_CONFIG} to {output_repo}...")
    with open(LOCAL_PATH, "r", encoding="utf-8") as f:
        loaded_rows = [json.loads(line) for line in f if line.strip()]
    annotated_ds = DatasetDict({split_name: Dataset.from_list(loaded_rows)})
    annotated_ds.push_to_hub(output_repo, config_name=SOURCE_CONFIG)
    print(f"Done! https://huggingface.co/datasets/{output_repo}")
