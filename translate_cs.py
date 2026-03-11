"""
Translates dataset subsets (question, option_a-d) into English using AWS Bedrock Claude,
and uploads each result as {SOURCE_CONFIG}_en to the same HF dataset repo.
"""

import json
import re
import boto3
from botocore.config import Config
from datasets import load_dataset, DatasetDict, Dataset
from tqdm import tqdm

HF_REPO_ID = "yangzhang33/cultural_eval_lite"
BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

# ---------------------------------------------------------------------------
# Configure source languages/configs here
# ---------------------------------------------------------------------------
CONFIGS = [
    # ("chinese_cs",    "Chinese"),
    # ("arabic_cs",     "Arabic"),
    # ("greek_cs",      "Greek"),
    # ("hindi_cs",      "Hindi"),
    # ("indonesian_cs", "Indonesian"),
    ("korean_cs",     "Korean"),
]

# Set to a positive integer to limit translation to N samples (for testing); None = all
MAX_SAMPLES = None

FIELDS_TO_TRANSLATE = ["question", "option_a", "option_b", "option_c", "option_d"]

config = Config(read_timeout=1000)
brt = boto3.client(service_name="bedrock-runtime", region_name="us-west-2", config=config)


def extract_json_from_text(text: str) -> str:
    """Extract JSON from text, handling markdown code blocks."""
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return text.strip()


def sanitize_json_quotes(json_str: str) -> str:
    """Replace unescaped double quotes inside JSON string values with single quotes.

    Uses a state machine: when inside a JSON string, a '"' is only treated as the
    closing delimiter if the next non-whitespace character is ':', ',', '}', or ']'.
    Any other '"' is an unescaped internal quote and gets replaced with "'".
    """
    result = []
    i = 0
    n = len(json_str)

    while i < n:
        c = json_str[i]
        if c != '"':
            result.append(c)
            i += 1
            continue

        # Opening quote of a JSON string
        result.append(c)
        i += 1
        while i < n:
            c = json_str[i]
            if c == '\\':
                result.append(c)
                i += 1
                if i < n:
                    result.append(json_str[i])
                    i += 1
            elif c == '"':
                # Look ahead past whitespace to decide if this closes the string
                j = i + 1
                while j < n and json_str[j] in ' \t\n\r':
                    j += 1
                if j >= n or json_str[j] in ':,}]':
                    result.append(c)
                    i += 1
                    break
                else:
                    result.append("'")
                    i += 1
            elif c == '\n':
                result.append('\\n')
                i += 1
            elif c == '\r':
                result.append('\\r')
                i += 1
            elif c == '\t':
                result.append('\\t')
                i += 1
            else:
                result.append(c)
                i += 1

    return ''.join(result)


def invoke_claude(prompt):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
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


def translate_row(row: dict, source_language: str) -> dict:
    """Translate all MCQ fields in a single row into English."""
    fields_text = "\n".join(f"{f}: {row[f]}" for f in FIELDS_TO_TRANSLATE)
    prompt = (
        f"Translate the following multiple-choice question fields from {source_language} to English. "
        "Return a JSON object with exactly these keys: question, option_a, option_b, option_c, option_d. "
        "Do not add explanations or extra keys.\n\n"
        f"{fields_text}"
    )

    content = invoke_claude(prompt)
    json_str = extract_json_from_text(content)

    try:
        translated = json.loads(json_str)
    except json.JSONDecodeError:
        json_str_sanitized = sanitize_json_quotes(json_str)
        try:
            translated = json.loads(json_str_sanitized)
        except json.JSONDecodeError as e:
            print(f"\nERROR: Failed to parse JSON response:")
            print(f"Raw content: {content}...")
            print(f"Extracted JSON string: {json_str}...")
            print(f"Sanitized JSON string: {json_str_sanitized}...")
            print(f"JSON Error: {e}")
            raise

    result = dict(row)
    for f in FIELDS_TO_TRANSLATE:
        result[f] = translated[f]
    return result


# ---------------------------------------------------------------------------
# Main loop over all configs
# ---------------------------------------------------------------------------
for SOURCE_CONFIG, SOURCE_LANGUAGE in CONFIGS:
    LOCAL_PATH = f"{SOURCE_CONFIG}_en.jsonl"
    OUTPUT_CONFIG = f"{SOURCE_CONFIG}_en"

    print(f"\n{'='*60}")
    print(f"Processing {SOURCE_CONFIG} ({SOURCE_LANGUAGE} -> English)")
    print(f"{'='*60}")

    # Load source test split
    print(f"Loading {SOURCE_CONFIG} from HuggingFace...")
    ds = load_dataset(HF_REPO_ID, SOURCE_CONFIG)
    test_data = ds["test"]
    if MAX_SAMPLES is not None:
        test_data = test_data.select(range(min(MAX_SAMPLES, len(test_data))))
    print(f"Loaded {len(test_data)} examples")

    # Load existing progress if file exists
    translated_rows = []
    try:
        with open(LOCAL_PATH, "r", encoding="utf-8") as f:
            translated_rows = [json.loads(line) for line in f if line.strip()]
        print(f"Resumed from {len(translated_rows)} previously translated examples")
    except FileNotFoundError:
        pass

    # Translate remaining rows
    start_idx = len(translated_rows)
    for i, row in enumerate(tqdm(test_data.select(range(start_idx, len(test_data))),
                                  desc="Translating", unit="row", initial=start_idx, total=len(test_data))):
        try:
            translated_row = translate_row(row, SOURCE_LANGUAGE)
            translated_rows.append(translated_row)

            if (i + 1) % 100 == 0:
                with open(LOCAL_PATH, "w", encoding="utf-8") as f:
                    for r in translated_rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"\nError at row {start_idx + i}: {e}")
            with open(LOCAL_PATH, "w", encoding="utf-8") as f:
                for r in translated_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            raise

    # Final save locally
    with open(LOCAL_PATH, "w", encoding="utf-8") as f:
        for row in translated_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved locally to {LOCAL_PATH}")

    # Upload
    print(f"Uploading {OUTPUT_CONFIG} to {HF_REPO_ID}...")
    with open(LOCAL_PATH, "r", encoding="utf-8") as f:
        loaded_rows = [json.loads(line) for line in f if line.strip()]
    reloaded_ds = DatasetDict({"test": Dataset.from_list(loaded_rows)})
    reloaded_ds.push_to_hub(HF_REPO_ID, config_name=OUTPUT_CONFIG)
    print(f"Done! https://huggingface.co/datasets/{HF_REPO_ID}")
