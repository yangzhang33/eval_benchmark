"""
Patches yangzhang33/culture-eval-benchmark-labeled with CEB_index and CEB_config
by matching rows against yangzhang33/culture-eval-benchmark (which already has
these fields). Matching is done on the 'question' field.

Only processes the specified CONFIGS (Italian by default).
"""

from datasets import load_dataset, DatasetDict

BENCHMARK_REPO_ID = "yangzhang33/culture-eval-benchmark"
LABELED_REPO_ID = "yangzhang33/culture-eval-benchmark-labeled"

CONFIGS = [
    "italic_cs",
    # "italic_ca",
]

for config_name in CONFIGS:
    print(f"\n=== {config_name} ===")

    # Build a lookup: question -> (CEB_index, CEB_config) from the benchmark
    bench_ds = load_dataset(BENCHMARK_REPO_ID, config_name)
    split_name = "test" if "test" in bench_ds else list(bench_ds.keys())[0]
    bench_data = bench_ds[split_name]

    lookup = {
        row["question"]: {"CEB_index": row["CEB_index"], "CEB_config": row["CEB_config"]}
        for row in bench_data
    }
    print(f"  Loaded {len(lookup)} entries from benchmark")

    # Load the labeled dataset and add the two fields
    labeled_ds = load_dataset(LABELED_REPO_ID, config_name)
    labeled_split_name = "test" if "test" in labeled_ds else list(labeled_ds.keys())[0]
    labeled_data = labeled_ds[labeled_split_name]

    def add_ceb_fields(row):
        match = lookup.get(row["question"])
        if match is None:
            return {"CEB_index": -1, "CEB_config": ""}
        return match

    patched = labeled_data.map(add_ceb_fields)
    missing = sum(1 for r in patched if r["CEB_index"] == -1)
    print(f"  Patched {len(patched)} rows ({missing} unmatched)")

    DatasetDict({labeled_split_name: patched}).push_to_hub(LABELED_REPO_ID, config_name=config_name)
    print(f"  Uploaded to {LABELED_REPO_ID} / {config_name}")

print(f"\nDone!")
