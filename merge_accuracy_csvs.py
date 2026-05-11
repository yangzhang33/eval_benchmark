import pandas as pd

file1 = "results/cs_filtered_lite_eval_loglik_include_v1/all_results/accuracy.csv"
file2 = "results/cs_filtered_lite_eval_loglik_v1/all_results/accuracy.csv"
output = "results/cs_filtered_lite_eval_loglik_include_v1/all_results/accuracy_merged.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Merge new columns from df1 into df2 (df2 as base), drop duplicate english_ca
df1_extra = df1.drop(columns=["english_ca"])
merged = df2.merge(df1_extra, on=["model", "country"], how="left")

# Reorder: model, country, then groups of x_ca, x_cs, x_cs_en
key_cols = ["model", "country", "english_ca"]
other_cols = [c for c in merged.columns if c not in key_cols]

# Extract language prefixes and sort into triplets
def get_prefix(col):
    return col.rsplit("_", 1)[0]  # e.g. "french_ca" -> "french"

prefixes = list(dict.fromkeys(get_prefix(c) for c in other_cols))

ordered_cols = key_cols.copy()
for prefix in prefixes:
    for suffix in ["ca", "cs", "cs_en"]:
        col = f"{prefix}_{suffix}"
        if col in merged.columns:
            ordered_cols.append(col)

merged = merged[ordered_cols]

merged.to_csv(output, index=False)
print(f"Saved to {output}")
print(f"Columns: {list(merged.columns)}")
print(f"Rows: {len(merged)}")
