import pandas as pd
import os

df = pd.read_csv("lite_eval_results.csv")

accuracy_cols = [c for c in df.columns if c not in ("model", "country")]

df = df.sort_values(["country", "model"]).reset_index(drop=True)

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lite_eval_results.md")

def fmt_val(x):
    try:
        return f"{float(x):.4f}"
    except (ValueError, TypeError):
        return ""

header_cols = ["Model", "Country"] + [c.replace("_", " ") for c in accuracy_cols]
separator = ["---", "---"] + ["---:"] * len(accuracy_cols)

lines = []
lines.append("# Lite Eval Accuracy Results\n")

current_country = None
for _, row in df.iterrows():
    country = row["country"]
    if country != current_country:
        lines.append(f"\n## {country}\n")
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("| " + " | ".join(separator) + " |")
        current_country = country
    values = [fmt_val(row[c]) for c in accuracy_cols]
    cells = [row["model"], row["country"]] + values
    lines.append("| " + " | ".join(cells) + " |")

output = "\n".join(lines)

lines.append("\n---\n")
lines.append("## Summary\n")

lines.append("### Models\n")
for _, row in df.iterrows():
    lines.append(f"- `{row['model']}` ({row['country']})")

lines.append("\n### Evaluation Subsets\n")
for col in accuracy_cols:
    lines.append(f"- {col.replace('_', ' ')}")

output = "\n".join(lines)

with open(OUTPUT_FILE, "w") as f:
    f.write(output + "\n")

print(f"Saved to {OUTPUT_FILE}")
print("\nModels:")
for _, row in df.iterrows():
    print(f"  {row['model']} ({row['country']})")
print("\nEvaluation Subsets:")
for col in accuracy_cols:
    print(f"  {col}")
