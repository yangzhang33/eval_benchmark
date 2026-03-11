import pandas as pd

df = pd.read_csv("accuracy_results.csv")

col_order = [
    "chinese_ca",
    "greek_ca",
    "english_ca",
    "chinese_cs",
    "chinese_cs_en",
    "greek_cs",
    "greek_cs_en",
]

pivot = df.pivot(index="model", columns="subset", values="accuracy")[col_order]
pivot.index.name = "model"
pivot.columns.name = None

pivot = pivot.applymap(lambda x: f"{x:.4f}")

print(pivot.to_string())
