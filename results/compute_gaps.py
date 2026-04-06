import pandas as pd

df = pd.read_csv("lite_eval_results_loglik_v1_5_filtered.csv")

languages = ["chinese", "arabic", "greek", "hindi", "indonesian", "korean"]

rows = []
for _, row in df.iterrows():
    entry = {"model": row["model"], "country": row["country"]}
    for lang in languages:
        global_gap = row[f"{lang}_ca"] - row["english_ca"]
        local_gap = row[f"{lang}_cs"] - row[f"{lang}_cs_en"]
        knowledge_gap = local_gap - global_gap
        entry[f"{lang}_global_gap"] = round(global_gap, 4)
        entry[f"{lang}_local_gap"] = round(local_gap, 4)
        entry[f"{lang}_knowledge_gap"] = round(knowledge_gap, 4)
    rows.append(entry)

out = pd.DataFrame(rows)
out.to_csv("gap_results_loglik_v1_5.csv", index=False)
print(out.to_string())
