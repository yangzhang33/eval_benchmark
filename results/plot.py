import os
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Optional but nice-looking; remove if seaborn is unavailable.
import seaborn as sns


# =========================
# Config
# =========================
CSV_PATH = "lite_eval_results_v1_filtered.csv"   # change if needed
OUTDIR = Path("paper_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Locales present in the CSV
LOCALES = ["arabic", "chinese", "greek", "hindi", "indonesian", "korean"]

# Country/ecosystem colors
PALETTE = {
    "China": "#d62728",
    "Greece": "#1f77b4",
    "India": "#2ca02c",
    "South Korea": "#9467bd",
    "Southeast Asian": "#ff7f0e",
    "UAE": "#8c564b",
    "USA": "#7f7f7f",
    "Multilingual": "#17becf",
}

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["figure.dpi"] = 180
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


# =========================
# Load + reshape
# =========================
df = pd.read_csv(CSV_PATH)

# Basic safety checks
required_cols = {"model", "country", "english_ca"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for loc in LOCALES:
    for suffix in ["_ca", "_cs", "_cs_en"]:
        col = f"{loc}{suffix}"
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

# Compute gap metrics
for loc in LOCALES:
    df[f"{loc}_GlobalGap"] = df[f"{loc}_ca"] - df["english_ca"]
    df[f"{loc}_LocalGap"] = df[f"{loc}_cs"] - df[f"{loc}_cs_en"]
    df[f"{loc}_KnowledgeGap"] = df[f"{loc}_LocalGap"] - df[f"{loc}_GlobalGap"]

# Long-form table for plotting
rows = []
for _, r in df.iterrows():
    for loc in LOCALES:
        rows.append({
            "model": r["model"],
            "country": r["country"],
            "locale": loc,
            "english_ca": r["english_ca"],
            "local_ca": r[f"{loc}_ca"],
            "local_cs": r[f"{loc}_cs"],
            "cs_en": r[f"{loc}_cs_en"],
            "GlobalGap": r[f"{loc}_GlobalGap"],
            "LocalGap": r[f"{loc}_LocalGap"],
            "KnowledgeGap": r[f"{loc}_KnowledgeGap"],
        })
long_df = pd.DataFrame(rows)

# Pretty locale names
locale_name = {
    "arabic": "Arabic",
    "chinese": "Chinese",
    "greek": "Greek",
    "hindi": "Hindi",
    "indonesian": "Indonesian",
    "korean": "Korean",
}
long_df["locale_pretty"] = long_df["locale"].map(locale_name)

# Model labels (shorten for figures)
def short_model_name(x: str) -> str:
    x = x.split("/")[-1]
    x = x.replace("Meta-Llama-", "Llama-")
    x = x.replace("-Instruction", "-it")
    x = x.replace("-Instruct", "-it")
    x = x.replace("-Chat", "-chat")
    x = x.replace("-Base", "-base")
    x = x.replace("-v1.5", "")
    return x

df["model_short"] = df["model"].map(short_model_name)
long_df["model_short"] = long_df["model"].map(short_model_name)

# Optional: more compact wrapped labels
def wrap_text(s, width=14):
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False))

long_df["model_wrap"] = long_df["model_short"].apply(wrap_text)
df["model_wrap"] = df["model_short"].apply(wrap_text)


# =========================
# Figure 1
# LocalGap vs GlobalGap scatter by locale
# Why useful:
# - If points lie above the diagonal, cultural knowledge is more language-sensitive
# - Directly visualizes "beyond general language proficiency"
# =========================
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
axes = axes.flatten()

xlim = (
    long_df["GlobalGap"].min() - 0.05,
    long_df["GlobalGap"].max() + 0.05,
)
ylim = (
    long_df["LocalGap"].min() - 0.05,
    long_df["LocalGap"].max() + 0.05,
)

for ax, loc in zip(axes, LOCALES):
    sub = long_df[long_df["locale"] == loc].copy()

    # Scatter points
    for country, g in sub.groupby("country"):
        ax.scatter(
            g["GlobalGap"],
            g["LocalGap"],
            s=55,
            alpha=0.9,
            color=PALETTE.get(country, "black"),
            edgecolor="white",
            linewidth=0.6,
            label=country,
        )

    # Diagonal: LocalGap == GlobalGap
    diag_min = min(xlim[0], ylim[0])
    diag_max = max(xlim[1], ylim[1])
    ax.plot([diag_min, diag_max], [diag_min, diag_max], ls="--", lw=1.2, c="black", alpha=0.7)

    # Horizontal/vertical zero lines
    ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
    ax.axvline(0, color="gray", lw=0.8, alpha=0.6)

    # Mean point
    ax.scatter(
        sub["GlobalGap"].mean(),
        sub["LocalGap"].mean(),
        marker="X",
        s=140,
        color="black",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )

    ax.set_title(locale_name[loc])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Axis labels only on outer axes
for ax in axes[3:]:
    ax.set_xlabel("GlobalGap = local CA − English CA")
for ax in axes[::3]:
    ax.set_ylabel("LocalGap = local CS − translated-English CS")

# Shared legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=k,
           markerfacecolor=v, markersize=8, markeredgecolor="white", markeredgewidth=0.8)
    for k, v in PALETTE.items() if k in set(long_df["country"])
]
legend_elements.append(
    Line2D([0], [0], marker='X', color='w', label='Locale mean',
           markerfacecolor='black', markersize=10, markeredgecolor="white", markeredgewidth=0.8)
)

fig.legend(
    handles=legend_elements,
    loc="upper center",
    ncol=min(len(legend_elements), 5),
    bbox_to_anchor=(0.5, 1.02),
    frameon=False,
)

fig.suptitle("Local language sensitivity exceeds general language sensitivity in many models", y=1.08, fontsize=14)
plt.tight_layout()
plt.savefig(OUTDIR / "fig1_localgap_vs_globalgap_scatter.png", bbox_inches="tight")
plt.close()


# =========================
# Figure 2
# KnowledgeGap heatmap (models x locales)
# Why useful:
# - Immediate overview of which models/locales show strongest effects
# - Great appendix/main paper figure depending on size
# =========================
kg_cols = [f"{loc}_KnowledgeGap" for loc in LOCALES]
heat = df[["model_wrap", "country"] + kg_cols].copy()

# Sort by ecosystem then average KnowledgeGap
heat["kg_mean"] = heat[kg_cols].mean(axis=1)
heat = heat.sort_values(["country", "kg_mean"], ascending=[True, False])

heatmap_df = heat.set_index("model_wrap")[kg_cols]
heatmap_df.columns = [locale_name[c.replace("_KnowledgeGap", "")] for c in heatmap_df.columns]

fig_h = max(8, 0.45 * len(heatmap_df))
fig, ax = plt.subplots(figsize=(9, fig_h))
sns.heatmap(
    heatmap_df,
    cmap="RdBu_r",
    center=0,
    linewidths=0.4,
    linecolor="white",
    cbar_kws={"label": "KnowledgeGap = LocalGap − GlobalGap"},
    ax=ax,
    annot=False,
)
ax.set_title("KnowledgeGap across models and locales")
ax.set_xlabel("Locale")
ax.set_ylabel("Model")
plt.tight_layout()
plt.savefig(OUTDIR / "fig2_knowledgegap_heatmap.png", bbox_inches="tight")
plt.close()


# =========================
# Figure 3
# KnowledgeGap grouped by ecosystem and locale
# Why useful:
# - Directly supports the "ecosystem-dependent" claim
# - Good main-paper summary figure
# =========================
# Order ecosystems by overall mean KnowledgeGap
eco_order = (
    long_df.groupby("country")["KnowledgeGap"]
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=long_df,
    x="locale_pretty",
    y="KnowledgeGap",
    hue="country",
    hue_order=eco_order,
    palette=PALETTE,
    linewidth=1.0,
    fliersize=0,
    ax=ax,
)
sns.stripplot(
    data=long_df,
    x="locale_pretty",
    y="KnowledgeGap",
    hue="country",
    hue_order=eco_order,
    dodge=True,
    palette=PALETTE,
    edgecolor="white",
    linewidth=0.4,
    size=4,
    alpha=0.75,
    ax=ax,
)

# Clean duplicated legend
handles, labels = ax.get_legend_handles_labels()
n = len(eco_order)
ax.legend(handles[:n], labels[:n], title="Ecosystem", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.8)
ax.set_title("KnowledgeGap varies systematically across regional model ecosystems")
ax.set_xlabel("Locale")
ax.set_ylabel("KnowledgeGap")
plt.tight_layout()
plt.savefig(OUTDIR / "fig3_knowledgegap_by_ecosystem_boxplot.png", bbox_inches="tight")
plt.close()


# =========================
# Figure 4
# Top/bottom models for each locale by KnowledgeGap
# Why useful:
# - Good qualitative main-text figure if you want a more narrative presentation
# - Helps highlight strongest cases and counterexamples
# =========================
summary_rows = []
for loc in LOCALES:
    tmp = long_df[long_df["locale"] == loc].copy()
    top = tmp.nlargest(3, "KnowledgeGap")
    bottom = tmp.nsmallest(3, "KnowledgeGap")
    pick = pd.concat([top, bottom], axis=0).drop_duplicates(subset=["model"])
    pick["locale_pretty"] = locale_name[loc]
    summary_rows.append(pick)

summary_df = pd.concat(summary_rows, axis=0)

# Sort within each locale
summary_df["rank_key"] = summary_df.groupby("locale")["KnowledgeGap"].rank(method="first", ascending=False)
summary_df = summary_df.sort_values(["locale_pretty", "KnowledgeGap"], ascending=[True, False])

g = sns.catplot(
    data=summary_df,
    x="KnowledgeGap",
    y="model_wrap",
    hue="country",
    col="locale_pretty",
    kind="bar",
    sharex=False,
    sharey=False,
    height=4.8,
    aspect=0.82,
    palette=PALETTE,
)
g.set_titles("{col_name}")
g.set_axis_labels("KnowledgeGap", "Model")
for ax in g.axes.flatten():
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.8)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Representative strongest and weakest KnowledgeGap models per locale")
plt.savefig(OUTDIR / "fig4_top_bottom_models_by_locale.png", bbox_inches="tight")
plt.close()


# =========================
# Figure 5
# Locale-level average bars: GlobalGap vs LocalGap
# Why useful:
# - Very compact summary of the main claim
# - Easy to explain in intro/results
# =========================
mean_df = long_df.groupby("locale_pretty")[["GlobalGap", "LocalGap", "KnowledgeGap"]].mean().reset_index()
mean_df = mean_df.sort_values("KnowledgeGap", ascending=False)

plot_df = mean_df.melt(
    id_vars="locale_pretty",
    value_vars=["GlobalGap", "LocalGap"],
    var_name="metric",
    value_name="value"
)

metric_labels = {
    "GlobalGap": "GlobalGap\n(local CA − English CA)",
    "LocalGap": "LocalGap\n(local CS − translated-English CS)",
}

fig, ax = plt.subplots(figsize=(10, 5.5))
sns.barplot(
    data=plot_df,
    x="locale_pretty",
    y="value",
    hue="metric",
    palette={"GlobalGap": "#9ecae1", "LocalGap": "#fb6a4a"},
    ax=ax,
)
ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.8)
ax.set_xlabel("Locale")
ax.set_ylabel("Mean gap across models")
ax.set_title("Culture-sensitive knowledge is generally more language-dependent than culture-agnostic knowledge")

# Replace legend labels
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, [metric_labels.get(x, x) for x in labels], title=None, frameon=False)

plt.tight_layout()
plt.savefig(OUTDIR / "fig5_mean_localgap_vs_globalgap.png", bbox_inches="tight")
plt.close()


# =========================
# Figure 6
# Per-model profile lines across locales (optional appendix figure)
# Why useful:
# - Shows if a model has a stable cross-locale pattern
# - Can be useful in appendix for detailed inspection
# =========================
# Sort models by mean KnowledgeGap
model_order = (
    long_df.groupby("model_wrap")["KnowledgeGap"]
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)

fig, ax = plt.subplots(figsize=(13, 7))
for model in model_order:
    sub = long_df[long_df["model_wrap"] == model].copy()
    country = sub["country"].dropna().iloc[0]
    sub = sub.set_index("locale_pretty").reindex([locale_name[l] for l in LOCALES]).reset_index()
    ax.plot(
        sub["locale_pretty"],
        sub["KnowledgeGap"],
        marker="o",
        markersize=4,
        linewidth=1.4,
        alpha=0.7,
        color=PALETTE.get(country, "black"),
    )

ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.8)
ax.set_xlabel("Locale")
ax.set_ylabel("KnowledgeGap")
ax.set_title("Per-model KnowledgeGap profiles across locales")

# Custom legend without duplicates
legend_elements = [
    Line2D([0], [0], color=v, lw=2, label=k)
    for k, v in PALETTE.items() if k in set(long_df["country"])
]
ax.legend(handles=legend_elements, title="Ecosystem", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

plt.tight_layout()
plt.savefig(OUTDIR / "fig6_model_profiles_knowledgegap.png", bbox_inches="tight")
plt.close()


# =========================
# Also save a tidy analysis table
# =========================
# Wide summary table for paper/report writing
summary = long_df.pivot_table(
    index=["model", "country"],
    columns="locale",
    values=["GlobalGap", "LocalGap", "KnowledgeGap"],
)
summary = summary.sort_index(axis=1)
summary.to_csv(OUTDIR / "gap_summary_by_model_locale.csv")

# Ecosystem summary
eco_summary = (
    long_df.groupby(["country", "locale_pretty"])[["GlobalGap", "LocalGap", "KnowledgeGap"]]
    .mean()
    .reset_index()
    .sort_values(["country", "locale_pretty"])
)
eco_summary.to_csv(OUTDIR / "gap_summary_by_ecosystem_locale.csv", index=False)

print(f"Saved figures to: {OUTDIR.resolve()}")
for p in sorted(OUTDIR.glob("*.png")):
    print(" -", p.name)
for p in sorted(OUTDIR.glob("*.csv")):
    print(" -", p.name)