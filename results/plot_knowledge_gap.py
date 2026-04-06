import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

# ── Data ──────────────────────────────────────────────────────────────────────

CSV_PATH = "accuracy_gaps_-_accuracy_gaps.csv"   # change if needed

LOCALE_COLS = {
    "chinese_knowledge_gap":   "ZH",
    "arabic_knowledge_gap":    "AR",
    "greek_knowledge_gap":     "EL",
    "hindi_knowledge_gap":     "HI",
    "indonesian_knowledge_gap":"ID",
    "korean_knowledge_gap":    "KO",
}

SHORT_NAMES = {
    "meta-llama/Meta-Llama-3.1-8B":            "Llama-3.1-8B",
    "google/gemma-2-9b":                        "Gemma-2-9B",
    "Qwen/Qwen2.5-7B":                          "Qwen2.5-7B",
    "deepseek-ai/deepseek-llm-7b-base":         "DeepSeek-7B",
    "inceptionai/Jais-2-8B-Chat":               "Jais-2-8B",
    "FreedomIntelligence/AceGPT-v2-8B-Chat":    "AceGPT-v2-8B",
    "mistralai/Ministral-3-8B-Base-2512":       "Ministral-3B",
    "ilsp/Llama-Krikri-8B-Instruct":            "Krikri-8B",
    "ilsp/Meltemi-7B-Instruct-v1.5":            "Meltemi-7B",
    "sarvamai/OpenHathi-7B-Hi-v0.1-Base":       "OpenHathi-7B",
    "aisingapore/Llama-SEA-LION-v3-8B":         "SEA-LION-8B",
    "aisingapore/Llama-SEA-LION-v3-8B-IT":      "SEA-LION-IT",
    "SeaLLMs/SeaLLM-7B-v2.5":                  "SeaLLM-7B",
    "SeaLLMs/SeaLLMs-v3-7B":                   "SeaLLMs-v3",
    "SeaLLMs/SeaLLMs-v3-7B-Chat":              "SeaLLMs-Chat",
    "beomi/Llama-3-Open-Ko-8B":                 "Ko-Llama-8B",
    "EleutherAI/polyglot-ko-12.8b":             "Polyglot-Ko",
    "CohereLabs/aya-expanse-8b":                "Aya-Expanse",
}

# Country → home locale column
HOME_LOCALE = {
    "China":     "chinese_knowledge_gap",
    "UAE":       "arabic_knowledge_gap",
    "Greece":    "greek_knowledge_gap",
    "India":     "hindi_knowledge_gap",
    "SE Asia":   "indonesian_knowledge_gap",
    "S. Korea":  "korean_knowledge_gap",
}

# Country display colors
COUNTRY_COLORS = {
    "USA":          "#378ADD",
    "China":        "#D85A30",
    "UAE":          "#BA7517",
    "France":       "#7F77DD",
    "Greece":       "#1D9E75",
    "India":        "#639922",
    "SE Asia":      "#5DCAA5",
    "S. Korea":     "#D4537E",
    "Multilingual": "#888780",
}

# ── Load & reshape ─────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)
df["short"] = df["model"].map(SHORT_NAMES).fillna(df["model"])

kg_df = df[["short", "country"] + list(LOCALE_COLS.keys())].copy()
kg_df = kg_df.rename(columns=LOCALE_COLS)
locale_labels = list(LOCALE_COLS.values())

# Matrix: rows = models, cols = locales
matrix = kg_df[locale_labels].values          # shape (18, 6)
model_labels = kg_df["short"].tolist()
countries    = kg_df["country"].tolist()

# ── Colormap: red → white → blue ──────────────────────────────────────────────

cmap = LinearSegmentedColormap.from_list(
    "rwb",
    ["#D85A30", "#FFFFFF", "#185FA5"],
    N=512
)
VMIN, VMAX = -0.10, 0.40

# ── Figure ────────────────────────────────────────────────────────────────────

n_models  = len(model_labels)
n_locales = len(locale_labels)

fig, ax = plt.subplots(figsize=(9, 0.55 * n_models + 2.2))
fig.patch.set_facecolor("white")

im = ax.imshow(matrix, cmap=cmap, vmin=VMIN, vmax=VMAX, aspect="auto")

# ── Cell annotations ──────────────────────────────────────────────────────────

for r, (model, country) in enumerate(zip(model_labels, countries)):
    home_col = HOME_LOCALE.get(country)
    home_locale_label = LOCALE_COLS.get(home_col) if home_col else None

    for c, loc in enumerate(locale_labels):
        v = matrix[r, c]
        is_home = (loc == home_locale_label)

        # Text color: white on dark cells, dark on light cells
        norm_v = (v - VMIN) / (VMAX - VMIN)
        text_color = "white" if (norm_v > 0.72 or norm_v < 0.18) else "#222"

        label = f"★{v:.2f}" if is_home else f"{v:.2f}"
        txt = ax.text(c, r, label, ha="center", va="center",
                      fontsize=8.5, fontweight="bold" if is_home else "normal",
                      color=text_color)

        # Highlight home cell with a colored border rectangle
        if is_home:
            border_color = COUNTRY_COLORS.get(country, "#333")
            rect = mpatches.FancyBboxPatch(
                (c - 0.48, r - 0.46), 0.96, 0.92,
                boxstyle="round,pad=0.02",
                linewidth=2.0,
                edgecolor=border_color,
                facecolor="none",
                transform=ax.transData,
                zorder=3,
            )
            ax.add_patch(rect)

# ── Axes labels ───────────────────────────────────────────────────────────────

ax.set_xticks(range(n_locales))
ax.set_xticklabels(locale_labels, fontsize=11, fontweight="bold")
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")

ax.set_yticks(range(n_models))
ax.set_yticklabels(model_labels, fontsize=9.5)

# Colored country dot before each model label
for r, (label, country) in enumerate(zip(model_labels, countries)):
    color = COUNTRY_COLORS.get(country, "#888")
    ax.get_yticklabels()[r].set_color("#222")
    # Draw a colored square to the left of the tick labels
    ax.annotate(
        "■",
        xy=(0, r),
        xytext=(-0.85, r),
        xycoords=("axes fraction", "data"),
        textcoords=("axes fraction", "data"),
        ha="right", va="center",
        fontsize=10,
        color=color,
        annotation_clip=False,
    )

ax.tick_params(left=False, top=False)
for spine in ax.spines.values():
    spine.set_visible(False)

# ── Colorbar ──────────────────────────────────────────────────────────────────

cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                    fraction=0.03, pad=0.18, aspect=40)
cbar.set_label("KnowledgeGap  (LocalGap − GlobalGap)", fontsize=9)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_visible(False)

# ── Country legend ────────────────────────────────────────────────────────────

unique_countries = list(dict.fromkeys(countries))   # preserve order
legend_handles = [
    mpatches.Patch(color=COUNTRY_COLORS.get(c, "#888"), label=c)
    for c in unique_countries
]
ax.legend(
    handles=legend_handles,
    title="Model origin",
    title_fontsize=8,
    fontsize=8,
    loc="lower left",
    bbox_to_anchor=(1.01, 0.0),
    frameon=False,
    handlelength=1.2,
    handleheight=1.2,
)

# ── Title & note ──────────────────────────────────────────────────────────────

ax.set_title(
    "Knowledge Gap Heatmap  —  local-language query advantage per locale\n"
    "★ marks each model's home locale  ·  blue = local language helps  ·  red = English outperforms",
    fontsize=9.5, pad=14, loc="left", color="#333"
)

plt.tight_layout()
plt.savefig("knowledge_gap_heatmap.pdf", bbox_inches="tight", dpi=150)
plt.savefig("knowledge_gap_heatmap.png", bbox_inches="tight", dpi=150)
print("Saved: knowledge_gap_heatmap.pdf  and  knowledge_gap_heatmap.png")
plt.show()
