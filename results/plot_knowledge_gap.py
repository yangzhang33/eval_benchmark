import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── Data ──────────────────────────────────────────────────────────────────────

CSV_PATH = "cs_filtered_lite_eval_loglik_v1/all_results/accuracy_gaps.csv"

LOCALES = ["ZH", "AR", "EL", "HI", "ID", "KO"]
LANG_PREFIXES = ["chinese", "arabic", "greek", "hindi", "indonesian", "korean"]

# Three panels: global_gap, local_gap, knowledge_gap
PANELS = [
    dict(
        col_suffix  = "global_gap",
        title       = "Global Gap",
        subtitle    = "local-lang − English  ·  global (non-locale-specific) questions",
        vmin        = -0.30,
        vmax        =  0.05,
        cbar_label  = "Global Gap",
    ),
    dict(
        col_suffix  = "local_gap",
        title       = "Local Gap",
        subtitle    = "local-lang − English  ·  locale-specific questions",
        vmin        = -0.20,
        vmax        =  0.15,
        cbar_label  = "Local Gap",
    ),
    dict(
        col_suffix  = "knowledge_gap",
        title       = "Knowledge Gap",
        subtitle    = "LocalGap − GlobalGap  ·  local-language query advantage",
        vmin        = -0.10,
        vmax        =  0.40,
        cbar_label  = "Knowledge Gap",
    ),
]

SHORT_NAMES = {
    # english models
    "meta-llama/Meta-Llama-3.1-8B":                    "Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct":                "Llama-3.1-8B-IT",
    "google/gemma-2-9b":                                "Gemma-2-9B",
    "google/gemma-2-9b-it":                             "Gemma-2-9B-IT",
    "google/gemma-2-27b":                               "Gemma-2-27B",
    "google/gemma-2-27b-it":                            "Gemma-2-27B-IT",
    "google/gemma-3-12b-pt":                            "Gemma-3-12B",
    "google/gemma-3-12b-it":                            "Gemma-3-12B-IT",
    "google/gemma-3-27b-pt":                            "Gemma-3-27B",
    "google/gemma-3-27b-it":                            "Gemma-3-27B-IT",
    # chinese models
    "Qwen/Qwen2.5-7B":                                  "Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct":                         "Qwen2.5-7B-IT",
    "Qwen/Qwen2.5-14B":                                 "Qwen2.5-14B",
    "Qwen/Qwen2.5-14B-Instruct":                        "Qwen2.5-14B-IT",
    "Qwen/Qwen2.5-32B":                                 "Qwen2.5-32B",
    "Qwen/Qwen2.5-32B-Instruct":                        "Qwen2.5-32B-IT",
    "deepseek-ai/deepseek-llm-7b-base":                 "DeepSeek-7B",
    "deepseek-ai/deepseek-llm-7b-chat":                 "DeepSeek-7B-Chat",
    # arabic models
    "inceptionai/jais-13b":                             "Jais-13B",
    "inceptionai/jais-13b-chat":                        "Jais-13B-Chat",
    "inceptionai/Jais-2-8B-Chat":                       "Jais-2-8B",
    "FreedomIntelligence/AceGPT-v2-8B":                 "AceGPT-v2-8B",
    "FreedomIntelligence/AceGPT-v2-8B-Chat":            "AceGPT-v2-8B-Chat",
    # mistral models
    "mistralai/Ministral-3-8B-Base-2512":               "Ministral-3B",
    "mistralai/Ministral-3-8B-Instruct-2512":           "Ministral-3B-IT",
    # greek models
    "ilsp/Llama-Krikri-8B-Instruct":                    "Krikri-8B",
    "ilsp/Meltemi-7B-Instruct-v1.5":                    "Meltemi-7B",
    # hindi models
    "sarvamai/OpenHathi-7B-Hi-v0.1-Base":               "OpenHathi-7B",
    "krutrim-ai-labs/Krutrim-1-instruct":               "Krutrim-1",
    # southeast asian models
    "aisingapore/Llama-SEA-LION-v3-8B":                 "SEA-LION-8B",
    "aisingapore/Llama-SEA-LION-v3-8B-IT":              "SEA-LION-IT",
    "SeaLLMs/SeaLLM-7B-v2.5":                          "SeaLLM-7B",
    "SeaLLMs/SeaLLMs-v3-7B":                           "SeaLLMs-v3",
    "SeaLLMs/SeaLLMs-v3-7B-Chat":                      "SeaLLMs-Chat",
    # korean models
    "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B":       "HyperCLOVA-8B",
    "beomi/Llama-3-Open-Ko-8B":                         "Ko-Llama-8B",
    "EleutherAI/polyglot-ko-12.8b":                     "Polyglot-Ko-12B",
    "EleutherAI/polyglot-ko-5.8b":                      "Polyglot-Ko-5B",
    # multilingual models
    "CohereLabs/aya-expanse-8b":                        "Aya-Expanse",
}

# Country → home locale label (shared across all three panels)
HOME_LOCALE_LABEL = {
    "China":           "ZH",
    "UAE":             "AR",
    "Greece":          "EL",
    "India":           "HI",
    "Southeast Asian": "ID",
    "South Korea":     "KO",
}

COUNTRY_COLORS = {
    "USA":             "#378ADD",
    "China":           "#D85A30",
    "UAE":             "#BA7517",
    "France":          "#7F77DD",
    "Greece":          "#1D9E75",
    "India":           "#639922",
    "Southeast Asian": "#5DCAA5",
    "South Korea":     "#D4537E",
    "Multilingual":    "#888780",
}

# ── Load & reshape ─────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)
df["short"] = df["model"].map(SHORT_NAMES).fillna(df["model"])

model_labels = df["short"].tolist()
countries    = df["country"].tolist()
n_models     = len(model_labels)
n_locales    = len(LOCALES)

# Build matrix for each panel
def build_matrix(panel):
    cols = [f"{lang}_{panel['col_suffix']}" for lang in LANG_PREFIXES]
    return df[cols].values

# ── Colormap ──────────────────────────────────────────────────────────────────

cmap = LinearSegmentedColormap.from_list(
    "rwb", ["#D85A30", "#FFFFFF", "#185FA5"], N=512
)

# ── Figure: 1 row × 3 columns ─────────────────────────────────────────────────

fig, axes = plt.subplots(
    1, 3,
    figsize=(32, 0.40 * n_models + 2.8),
    gridspec_kw={"wspace": 0.06},
)
fig.patch.set_facecolor("white")

for p_idx, (ax, panel) in enumerate(zip(axes, PANELS)):
    matrix = build_matrix(panel)
    vmin, vmax = panel["vmin"], panel["vmax"]

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # ── Cell annotations ──────────────────────────────────────────────────────
    for r, (model, country) in enumerate(zip(model_labels, countries)):
        home_label = HOME_LOCALE_LABEL.get(country)
        for c, loc in enumerate(LOCALES):
            v = matrix[r, c]
            is_home = (loc == home_label)

            norm_v = (v - vmin) / (vmax - vmin)
            text_color = "white" if (norm_v > 0.72 or norm_v < 0.18) else "#222"

            label = f"★{v:.2f}" if is_home else f"{v:.2f}"
            ax.text(c, r, label, ha="center", va="center",
                    fontsize=7, fontweight="bold" if is_home else "normal",
                    color=text_color)

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

    # ── Axes ticks ────────────────────────────────────────────────────────────
    ax.set_xticks(range(n_locales))
    ax.set_xticklabels(LOCALES, fontsize=10, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    if p_idx == 0:
        # Model labels + colored squares only on the leftmost panel
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(model_labels, fontsize=8)
        for r, (label, country) in enumerate(zip(model_labels, countries)):
            color = COUNTRY_COLORS.get(country, "#888")
            ax.get_yticklabels()[r].set_color("#222")
            ax.annotate(
                "■",
                xy=(0, r), xytext=(-0.72, r),
                xycoords=("axes fraction", "data"),
                textcoords=("axes fraction", "data"),
                ha="right", va="center",
                fontsize=8, color=color,
                annotation_clip=False,
            )
    else:
        ax.set_yticks([])

    ax.tick_params(left=False, top=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Per-panel colorbar ────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        fraction=0.03, pad=0.18, aspect=40)
    cbar.set_label(panel["cbar_label"], fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    # ── Panel title ───────────────────────────────────────────────────────────
    ax.set_title(
        f"{panel['title']}\n{panel['subtitle']}",
        fontsize=9.5, pad=14, loc="left", color="#333",
    )

# ── Country legend (right of last panel) ──────────────────────────────────────

unique_countries = list(dict.fromkeys(countries))
legend_handles = [
    mpatches.Patch(color=COUNTRY_COLORS.get(c, "#888"), label=c)
    for c in unique_countries
]
axes[-1].legend(
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

# ── Save ──────────────────────────────────────────────────────────────────────

plt.savefig("gap_heatmaps.pdf", bbox_inches="tight", dpi=150)
plt.savefig("gap_heatmaps.png", bbox_inches="tight", dpi=150)
print("Saved: gap_heatmaps.pdf  and  gap_heatmaps.png")
plt.show()
