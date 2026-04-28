import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE = Path("cs_filtered_lite_eval_loglik_v1/all_results")
CSV_ABSTAIN   = BASE / "abstain_rate.csv"
CSV_COND_ACC  = BASE / "cond_acc.csv"
CSV_CONF_ERR  = BASE / "conf_err_rate.csv"

# ── Column layout ─────────────────────────────────────────────────────────────
# 19 columns: english_ca + (ca / cs / cs_en) × 6 languages

LANG_INFO = [
    ("english",    "EN",  None),   # only has _ca
    ("chinese",    "ZH",  "China"),
    ("arabic",     "AR",  "UAE"),
    ("greek",      "EL",  "Greece"),
    ("hindi",      "HI",  "India"),
    ("indonesian", "ID",  "Southeast Asian"),
    ("korean",     "KO",  "South Korea"),
]

# Build ordered column list and display labels
SUBSET_COLS   = []   # CSV column names
SUBSET_LABELS = []   # short display labels
GROUP_ENDS    = []   # col indices where a language group ends (for separator lines)

for lang, locale, _ in LANG_INFO:
    if lang == "english":
        SUBSET_COLS.append("english_ca")
        SUBSET_LABELS.append(f"EN\nca")
        GROUP_ENDS.append(len(SUBSET_COLS) - 1)
    else:
        for sfx, lbl in [("ca", "ca"), ("cs", "cs"), ("cs_en", "cs↑EN")]:
            SUBSET_COLS.append(f"{lang}_{sfx}")
            SUBSET_LABELS.append(f"{locale}\n{lbl}")
        GROUP_ENDS.append(len(SUBSET_COLS) - 1)

# locale label → (col_start, col_end) index range (inclusive) in SUBSET_COLS
LOCALE_COL_RANGE = {}
for lang, locale, country in LANG_INFO:
    if lang == "english":
        continue
    col_start = SUBSET_COLS.index(f"{lang}_ca")
    col_end   = SUBSET_COLS.index(f"{lang}_cs_en")
    LOCALE_COL_RANGE[locale] = (col_start, col_end)

# ── Short model names ─────────────────────────────────────────────────────────

SHORT_NAMES = {
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
    "Qwen/Qwen2.5-7B":                                  "Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct":                         "Qwen2.5-7B-IT",
    "Qwen/Qwen2.5-14B":                                 "Qwen2.5-14B",
    "Qwen/Qwen2.5-14B-Instruct":                        "Qwen2.5-14B-IT",
    "Qwen/Qwen2.5-32B":                                 "Qwen2.5-32B",
    "Qwen/Qwen2.5-32B-Instruct":                        "Qwen2.5-32B-IT",
    "deepseek-ai/deepseek-llm-7b-base":                 "DeepSeek-7B",
    "deepseek-ai/deepseek-llm-7b-chat":                 "DeepSeek-7B-Chat",
    "inceptionai/jais-13b":                             "Jais-13B",
    "inceptionai/jais-13b-chat":                        "Jais-13B-Chat",
    "inceptionai/Jais-2-8B-Chat":                       "Jais-2-8B",
    "FreedomIntelligence/AceGPT-v2-8B":                 "AceGPT-v2-8B",
    "FreedomIntelligence/AceGPT-v2-8B-Chat":            "AceGPT-v2-8B-Chat",
    "mistralai/Ministral-3-8B-Base-2512":               "Ministral-3B",
    "mistralai/Ministral-3-8B-Instruct-2512":           "Ministral-3B-IT",
    "ilsp/Llama-Krikri-8B-Instruct":                    "Krikri-8B",
    "ilsp/Meltemi-7B-Instruct-v1.5":                    "Meltemi-7B",
    "sarvamai/OpenHathi-7B-Hi-v0.1-Base":               "OpenHathi-7B",
    "krutrim-ai-labs/Krutrim-1-instruct":               "Krutrim-1",
    "aisingapore/Llama-SEA-LION-v3-8B":                 "SEA-LION-8B",
    "aisingapore/Llama-SEA-LION-v3-8B-IT":              "SEA-LION-IT",
    "SeaLLMs/SeaLLM-7B-v2.5":                          "SeaLLM-7B",
    "SeaLLMs/SeaLLMs-v3-7B":                           "SeaLLMs-v3",
    "SeaLLMs/SeaLLMs-v3-7B-Chat":                      "SeaLLMs-Chat",
    "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B":       "HyperCLOVA-8B",
    "beomi/Llama-3-Open-Ko-8B":                         "Ko-Llama-8B",
    "EleutherAI/polyglot-ko-12.8b":                     "Polyglot-Ko-12B",
    "EleutherAI/polyglot-ko-5.8b":                      "Polyglot-Ko-5B",
    "CohereLabs/aya-expanse-8b":                        "Aya-Expanse",
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

# Country → home locale label
HOME_LOCALE_LABEL = {
    "China":           "ZH",
    "UAE":             "AR",
    "Greece":          "EL",
    "India":           "HI",
    "Southeast Asian": "ID",
    "South Korea":     "KO",
}

# ── Panel definitions ─────────────────────────────────────────────────────────

cmap_wr  = LinearSegmentedColormap.from_list("wr",  ["#FFFFFF", "#C0392B"], N=256)
cmap_rwb = LinearSegmentedColormap.from_list("rwb", ["#C0392B", "#FFFFFF", "#185FA5"], N=512)

PANELS = [
    dict(
        csv        = CSV_ABSTAIN,
        title      = "Abstain Rate",
        subtitle   = "fraction of questions where model refuses to answer  ·  lower is better",
        cmap       = cmap_wr,
        vmin       = 0.0,
        vmax       = 0.15,
        cbar_label = "Abstain Rate",
        center     = None,
    ),
    dict(
        csv        = CSV_COND_ACC,
        title      = "Conditional Accuracy",
        subtitle   = "accuracy on non-abstained questions  ·  higher is better",
        cmap       = cmap_rwb,
        vmin       = 0.20,
        vmax       = 0.80,
        cbar_label = "Cond. Accuracy",
        center     = 0.50,
    ),
    dict(
        csv        = CSV_CONF_ERR,
        title      = "Confident Error Rate",
        subtitle   = "rate of confident wrong answers  ·  lower is better",
        cmap       = cmap_rwb,
        vmin       = 0.20,
        vmax       = 0.80,
        cbar_label = "Conf. Error Rate",
        center     = 0.50,
    ),
]

# ── Load data ─────────────────────────────────────────────────────────────────

def load(path):
    d = pd.read_csv(path)
    d["short"] = d["model"].map(SHORT_NAMES).fillna(d["model"])
    return d

dfs = [load(p["csv"]) for p in PANELS]

# Use first panel for model order / countries
ref       = dfs[0]
model_labels = ref["short"].tolist()
countries    = ref["country"].tolist()
n_models     = len(model_labels)
n_cols       = len(SUBSET_COLS)

# ── Figure ────────────────────────────────────────────────────────────────────

row_h  = 0.38          # inches per model row
pad_h  = 2.8           # header/footer padding per panel
hspace = 0.55          # vertical space between panels (inches fraction)

panel_h = row_h * n_models + pad_h
fig_h   = len(PANELS) * panel_h + (len(PANELS) - 1) * 0.6
fig_w   = 22

fig, axes = plt.subplots(
    len(PANELS), 1,
    figsize=(fig_w, fig_h),
    gridspec_kw={"hspace": hspace},
)
fig.patch.set_facecolor("white")

for p_idx, (ax, panel, df) in enumerate(zip(axes, PANELS, dfs)):
    matrix = df[SUBSET_COLS].values
    vmin, vmax = panel["vmin"], panel["vmax"]

    im = ax.imshow(matrix, cmap=panel["cmap"], vmin=vmin, vmax=vmax, aspect="auto")

    # ── Cell annotations ──────────────────────────────────────────────────────
    for r, country in enumerate(countries):
        home_locale = HOME_LOCALE_LABEL.get(country)
        home_range  = LOCALE_COL_RANGE.get(home_locale) if home_locale else None
        # col index of the _cs column within the home group (for ★)
        home_cs_col = home_range[0] + 1 if home_range else None

        for c in range(n_cols):
            v = matrix[r, c]
            norm_v = (v - vmin) / (vmax - vmin)
            if panel["center"] is None:
                text_color = "white" if norm_v > 0.65 else "#222"
            else:
                text_color = "white" if (norm_v > 0.72 or norm_v < 0.18) else "#222"

            is_home_cs = (c == home_cs_col)
            label = f"★{v:.2f}" if is_home_cs else f"{v:.2f}"
            ax.text(c, r, label, ha="center", va="center",
                    fontsize=6.2, fontweight="bold" if is_home_cs else "normal",
                    color=text_color)

        # Draw a single wide border spanning all 3 home locale columns
        if home_range is not None:
            col_start, col_end = home_range
            border_color = COUNTRY_COLORS.get(country, "#333")
            rect = mpatches.FancyBboxPatch(
                (col_start - 0.48, r - 0.46),
                (col_end - col_start) + 0.96, 0.92,
                boxstyle="round,pad=0.02",
                linewidth=2.0,
                edgecolor=border_color,
                facecolor="none",
                transform=ax.transData,
                zorder=3,
            )
            ax.add_patch(rect)

    # ── Language group separator lines ────────────────────────────────────────
    for g_end in GROUP_ENDS[:-1]:
        ax.axvline(g_end + 0.5, color="#aaa", linewidth=0.8, linestyle="--", zorder=4)

    # ── Locale header bands (colored background above each group) ─────────────
    for lang, locale, country in LANG_INFO:
        if lang == "english":
            col_start, col_end = 0, 0
        else:
            idx = SUBSET_COLS.index(f"{lang}_ca")
            col_start, col_end = idx, idx + 2
        color = COUNTRY_COLORS.get(country, "#999") if country else "#378ADD"
        ax.add_patch(mpatches.FancyBboxPatch(
            (col_start - 0.48, -1.45), col_end - col_start + 0.96, 0.75,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="none", alpha=0.25,
            transform=ax.transData, zorder=5, clip_on=False,
        ))
        ax.text((col_start + col_end) / 2, -1.08, locale,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color=color, transform=ax.transData, zorder=6, clip_on=False)

    # ── Axes ticks ────────────────────────────────────────────────────────────
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(SUBSET_LABELS, fontsize=7, linespacing=1.2)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", pad=14)   # push labels up to make room for bands

    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_labels, fontsize=7.5)

    # Colored country square before each model label
    for r, country in enumerate(countries):
        color = COUNTRY_COLORS.get(country, "#888")
        ax.get_yticklabels()[r].set_color("#222")
        ax.annotate("■", xy=(0, r), xytext=(-0.55, r),
                    xycoords=("axes fraction", "data"),
                    textcoords=("axes fraction", "data"),
                    ha="right", va="center", fontsize=7.5,
                    color=color, annotation_clip=False)

    ax.tick_params(left=False, top=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                        fraction=0.012, pad=0.01, aspect=30)
    cbar.set_label(panel["cbar_label"], fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)

    # ── Panel title ───────────────────────────────────────────────────────────
    ax.set_title(
        f"{panel['title']}\n{panel['subtitle']}",
        fontsize=9.5, pad=32, loc="left", color="#333",
    )

# ── Shared country legend (below last panel) ──────────────────────────────────

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
    loc="upper left",
    bbox_to_anchor=(0.0, -0.08),
    ncol=len(unique_countries),
    frameon=False,
    handlelength=1.2,
    handleheight=1.2,
)

# ── Save ──────────────────────────────────────────────────────────────────────

output_dir = BASE
pdf_path = output_dir / "metrics_heatmap.pdf"
png_path = output_dir / "metrics_heatmap.png"

plt.savefig(pdf_path, bbox_inches="tight", dpi=150)
plt.savefig(png_path, bbox_inches="tight", dpi=150)
print(f"Saved: {pdf_path} and {png_path}")
plt.show()
