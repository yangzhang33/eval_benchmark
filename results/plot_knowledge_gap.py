import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────

CSV_PATH = Path("aresult_paper/cs_filtered_lite_eval_loglik_v1_8_tokens_cultural/all_results/accuracy_gaps.csv")
# CSV_PATH = Path("cs_filtered_lite_eval_loglik_cs_v3/merged/all_results/accuracy_gaps.csv")

LOCALES = ["ZH", "AR", "EL", "HI", "ID", "KO", "IT", "FR", "JA", "ES",
           "BN", "NL", "HE", "NE", "FA", "PL", "RU", "TE", "UK"]
LANG_PREFIXES = ["chinese", "arabic", "greek", "hindi", "indonesian", "korean", "italic", "french", "japanese", "spanish",
                 "bengali", "dutch", "hebrew", "nepali", "persian", "polish", "russian", "telugu", "ukrainian"]

# Three panels: global_gap, local_gap, knowledge_gap
PANELS = [
    dict(
        col_suffix = "global_gap",
        title      = "Global Gap",
        subtitle   = "local-lang − English  ·  global (non-locale-specific) questions",
    ),
    dict(
        col_suffix = "local_gap",
        title      = "Local Gap",
        subtitle   = "local-lang − English  ·  locale-specific questions",
    ),
    dict(
        col_suffix = "knowledge_gap",
        title      = "Knowledge Gap",
        subtitle   = "LocalGap − GlobalGap  ·  local-language query advantage",
    ),
]

SHORT_NAMES = {
    # ── english / meta llama ──────────────────────────────────────────────────
    "meta-llama/Meta-Llama-3.1-8B":                                         "Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct":                                     "Llama-3.1-8B-IT",
    "meta-llama/Llama-3.1-70B":                                             "Llama-3.1-70B",
    "meta-llama/Llama-3.1-70B-Instruct":                                    "Llama-3.1-70B-IT",
    "/datalake/datastore1/yang/_hf_models/Llama-3.2-1B":                    "Llama-3.2-1B",
    "/datalake/datastore1/yang/_hf_models/Llama-3.2-1B-Instruct":           "Llama-3.2-1B-IT",
    "/datalake/datastore1/yang/_hf_models/Llama-3.2-3B":                    "Llama-3.2-3B",
    "/datalake/datastore1/yang/_hf_models/Llama-3.2-3B-Instruct":           "Llama-3.2-3B-IT",
    # ── english / google gemma ────────────────────────────────────────────────
    "google/gemma-2-9b":                                                     "Gemma-2-9B",
    "google/gemma-2-9b-it":                                                  "Gemma-2-9B-IT",
    "google/gemma-2-27b":                                                    "Gemma-2-27B",
    "google/gemma-2-27b-it":                                                 "Gemma-2-27B-IT",
    "/datalake/datastore1/yang/_hf_models/gemma-3-270m":                     "Gemma-3-270M",
    "/datalake/datastore1/yang/_hf_models/gemma-3-270m-it":                  "Gemma-3-270M-IT",
    "/datalake/datastore1/yang/_hf_models/gemma-3-1b-pt":                    "Gemma-3-1B",
    "/datalake/datastore1/yang/_hf_models/gemma-3-1b-it":                    "Gemma-3-1B-IT",
    "/datalake/datastore1/yang/_hf_models/gemma-3-4b-pt":                    "Gemma-3-4B",
    "/datalake/datastore1/yang/_hf_models/gemma-3-4b-it":                    "Gemma-3-4B-IT",
    "google/gemma-3-12b-pt":                                                 "Gemma-3-12B",
    "google/gemma-3-12b-it":                                                 "Gemma-3-12B-IT",
    "google/gemma-3-27b-pt":                                                 "Gemma-3-27B",
    "google/gemma-3-27b-it":                                                 "Gemma-3-27B-IT",
    "/datalake/datastore1/yang/_hf_models/gemma-4-E2B":                      "Gemma-4-E2B",
    "/datalake/datastore1/yang/_hf_models/gemma-4-E2B-it":                   "Gemma-4-E2B-IT",
    "/datalake/datastore1/yang/_hf_models/gemma-4-E4B":                      "Gemma-4-E4B",
    "/datalake/datastore1/yang/_hf_models/gemma-4-E4B-it":                   "Gemma-4-E4B-IT",
    "google/gemma-4-26B-A4B":                                                "Gemma-4-26B",
    "google/gemma-4-26B-A4B-it":                                             "Gemma-4-26B-IT",
    "google/gemma-4-31B":                                                    "Gemma-4-31B",
    "google/gemma-4-31B-it":                                                 "Gemma-4-31B-IT",
    # ── chinese / qwen3 ───────────────────────────────────────────────────────
    "/datalake/datastore1/yang/_hf_models/Qwen3-0.6B":                       "Qwen3-0.6B",
    "/datalake/datastore1/yang/_hf_models/Qwen3-1.7B":                       "Qwen3-1.7B",
    "/datalake/datastore1/yang/_hf_models/Qwen3-4B":                         "Qwen3-4B",
    "/datalake/datastore1/yang/_hf_models/Qwen3-4B-Instruct-2507":           "Qwen3-4B-IT",
    "Qwen/Qwen3-14B":                                                        "Qwen3-14B",
    "Qwen/Qwen3-30B-A3B":                                                    "Qwen3-30B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507":                                      "Qwen3-30B-IT",
    # ── chinese / qwen3.5 ────────────────────────────────────────────────────
    "/datalake/datastore1/yang/_hf_models/Qwen3.5-0.8B-Base":                "Qwen3.5-0.8B",
    "/datalake/datastore1/yang/_hf_models/Qwen3.5-0.8B":                     "Qwen3.5-0.8B-IT",
    "/datalake/datastore1/yang/_hf_models/Qwen3.5-2B-Base":                  "Qwen3.5-2B",
    "/datalake/datastore1/yang/_hf_models/Qwen3.5-2B":                       "Qwen3.5-2B-IT",
    "/datalake/datastore1/yang/_hf_models/Qwen3.5-4B-Base":                  "Qwen3.5-4B",
    "/datalake/datastore1/yang/_hf_models/Qwen3.5-4B":                       "Qwen3.5-4B-IT",
    "/datalake/datastore1/yang/_hf_models/Qwen3.5-9B-Base":                  "Qwen3.5-9B",
    "/datalake/datastore1/yang/_hf_models/Qwen3.5-9B":                       "Qwen3.5-9B-IT",
    "Qwen/Qwen3.5-27B":                                                      "Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B-Base":                                             "Qwen3.5-35B",
    "Qwen/Qwen3.5-35B-A3B":                                                  "Qwen3.5-35B-IT",
    # ── chinese / qwen2.5 ────────────────────────────────────────────────────
    "Qwen/Qwen2.5-7B":                                                       "Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct":                                              "Qwen2.5-7B-IT",
    "Qwen/Qwen2.5-14B":                                                      "Qwen2.5-14B",
    "Qwen/Qwen2.5-14B-Instruct":                                             "Qwen2.5-14B-IT",
    "Qwen/Qwen2.5-32B":                                                      "Qwen2.5-32B",
    "Qwen/Qwen2.5-32B-Instruct":                                             "Qwen2.5-32B-IT",
    # ── chinese / deepseek & glm ──────────────────────────────────────────────
    "deepseek-ai/deepseek-llm-7b-base":                                      "DeepSeek-7B",
    "deepseek-ai/deepseek-llm-7b-chat":                                      "DeepSeek-7B-Chat",
    "/datalake/datastore1/yang/_hf_models/DeepSeek-V2-Lite":                 "DeepSeek-V2-Lite",
    "/datalake/datastore1/yang/_hf_models/DeepSeek-V2-Lite-Chat":            "DeepSeek-V2-Lite-Chat",
    "/datalake/datastore1/yang/_hf_models/glm-4-9b-hf":                      "GLM-4-9B",
    "/datalake/datastore1/yang/_hf_models/glm-4-9b-chat-hf":                 "GLM-4-9B-Chat",
    "/datalake/datastore1/yang/_hf_models/GLM-4.7-Flash":                    "GLM-4.7-Flash",
    # ── arabic models ─────────────────────────────────────────────────────────
    "inceptionai/jais-13b":                                                  "Jais-13B",
    "inceptionai/jais-13b-chat":                                             "Jais-13B-Chat",
    "inceptionai/Jais-2-8B-Chat":                                            "Jais-2-8B",
    "inceptionai/Jais-2-70B-Chat":                                           "Jais-2-70B",
    "FreedomIntelligence/AceGPT-v2-8B":                                      "AceGPT-v2-8B",
    "FreedomIntelligence/AceGPT-v2-8B-Chat":                                 "AceGPT-v2-8B-Chat",
    "QCRI/Fanar-2-27B-Instruct":                                             "Fanar-2-27B",
    # ── french / mistral models ───────────────────────────────────────────────
    "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Base-2512":         "Ministral-8B",
    "/datalake/datastore1/yang/_hf_models/Ministral-3-8B-Instruct-2512":     "Ministral-8B-IT",
    "mistralai/Ministral-3-3B-Base-2512":                                    "Ministral-3B",
    "mistralai/Ministral-3-3B-Instruct-2512":                                "Ministral-3B-IT",
    "mistralai/Ministral-3-8B-Base-2512":                                    "Ministral-8B",
    "mistralai/Ministral-3-8B-Instruct-2512":                                "Ministral-8B-IT",
    "mistralai/Ministral-3-14B-Base-2512":                                   "Ministral-14B",
    "mistralai/Ministral-3-14B-Instruct-2512":                               "Ministral-14B-IT",
    "mistralai/Mistral-Medium-3.5-128B":                                     "Mistral-Medium-3.5",
    "mistralai/Mistral-Small-4-119B-2603":                                   "Mistral-Small-4",
    "/datalake/datastore1/yang/_hf_models/Lucie-7B":                         "Lucie-7B",
    "/datalake/datastore1/yang/_hf_models/Gaperon-1125-8B":                  "Gaperon-8B",
    "/datalake/datastore1/yang/_hf_models/Gaperon-1125-8B-SFT":              "Gaperon-8B-SFT",
    "/datalake/datastore1/yang/_hf_models/Gaperon-1125-24B":                 "Gaperon-24B",
    "/datalake/datastore1/yang/_hf_models/Gaperon-1125-24B-SFT":             "Gaperon-24B-SFT",
    # ── european models ───────────────────────────────────────────────────────
    "/datalake/datastore1/yang/_hf_models/EuroLLM-9B-2512":                  "EuroLLM-9B",
    "/datalake/datastore1/yang/_hf_models/EuroLLM-9B-Instruct-2512":         "EuroLLM-9B-IT",
    "/datalake/datastore1/yang/_hf_models/EuroLLM-22B-2512":                 "EuroLLM-22B",
    "/datalake/datastore1/yang/_hf_models/EuroLLM-22B-Instruct-2512":        "EuroLLM-22B-IT",
    "/datalake/datastore1/yang/_hf_models/Teuken-7B-base-v0.6":              "Teuken-7B",
    "/datalake/datastore1/yang/_hf_models/Teuken-7B-instruct-v0.6":          "Teuken-7B-IT",
    # ── greek models ──────────────────────────────────────────────────────────
    "ilsp/Llama-Krikri-8B-Instruct":                                         "Krikri-8B",
    "ilsp/Meltemi-7B-Instruct-v1.5":                                         "Meltemi-7B",
    "/datalake/datastore1/yang/_hf_models/Minerva-7B-base-v1.0":             "Minerva-7B",
    # ── hindi models ──────────────────────────────────────────────────────────
    "sarvamai/OpenHathi-7B-Hi-v0.1-Base":                                    "OpenHathi-7B",
    "krutrim-ai-labs/Krutrim-1-instruct":                                    "Krutrim-1",
    # ── southeast asian models ────────────────────────────────────────────────
    "aisingapore/Llama-SEA-LION-v3-8B":                                      "SEA-LION-8B",
    "aisingapore/Llama-SEA-LION-v3-8B-IT":                                   "SEA-LION-IT",
    "SeaLLMs/SeaLLM-7B-v2.5":                                               "SeaLLM-7B",
    "SeaLLMs/SeaLLMs-v3-7B":                                                "SeaLLMs-v3",
    "SeaLLMs/SeaLLMs-v3-7B-Chat":                                           "SeaLLMs-Chat",
    # ── korean models ─────────────────────────────────────────────────────────
    "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B":                           "HyperCLOVA-8B",
    "beomi/Llama-3-Open-Ko-8B":                                              "Ko-Llama-8B",
    "EleutherAI/polyglot-ko-5.8b":                                           "Polyglot-Ko-5B",
    "EleutherAI/polyglot-ko-12.8b":                                          "Polyglot-Ko-12B",
    "/datalake/datastore1/yang/_hf_models/KORMo-10B-base":                   "KORMo-10B",
    "/datalake/datastore1/yang/_hf_models/KORMo-10B-sft":                    "KORMo-10B-SFT",
    # ── italian models ────────────────────────────────────────────────────────
    "/datalake/datastore1/yang/_hf_models/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA": "LLaMAntino-8B",
    # ── persian models ────────────────────────────────────────────────────────
    "/datalake/datastore1/yang/_hf_models/PersianMind-v1.0":                 "PersianMind",
    # ── japanese models ───────────────────────────────────────────────────────
    "/datalake/datastore1/yang/_hf_models/Qwen3-Swallow-8B-CPT-v0.2":        "Swallow-8B-CPT",
    "/datalake/datastore1/yang/_hf_models/Qwen3-Swallow-8B-SFT-v0.2":        "Swallow-8B-SFT",
    # ── polish models ─────────────────────────────────────────────────────────
    "/datalake/datastore1/yang/_hf_models/Bielik-Minitron-7B-v3.0-Instruct": "Bielik-7B",
    "/datalake/datastore1/yang/_hf_models/Bielik-11B-v3-Base-20250730":       "Bielik-11B",
    # ── multilingual models ───────────────────────────────────────────────────
    "CohereLabs/aya-expanse-8b":                                             "Aya-Expanse",
    # ── api models ────────────────────────────────────────────────────────────
    "gemini-3-flash-preview":                                                "Gemini-3-Flash",
    "gpt-5.4-mini-2026-03-17":                                               "GPT-5.4-Mini",
    "gpt-5.4-2026-03-05":                                                     "GPT-5.4",
}

# Country → home locale label (shared across all three panels)
HOME_LOCALE_LABEL = {
    "China":           "ZH",
    "UAE":             "AR",
    "Qatar":           "AR",
    "Greece":          "EL",
    "India":           "HI",
    "Southeast Asian": "ID",
    "South Korea":     "KO",
    "Italy":           "IT",
    "France":          "FR",
    "Japan":           "JA",
    "Bangladesh":      "BN",
    "Netherlands":     "NL",
    "Israel":          "HE",
    "Nepal":           "NE",
    "Iran":            "FA",
    "Poland":          "PL",
    "Russia":          "RU",
    "Ukraine":         "UK",
}

COUNTRY_COLORS = {
    "USA":             "#378ADD",
    "China":           "#D85A30",
    "UAE":             "#BA7517",
    "Qatar":           "#8D1B3D",
    "France":          "#7F77DD",
    "Greece":          "#1D9E75",
    "India":           "#639922",
    "Southeast Asian": "#5DCAA5",
    "South Korea":     "#D4537E",
    "Italy":           "#CE2B37",
    "Japan":           "#BC002D",
    "Europe":          "#2E86AB",
    "Multilingual":    "#888780",
    "Bangladesh":      "#006A4E",
    "Netherlands":     "#FF6F1F",
    "Israel":          "#7AB7E0",
    "Nepal":           "#8B0000",
    "Iran":            "#239F40",
    "Poland":          "#E2B007",
    "Russia":          "#A52A2A",
    "Ukraine":         "#FFD500",
}

# ── Load & reshape ─────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)
df["short"] = df["model"].map(SHORT_NAMES).fillna(df["model"])

# Filter to languages present in the CSV (based on the first panel's suffix)
available = [(loc, lang) for loc, lang in zip(LOCALES, LANG_PREFIXES)
             if f"{lang}_{PANELS[0]['col_suffix']}" in df.columns]
LOCALES, LANG_PREFIXES = zip(*available) if available else ([], [])
LOCALES = list(LOCALES)
LANG_PREFIXES = list(LANG_PREFIXES)

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

# ── Unified norm centered at 0 ────────────────────────────────────────────────

matrices = [build_matrix(panel) for panel in PANELS]
all_values = np.concatenate([m.flatten() for m in matrices])
all_values = all_values[~np.isnan(all_values)]
global_vmin = float(np.min(all_values))
global_vmax = float(np.max(all_values))
norm = TwoSlopeNorm(vmin=global_vmin, vcenter=0.0, vmax=global_vmax)

# ── Figure: 1 row × 3 columns ─────────────────────────────────────────────────

fig, axes = plt.subplots(
    1, 3,
    figsize=(32, 0.40 * n_models + 2.8),
    gridspec_kw={"wspace": 0.35},
)
fig.patch.set_facecolor("white")

for p_idx, (ax, panel) in enumerate(zip(axes, PANELS)):
    matrix = matrices[p_idx]

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # ── Cell annotations ──────────────────────────────────────────────────────
    for r, (model, country) in enumerate(zip(model_labels, countries)):
        home_label = HOME_LOCALE_LABEL.get(country)
        for c, loc in enumerate(LOCALES):
            v = matrix[r, c]
            is_home = (loc == home_label)

            norm_v = float(norm(v))
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

    ax.tick_params(left=False, top=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


    # ── Panel title ───────────────────────────────────────────────────────────
    ax.set_title(
        f"{panel['title']}\n{panel['subtitle']}",
        fontsize=9.5, pad=14, loc="left", color="#333",
    )

# ── Shared colorbar (below all panels) ───────────────────────────────────────

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.tolist(), orientation="horizontal",
                    fraction=0.02, pad=0.08, aspect=50, shrink=0.6)
cbar.set_label("Gap  (positive = local-language advantage over English)", fontsize=9)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_visible(False)
cbar.ax.axvline(0, color="#333", linewidth=1.0, linestyle="--")

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

output_dir = CSV_PATH.parent
pdf_path = output_dir / "gap_heatmaps.pdf"
png_path = output_dir / "gap_heatmaps.png"

plt.savefig(pdf_path, bbox_inches="tight", dpi=150)
plt.savefig(png_path, bbox_inches="tight", dpi=150)
print(f"Saved: {pdf_path} and {png_path}")
plt.show()
