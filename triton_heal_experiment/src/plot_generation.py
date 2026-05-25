#!/usr/bin/env python3
"""Figures for generation experiment."""
import shutil
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import FIGURES_DIR, REPORT_FIGURES_DIR, RESULTS_DIR
from src.verify import CONFIG_LABELS, CONFIGS

sns.set_theme(style="whitegrid", font_scale=1.05)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, stem: str):
    for ext in ("pdf", "png"):
        p = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        shutil.copy2(p, REPORT_FIGURES_DIR / p.name)
    plt.close(fig)


def fig_compile_rate(df: pd.DataFrame):
    rows = []
    for c in CONFIGS:
        sub = df[df["config"] == c]
        rows.append({
            "label": CONFIG_LABELS[c].split("(")[0].strip(),
            "compile_pct": 100 * sub["compile_ok"].astype(bool).mean(),
        })
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=plot_df, x="label", y="compile_pct", ax=ax, color="#1565c0")
    ax.set_ylabel("Tasa de compilación (%)")
    ax.set_xlabel("Configuración")
    ax.set_title("Tasa de éxito de compilación (kernels generados, M4)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, "fig_compile_rate")


def _short_label(config: str) -> str:
    short = {
        "solo_llama": "Llama-8B",
        "solo_deepseek": "R1-14B",
        "solo_deepseek_coder": "Coder-V2",
        "solo_claude": "Claude (heur.)",
        "solo_gpt4o": "GPT-4o (heur.)",
        "triton_heal_dual": "Triton Heal",
    }
    return short.get(config, CONFIG_LABELS.get(config, config))


def fig_speedup_box(df: pd.DataFrame):
    sub = df[df["compile_ok"].astype(bool) & df["speedup_proxy"].notna()].copy()
    if sub.empty:
        return
    sub = sub[sub["speedup_proxy"] > 0]
    sub["label"] = sub["config"].map(_short_label)
    order = [_short_label(c) for c in CONFIGS if c in sub["config"].values]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.boxplot(
        data=sub,
        x="label",
        y="speedup_proxy",
        order=order,
        ax=ax,
        color="#5c6bc0",
        linewidth=1.2,
        fliersize=4,
    )
    ax.set_yscale("log")
    ax.set_ylim(sub["speedup_proxy"].min() * 0.7, sub["speedup_proxy"].max() * 1.3)
    ax.set_ylabel("Speedup proxy (baseline / pipeline), escala log")
    ax.set_xlabel("")
    ax.set_title("Speedup proxy en tareas con compilación exitosa (M4)")
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    for t in ax.get_xticklabels():
        t.set_ha("right")
    ax.yaxis.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig_speedup_proxy")


def fig_robustness_compile(df: pd.DataFrame):
    """Mapa por tarea: verde = compiló, rojo = no (evita líneas binarias superpuestas)."""
    sub = df[df["config"].isin(CONFIGS)].copy()
    sub["ok"] = sub["compile_ok"].astype(bool).astype(int)
    sub["task_num"] = sub["task_id"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)

    pivot = sub.pivot_table(index="config", columns="task_num", values="ok", aggfunc="max")
    order = [c for c in CONFIGS if c in pivot.index]
    pivot = pivot.reindex(order)
    row_labels = [CONFIG_LABELS[c].split("(")[0].strip() for c in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 3.2))
    cmap = mcolors.ListedColormap(["#e57373", "#66bb6a"])
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar_kws={
            "label": "Compilación",
            "ticks": [0.25, 0.75],
            "format": "%.0f",
        },
        linewidths=0.8,
        linecolor="white",
        annot=np.where(pivot.values.astype(int) == 1, "Sí", "No"),
        fmt="",
        annot_kws={"size": 8, "weight": "bold"},
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticklabels(["No", "Sí"], fontsize=9)

    ax.set_yticklabels(row_labels, rotation=0, fontsize=9)
    ax.set_xlabel("Tarea (índice)")
    ax.set_ylabel("")
    ax.set_title("Compilación por tarea y configuración (1 = éxito L1--L3)")
    fig.tight_layout()
    _save(fig, "fig_robustness_compile")


def main():
    path = RESULTS_DIR / "generation.csv"
    if not path.exists():
        print("No generation.csv — skip generation figures")
        return
    df = pd.read_csv(path)
    fig_compile_rate(df)
    fig_speedup_box(df)
    fig_robustness_compile(df)
    print("Generation figures saved")


if __name__ == "__main__":
    main()
