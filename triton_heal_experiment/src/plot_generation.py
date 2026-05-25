#!/usr/bin/env python3
"""Figures for generation experiment."""
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


def fig_speedup_box(df: pd.DataFrame):
    sub = df[df["compile_ok"].astype(bool) & df["speedup_proxy"].notna()].copy()
    sub["label"] = sub["config"].map(CONFIG_LABELS)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=sub, x="label", y="speedup_proxy", ax=ax)
    ax.set_ylabel("Speedup proxy (baseline_ms / pipeline_ms)")
    ax.set_xlabel("Configuración")
    ax.set_title("Speedup proxy por configuración (M4, sin CUDA)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, "fig_speedup_proxy")


def fig_robustness_compile(df: pd.DataFrame):
    sub = df.groupby(["config", "run_id"]).agg(
        compile_pct=("compile_ok", lambda x: 100 * x.astype(bool).mean())
    ).reset_index()
    sub["label"] = sub["config"].map(CONFIG_LABELS)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=sub, x="run_id", y="compile_pct", hue="label", marker="o", ax=ax)
    ax.set_xlabel("Réplica (run_id)")
    ax.set_ylabel("Tasa compilación (%)")
    ax.set_title("Robustez: variación de tasa de compilación entre réplicas")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
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
