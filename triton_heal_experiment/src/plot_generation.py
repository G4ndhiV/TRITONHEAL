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
    """Variación entre réplicas si existen; si no, entre tareas del benchmark."""
    df = df[df["config"].isin(CONFIGS)].copy()
    n_runs = df["run_id"].nunique() if "run_id" in df.columns else 1

    if n_runs > 1:
        sub = (
            df.groupby(["config", "run_id"])
            .agg(compile_pct=("compile_ok", lambda x: 100 * x.astype(bool).mean()))
            .reset_index()
        )
        x_col, x_label, title = "run_id", "Réplica (run_id)", "Tasa de compilación por réplica"
    else:
        sub = df.copy()
        sub["compile_pct"] = 100 * sub["compile_ok"].astype(bool)
        sub["task_num"] = (
            sub["task_id"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
        )
        x_col, x_label, title = "task_num", "Tarea (índice)", "Tasa de compilación por tarea"
        sub = sub.groupby(["config", "task_num"], as_index=False)["compile_pct"].max()

    sub["label"] = sub["config"].map(CONFIG_LABELS)
    sub = sub.sort_values(["config", x_col])

    palette = sns.color_palette("tab10", n_colors=sub["config"].nunique())
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, (config, grp) in enumerate(sub.groupby("config")):
        short = CONFIG_LABELS[config].split("(")[0].strip()
        ax.plot(
            grp[x_col],
            grp["compile_pct"],
            marker="o",
            linewidth=1.8,
            markersize=7,
            label=short,
            color=palette[i],
            alpha=0.9,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Tasa de compilación (%)")
    ax.set_title(title)
    ax.set_ylim(-5, 105)
    if n_runs <= 1:
        ax.set_xticks(sorted(sub["task_num"].unique()))
    else:
        ax.set_xticks(sorted(sub["run_id"].unique()))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
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
