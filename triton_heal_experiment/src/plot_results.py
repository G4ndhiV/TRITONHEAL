#!/usr/bin/env python3
"""Generate figures for LaTeX report."""
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import msr_from_df
from src.paths import FIGURES_DIR, REPORT_FIGURES_DIR, RESULTS_DIR
from src.verify import CONFIG_LABELS, CONFIGS

sns.set_theme(style="whitegrid", font_scale=1.05)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, stem: str) -> None:
    """Save PDF+PNG for LaTeX (PNG embeds reliably in Tectonic/XeTeX)."""
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    png_path = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    shutil.copy2(pdf_path, REPORT_FIGURES_DIR / pdf_path.name)
    shutil.copy2(png_path, REPORT_FIGURES_DIR / png_path.name)


def fig_msr_f1(df: pd.DataFrame):
    sub = df[df["run_id"] == 0].drop_duplicates(["kernel_id", "config"])
    labels, msrs, f1s = [], [], []
    for c in CONFIGS:
        s = sub[sub["config"] == c]
        labels.append(CONFIG_LABELS[c].split("(")[0].strip())
        msrs.append(msr_from_df(s) * 100)
        from src.metrics import f1_from_df
        f1s.append(f1_from_df(s))
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - w / 2 for i in x], msrs, width=w, label="MSR (%)", color="#1f4e79")
    ax.bar([i + w / 2 for i in x], [f * 100 for f in f1s], width=w, label="F1×100", color="#5b9bd5")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Porcentaje / escala")
    ax.set_title("MSR y F1 por configuración")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    _save_fig(fig, "fig_msr_f1")
    plt.close(fig)


def fig_latency_box(df: pd.DataFrame):
    sub = df.groupby(["config", "kernel_id"]).agg(latency_ms=("latency_ms", "median")).reset_index()
    sub["label"] = sub["config"].map(CONFIG_LABELS)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=sub, x="label", y="latency_ms", ax=ax)
    ax.set_yscale("log")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_ylabel("Latencia (ms, escala log)")
    ax.set_title("Distribución de latencia de inferencia")
    fig.tight_layout()
    _save_fig(fig, "fig_latency")
    plt.close(fig)


def fig_msr_by_category(df: pd.DataFrame):
    sub = df[df["run_id"] == 0].drop_duplicates(["kernel_id", "config"])
    sub = sub[sub["label"] == "unsafe"]
    rows = []
    for c in CONFIGS:
        for cat in sub["category"].unique():
            s = sub[(sub["config"] == c) & (sub["category"] == cat)]
            if len(s) == 0:
                continue
            det = (~s["pred_safe"]).sum()
            rows.append({
                "config": CONFIG_LABELS[c][:12],
                "category": cat,
                "detection_rate": det / len(s) * 100,
            })
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="category", y="detection_rate", hue="config", ax=ax)
    ax.set_ylabel("Tasa detección unsafe (%)")
    ax.set_title("Detección por categoría de kernel (solo unsafe)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save_fig(fig, "fig_msr_category")
    plt.close(fig)


def fig_valid_json(df: pd.DataFrame):
    rows = []
    for c in CONFIGS:
        sub = df[df["config"] == c]
        if "valid_json" not in sub.columns:
            continue
        pct = 100 * sub["valid_json"].astype(bool).mean()
        rows.append({"config": CONFIG_LABELS[c].split("(")[0].strip(), "valid_pct": pct})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=plot_df, x="config", y="valid_pct", ax=ax, color="#2e7d32")
    ax.set_ylabel("Respuestas JSON válidas (%)")
    ax.set_xlabel("Configuración")
    ax.set_title("Tasa de éxito de inferencia (proxy ejecución)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    _save_fig(fig, "fig_valid_json")
    plt.close(fig)


def fig_fnr_dual(df: pd.DataFrame):
    from src.metrics import fnr_from_df
    sub = df[df["run_id"] == 0].drop_duplicates("kernel_id")
    configs = ["solo_llama", "triton_heal_dual"]
    vals = []
    for c in configs:
        s = sub[sub["config"] == c] if "config" in sub.columns else sub
        s = df[(df["config"] == c) & (df["run_id"] == 0)].drop_duplicates("kernel_id")
        vals.append(fnr_from_df(s) * 100)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Solo Llama", "Triton Heal dual"], vals, color=["#c55a11", "#1f4e79"])
    ax.set_ylabel("FNR (%)")
    ax.set_title("Falsos negativos: dual vs solo local (PRI-2)")
    fig.tight_layout()
    _save_fig(fig, "fig_fnr_dual")
    plt.close(fig)


def main():
    pred = RESULTS_DIR / "predictions.csv"
    df = pd.read_csv(pred)
    fig_msr_f1(df)
    fig_valid_json(df)
    fig_latency_box(df)
    fig_msr_by_category(df)
    fig_fnr_dual(df)
    print("Figures saved to", FIGURES_DIR, "and", REPORT_FIGURES_DIR)


if __name__ == "__main__":
    main()
