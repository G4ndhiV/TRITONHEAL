#!/usr/bin/env python3
"""Statistics for kernel generation experiment."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.metrics import wilson_ci
from src.paths import REPORT_TABLES, RESULTS_DIR
from src.stats_analysis import holm_adjust, effect_size_label_phi
from src.verify import CONFIG_LABELS, CONFIGS


def mcnemar_compile(df: pd.DataFrame, config_a: str, config_b: str) -> dict:
    """McNemar on compile_ok per task (run_id=0 mean if multiple)."""
    sub = df.groupby(["task_id", "config"]).agg(compile_ok=("compile_ok", "max")).reset_index()
    a = sub[sub["config"] == config_a].set_index("task_id")["compile_ok"].astype(bool)
    b = sub[sub["config"] == config_b].set_index("task_id")["compile_ok"].astype(bool)
    common = a.index.intersection(b.index)
    table = np.zeros((2, 2), dtype=int)
    for tid in common:
        table[int(a.loc[tid]), int(b.loc[tid])] += 1
    try:
        result = mcnemar(table, exact=False, correction=True)
        chi2, p = float(result.statistic), float(result.pvalue)
    except Exception:
        chi2, p = float("nan"), 1.0
    b_disc, c_disc = int(table[0, 1]), int(table[1, 0])
    n_disc = b_disc + c_disc
    phi = (b_disc - c_disc) / np.sqrt(n_disc) if n_disc > 0 else 0.0
    return {"chi2": chi2, "p": p, "phi": phi, "table": table.tolist(), "n_discordant": n_disc}


def wilcoxon_speedup(df: pd.DataFrame, config_a: str, config_b: str) -> dict:
    sub = df[(df["compile_ok"]) & df["speedup_proxy"].notna()]
    a = sub[sub["config"] == config_a].groupby("task_id")["speedup_proxy"].median()
    b = sub[sub["config"] == config_b].groupby("task_id")["speedup_proxy"].median()
    common = a.index.intersection(b.index)
    if len(common) < 5:
        return {"stat": float("nan"), "p": 1.0, "r": 0.0, "median_diff": float("nan")}
    diff = a.loc[common].values - b.loc[common].values
    stat, p_two = stats.wilcoxon(diff)
    _, p_greater = stats.wilcoxon(diff, alternative="greater")
    n = len(diff)
    r = 1 - (2 * stat) / (n * (n + 1)) if n else 0
    return {
        "stat": float(stat),
        "p": float(p_greater),
        "r": float(r),
        "median_diff": float(np.median(diff)),
    }


def descriptive_generation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config in CONFIGS:
        sub = df[df["config"] == config]
        n = len(sub)
        n_ok = int(sub["compile_ok"].sum())
        sp = sub.loc[sub["compile_ok"], "speedup_proxy"].dropna()
        rows.append({
            "config": config,
            "label": CONFIG_LABELS[config],
            "n": n,
            "compile_pct": 100 * n_ok / n if n else 0,
            "compile_n_ok": n_ok,
            "speedup_mean": sp.mean() if len(sp) else float("nan"),
            "speedup_std": sp.std() if len(sp) > 1 else float("nan"),
            "speedup_min": sp.min() if len(sp) else float("nan"),
            "speedup_max": sp.max() if len(sp) else float("nan"),
            "gen_latency_mean": sub["gen_latency_ms"].mean(),
            "gen_latency_std": sub["gen_latency_ms"].std(),
            "gpu_eff_mean": sub["gpu_efficiency_proxy"].mean(),
        })
    return pd.DataFrame(rows)


def write_generation_descriptive_tex(df: pd.DataFrame, path: Path) -> None:
    cols = [
        ("label", "Configuración"),
        ("n", "N"),
        ("compile_pct", "Compilación (\\%)"),
        ("speedup_mean", "Speedup proxy"),
        ("speedup_std", "Desv."),
        ("gen_latency_mean", "Lat. gen. (ms)"),
        ("gpu_eff_mean", "GPU eff. proxy"),
    ]
    ncol = len(cols)
    colspec = "@{}>{\\raggedright\\arraybackslash}p{0.26\\linewidth}" + "r" * (ncol - 1) + "@{}"
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Resultados descriptivos de generación de kernels (M4).}",
        "\\label{tab:gen_desc}",
        "\\footnotesize",
        "\\begin{adjustbox}{max width=\\linewidth,center}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(h for _, h in cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = []
        for key, _ in cols:
            v = row.get(key, "")
            if isinstance(v, float):
                if np.isnan(v):
                    cells.append("---")
                elif key.startswith("speedup"):
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(f"{v:.2f}")
            else:
                cells.append(str(v).replace("_", "\\_"))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_generation_inference(tests: dict, holm: dict, desc: pd.DataFrame) -> list[dict]:
    rows = []

    def fmt_p(p):
        return "$<0.001$" if p < 0.001 else f"{p:.4f}"

    def reject(p):
        return p < 0.05

    if "C4_dual_vs_gpt_compile" in tests:
        c4 = tests["C4_dual_vs_gpt_compile"]
        p4 = holm["C4_dual_vs_gpt_compile"]
        dual_pct = desc.loc[desc["config"] == "triton_heal_dual", "compile_pct"].iloc[0]
        gpt_pct = desc.loc[desc["config"] == "solo_gpt4o", "compile_pct"].iloc[0]
        rows.append({
            "nombre": "Dual vs GPT-4o (compilación)",
            "prueba": "McNemar",
            "stat": f"$\\chi^2={c4['chi2']:.2f}$",
            "p_holm": fmt_p(p4),
            "efecto": f"$\\phi={c4['phi']:.2f}$ ({effect_size_label_phi(c4['phi'])})",
            "interp": (
                f"Rechaza $H_0$; dual {dual_pct:.1f}\\% vs GPT {gpt_pct:.1f}\\%."
                if reject(p4)
                else f"No rechaza $H_0$; dual {dual_pct:.1f}\\% vs GPT {gpt_pct:.1f}\\%."
            ),
        })

    if "C7_dual_vs_llama_compile" in tests:
        c7 = tests["C7_dual_vs_llama_compile"]
        p7 = holm["C7_dual_vs_llama_compile"]
        rows.append({
            "nombre": "Dual vs Llama (compilación)",
            "prueba": "McNemar",
            "stat": f"$\\chi^2={c7['chi2']:.2f}$",
            "p_holm": fmt_p(p7),
            "efecto": f"$\\phi={c7['phi']:.2f}$ ({effect_size_label_phi(c7['phi'])})",
            "interp": "Rechaza $H_0$." if reject(p7) else "No rechaza $H_0$.",
        })

    if "C8_speedup_dual_vs_gpt" in tests:
        c8 = tests["C8_speedup_dual_vs_gpt"]
        p8 = holm["C8_speedup_dual_vs_gpt"]
        rows.append({
            "nombre": "Speedup proxy dual vs GPT",
            "prueba": "Wilcoxon",
            "stat": f"$W={c8['stat']:.0f}$",
            "p_holm": fmt_p(p8),
            "efecto": f"$r={c8['r']:.2f}$",
            "interp": (
                f"Med. $\\Delta$ speedup={c8['median_diff']:.2f}."
                if not reject(p8)
                else f"Rechaza $H_0$; med. $\\Delta$={c8['median_diff']:.2f}."
            ),
        })
    return rows


def write_generation_inference_tex(rows: list[dict], path: Path) -> None:
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Inferencia estadistica -- generacion de kernels (Holm).}",
        "\\label{tab:gen_infer}",
        "\\footnotesize",
        "\\begin{adjustbox}{max width=\\linewidth}",
        "\\begin{tabularx}{\\linewidth}{@{}lcccp{0.35\\linewidth}@{}}",
        "\\toprule",
        "Contraste & Prueba & Estad. & $p$ Holm & Interpretación \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['nombre']} & {r['prueba']} & {r['stat']} & {r['p_holm']} & {r['interp']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabularx}", "\\end{adjustbox}", "\\end{table}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    path = RESULTS_DIR / "generation.csv"
    if not path.exists():
        raise FileNotFoundError("Run: python -m src.run_generation")
    df = pd.read_csv(path)
    df["compile_ok"] = df["compile_ok"].astype(bool)

    desc = descriptive_generation(df)
    desc.to_csv(RESULTS_DIR / "generation_descriptive.csv", index=False)

    tests = {
        "C4_dual_vs_gpt_compile": mcnemar_compile(df, "triton_heal_dual", "solo_gpt4o"),
        "C7_dual_vs_llama_compile": mcnemar_compile(df, "triton_heal_dual", "solo_llama"),
        "C8_speedup_dual_vs_gpt": wilcoxon_speedup(df, "triton_heal_dual", "solo_gpt4o"),
    }
    pvals = {k: v["p"] for k, v in tests.items()}
    holm = holm_adjust(pvals)
    inf_display = build_generation_inference(tests, holm, desc)

    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    write_generation_descriptive_tex(desc, REPORT_TABLES / "generation_descriptive.tex")
    write_generation_inference_tex(inf_display, REPORT_TABLES / "generation_inference.tex")

    summary = {
        "descriptive": desc.to_dict(orient="records"),
        "tests": {k: {**v, "p_holm": holm[k]} for k, v in tests.items()},
        "holm": holm,
        "n_rows": len(df),
        "n_tasks": int(df["task_id"].nunique()),
    }
    (RESULTS_DIR / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    print("Generation stats written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
