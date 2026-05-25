#!/usr/bin/env python3
"""Inferential statistics per pre-registered plan."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.metrics import fnr_from_df, f1_from_df, msr_from_df, wilson_ci
from src.paths import REPORT_TABLES, RESULTS_DIR
from src.verify import CONFIG_LABELS, CONFIGS


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)
    adjusted = {}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        adj = max(adj, prev)
        prev = adj
        adjusted[k] = adj
    return adjusted


def mcnemar_test(df: pd.DataFrame, config_a: str, config_b: str) -> dict:
    a = df[df["config"] == config_a].drop_duplicates("kernel_id").set_index("kernel_id")
    b = df[df["config"] == config_b].drop_duplicates("kernel_id").set_index("kernel_id")
    common = a.index.intersection(b.index)
    gt = a.loc[common, "label"] == "unsafe"

    def detected(sub):
        return ~sub.loc[common, "pred_safe"].astype(bool)

    det_a = detected(a)
    det_b = detected(b)
    table = np.zeros((2, 2), dtype=int)
    for i in range(len(common)):
        if gt.iloc[i]:
            table[int(det_a.iloc[i]), int(det_b.iloc[i])] += 1
    try:
        result = mcnemar(table, exact=False, correction=True)
        chi2 = float(result.statistic)
        p = float(result.pvalue)
    except Exception:
        chi2, p = float("nan"), 1.0
    b_disc, c_disc = int(table[0, 1]), int(table[1, 0])
    n_disc = b_disc + c_disc
    phi = (b_disc - c_disc) / np.sqrt(n_disc) if n_disc > 0 else 0.0
    return {
        "chi2": chi2,
        "p": p,
        "phi": phi,
        "table": table.tolist(),
        "n_discordant": n_disc,
    }


def wilcoxon_latency(df: pd.DataFrame, config_local: str, config_frontier: str) -> dict:
    a = df[df["config"] == config_local].groupby("kernel_id")["latency_ms"].median()
    b = df[df["config"] == config_frontier].groupby("kernel_id")["latency_ms"].median()
    common = a.index.intersection(b.index)
    diff = b.loc[common].values - a.loc[common].values
    if len(diff) < 5:
        return {"stat": float("nan"), "p": 1.0, "r": 0.0, "median_diff": float("nan")}
    stat, p_two = stats.wilcoxon(diff)
    _, p_greater = stats.wilcoxon(diff, alternative="greater")
    n = len(diff)
    r = 1 - (2 * stat) / (n * (n + 1)) if n else 0
    return {
        "stat": float(stat),
        "p": float(p_greater),
        "p_two_sided": float(p_two),
        "r": float(r),
        "median_diff": float(np.median(diff)),
        "mean_diff": float(np.mean(diff)),
    }


def friedman_f1(df: pd.DataFrame) -> dict:
    configs = [
        "solo_llama",
        "solo_deepseek",
        "solo_deepseek_coder",
        "solo_claude",
        "solo_gpt4o",
    ]
    per_kernel = []
    for kid in df["kernel_id"].unique():
        row = []
        sub_k = df[df["kernel_id"] == kid]
        gt_unsafe = sub_k["label"].iloc[0] == "unsafe"
        for c in configs:
            s = sub_k[sub_k["config"] == c]
            if len(s) == 0:
                row.append(np.nan)
            else:
                pred = bool(s["pred_safe"].iloc[0])
                correct = (gt_unsafe and not pred) or (not gt_unsafe and pred)
                row.append(1.0 if correct else 0.0)
        if not any(np.isnan(row)):
            per_kernel.append(row)
    if len(per_kernel) < 3:
        return {"chi2": float("nan"), "p": 1.0, "w": 0.0}
    arr = np.array(per_kernel)
    stat, p = stats.friedmanchisquare(*[arr[:, i] for i in range(arr.shape[1])])
    n, k = arr.shape
    w = stat / (n * (k - 1)) if k > 1 else 0
    return {"chi2": float(stat), "p": float(p), "w": float(w)}


def descriptive_table(df: pd.DataFrame, df_all: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for config in CONFIGS:
        sub = df[df["config"] == config].drop_duplicates("kernel_id")
        all_c = df_all[df_all["config"] == config] if df_all is not None else sub
        gt_unsafe = sub["label"] == "unsafe"
        pred_unsafe = ~sub["pred_safe"].astype(bool)
        tp = int((gt_unsafe & pred_unsafe).sum())
        fn = int((gt_unsafe & ~pred_unsafe).sum())
        lat = all_c["latency_ms"]
        n_valid = int(all_c["valid_json"].astype(bool).sum()) if "valid_json" in all_c.columns else len(sub)
        n_total = len(all_c)
        rows.append({
            "config": config,
            "label": CONFIG_LABELS[config],
            "n": len(sub),
            "valid_json_pct": 100 * n_valid / n_total if n_total else 0,
            "msr_pct": msr_from_df(sub) * 100,
            "fnr_pct": fnr_from_df(sub) * 100,
            "f1": f1_from_df(sub),
            "latency_mean": lat.mean(),
            "latency_std": lat.std(),
            "latency_median": lat.median(),
            "latency_min": lat.min(),
            "latency_max": lat.max(),
            "tp": tp,
            "fn": fn,
        })
    return pd.DataFrame(rows)


def descriptive_extended(df_all: pd.DataFrame, df_run0: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config in CONFIGS:
        sub = df_run0[df_run0["config"] == config].drop_duplicates("kernel_id")
        all_c = df_all[df_all["config"] == config]
        n_valid = int(all_c["valid_json"].sum()) if "valid_json" in all_c.columns else len(sub)
        n_total = len(all_c)
        gt_unsafe = sub["label"] == "unsafe"
        pred_unsafe = ~sub["pred_safe"].astype(bool)
        tp = int((gt_unsafe & pred_unsafe).sum())
        fn = int((gt_unsafe & ~pred_unsafe).sum())
        n_unsafe = int(gt_unsafe.sum())
        msr_lo, msr_hi = wilson_ci(tp, n_unsafe) if n_unsafe else (float("nan"), float("nan"))
        lat = all_c["latency_ms"]
        timeout_pct = 0.0
        if "reason" in all_c.columns:
            timeout_pct = (
                all_c["reason"].astype(str).str.contains("timeout|ollama_error", case=False, na=False).mean()
                * 100
            )
        rows.append({
            "config": config,
            "label": CONFIG_LABELS[config],
            "n_kernels": len(sub),
            "n_evaluations": n_total,
            "n_valid_json": n_valid,
            "valid_json_pct": 100 * n_valid / n_total if n_total else 0,
            "timeout_pct": timeout_pct,
            "msr_pct": msr_from_df(sub) * 100,
            "msr_ci_lo": msr_lo * 100 if n_unsafe else float("nan"),
            "msr_ci_hi": msr_hi * 100 if n_unsafe else float("nan"),
            "fnr_pct": fnr_from_df(sub) * 100,
            "f1": f1_from_df(sub),
            "latency_mean": lat.mean(),
            "latency_std": lat.std(),
            "latency_median": lat.median(),
            "latency_min": lat.min(),
            "latency_max": lat.max(),
            "tp": tp,
            "fn": fn,
        })
    return pd.DataFrame(rows)


def effect_size_label_phi(phi: float) -> str:
    a = abs(phi)
    if a < 0.2:
        return "pequeño"
    if a < 0.5:
        return "mediano"
    return "grande"


def write_descriptive_extended_tex(df: pd.DataFrame, path: Path) -> None:
    cols = [
        ("label", "Config.", "X"),
        ("n_kernels", "N", "c"),
        ("valid_json_pct", "JSON\\,vál. (\\%)", "c"),
        ("msr_pct", "MSR (\\%)", "c"),
        ("msr_ci_lo", "IC lo", "c"),
        ("msr_ci_hi", "IC hi", "c"),
        ("f1", "F1", "c"),
        ("fnr_pct", "FNR (\\%)", "c"),
        ("latency_mean", "Lat. media", "c"),
        ("latency_std", "Lat. desv.", "c"),
        ("latency_median", "Lat. med.", "c"),
        ("latency_min", "Lat. min", "c"),
        ("latency_max", "Lat. max", "c"),
    ]
    ncol = len(cols)
    colspec = "@{}l" + "r" * (ncol - 1) + "@{}"
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Resultados descriptivos extendidos (IC 95\\% Wilson para MSR).}",
        "\\label{tab:desc_ext}",
        "\\begin{adjustbox}{max width=\\linewidth,center}",
        "{\\scriptsize\\setlength{\\tabcolsep}{2.5pt}",
        f"\\resizebox{{\\linewidth}}{{!}}{{%",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(h for _, h, _ in cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        line_cells = []
        for key, _, _spec in cols:
            v = row.get(key, "")
            if isinstance(v, float):
                text = f"{v:.1f}"
            else:
                text = str(v).replace("_", "\\_")
            line_cells.append(text)
        lines.append(" & ".join(line_cells) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}}",
        "\\end{adjustbox}",
        "\\end{table}",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_inference_tex(inf_rows: list[dict], path: Path) -> None:
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Inferencia estadística (plan pre-registrado, corrección Holm).}",
        "\\label{tab:infer}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{adjustbox}{max width=\\linewidth}",
        "\\begin{tabularx}{\\linewidth}{@{}"
        ">{\\raggedright\\arraybackslash}p{0.17\\linewidth}"
        "c c c c"
        ">{\\raggedright\\arraybackslash}X@{}}",
        "\\toprule",
        "Contraste & Prueba & Estad. & $p$ Holm & Efecto & Interpretación \\\\",
        "\\midrule",
    ]
    for r in inf_rows:
        lines.append(
            f"{r['nombre']} & {r['prueba']} & {r['stat']} & {r['p_holm']} & {r['efecto']} & {r['interp']} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabularx}",
        "\\end{adjustbox}",
        "\\end{table}",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _reject_h0(p_holm: float, alpha: float = 0.05) -> bool:
    return p_holm < alpha


def build_inference_rows(tests: dict, holm: dict, desc_ext: pd.DataFrame) -> list[dict]:
    rows = []

    def fmt_p(p):
        if p < 0.001:
            return "$<0.001$"
        return f"{p:.4f}"

    c1 = tests["C1_claude_vs_llama"]
    p1 = holm["C1_claude_vs_llama"]
    msr_claude = desc_ext.loc[desc_ext["config"] == "solo_claude", "msr_pct"].iloc[0]
    msr_llama = desc_ext.loc[desc_ext["config"] == "solo_llama", "msr_pct"].iloc[0]
    delta = msr_claude - msr_llama
    c1_interp = (
        f"No rechaza $H_0$; $\\Delta$MSR={delta:.1f} pp (Claude heur.\\ vs.\\ Llama Ollama/M4)."
        if not _reject_h0(p1)
        else f"Rechaza $H_0$; $\\Delta$MSR={delta:.1f} pp."
    )
    rows.append({
        "nombre": "Claude vs Llama (MSR)",
        "prueba": "McNemar",
        "stat": f"$\\chi^2={c1['chi2']:.2f}$",
        "p_holm": fmt_p(p1),
        "efecto": f"$\\phi={c1['phi']:.2f}$ ({effect_size_label_phi(c1['phi'])})",
        "interp": c1_interp,
    })

    c2 = tests["C2_gpt_vs_deepseek_coder"]
    p2 = holm["C2_gpt_vs_deepseek_coder"]
    msr_gpt = desc_ext.loc[desc_ext["config"] == "solo_gpt4o", "msr_pct"].iloc[0]
    msr_coder = desc_ext.loc[desc_ext["config"] == "solo_deepseek_coder", "msr_pct"].iloc[0]
    c2_interp = (
        f"No rechaza $H_0$; GPT ({msr_gpt:.1f}\\%) vs.\\ DeepSeek-Coder-V2 frontera ({msr_coder:.1f}\\%)."
        if not _reject_h0(p2)
        else f"Rechaza $H_0$; $\\Delta$MSR={msr_gpt - msr_coder:.1f} pp."
    )
    rows.append({
        "nombre": "GPT-4o vs DeepSeek-Coder",
        "prueba": "McNemar",
        "stat": f"$\\chi^2={c2['chi2']:.2f}$",
        "p_holm": fmt_p(p2),
        "efecto": f"$\\phi={c2['phi']:.2f}$ ({effect_size_label_phi(c2['phi'])})",
        "interp": c2_interp,
    })

    c3 = tests["C3_dual_vs_llama_fnr"]
    p3 = holm["C3_dual_vs_llama_fnr"]
    c3_interp = (
        "Rechaza $H_0$; dual reduce FNR (más detecciones en unsafe)."
        if _reject_h0(p3)
        else "No rechaza $H_0$ en FNR pareado."
    )
    rows.append({
        "nombre": "Dual vs Llama",
        "prueba": "McNemar",
        "stat": f"$\\chi^2={c3['chi2']:.2f}$",
        "p_holm": fmt_p(p3),
        "efecto": f"$\\phi={c3['phi']:.2f}$ ({effect_size_label_phi(c3['phi'])})",
        "interp": c3_interp,
    })

    c5 = tests["C5_latency"]
    p5 = holm["C5_latency"]
    rows.append({
        "nombre": "Lat. Claude vs Llama",
        "prueba": "Wilcoxon",
        "stat": f"$W={c5['stat']:.0f}$",
        "p_holm": fmt_p(p5),
        "efecto": f"$r={c5['r']:.2f}$",
        "interp": (
            f"No rechaza $H_4$; med. $\\Delta$={c5['median_diff']:.0f} ms (Claude heur.\\ más rápido)."
            if not _reject_h0(p5)
            else f"Rechaza $H_4$; frontera más lenta (med. $\\Delta$={c5['median_diff']:.0f} ms)."
        ),
    })

    c6 = tests["C6_friedman"]
    p6 = holm["C6_friedman"]
    rows.append({
        "nombre": "F1 global (5 cfg.)",
        "prueba": "Friedman",
        "stat": f"$\\chi^2={c6['chi2']:.1f}$",
        "p_holm": fmt_p(p6),
        "efecto": f"$W={c6['w']:.2f}$",
        "interp": (
            "Diferencias globales entre configuraciones (Holm)."
            if _reject_h0(p6)
            else "No rechaza igualdad global de F1 tras Holm."
        ),
    })
    return rows


def to_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    cols = [
        ("label", "Configuración"),
        ("n", "N"),
        ("valid_json_pct", "JSON vál. (\\%)"),
        ("msr_pct", "MSR (\\%)"),
        ("f1", "F1"),
        ("fnr_pct", "FNR (\\%)"),
        ("latency_mean", "Lat. media (ms)"),
        ("latency_std", "Lat. desv. (ms)"),
        ("latency_median", "Lat. med. (ms)"),
    ]
    ncol = len(cols)
    colspec = "@{}>{\\raggedright\\arraybackslash}p{0.28\\linewidth}" + "r" * (ncol - 1) + "@{}"
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{adjustbox}{max width=\\linewidth,center}",
        "{\\footnotesize\\setlength{\\tabcolsep}{3pt}",
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
                cells.append(f"{v:.1f}")
            else:
                cells.append(str(v).replace("_", "\\_"))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{adjustbox}",
        "\\end{table}",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _detect_configs_backend(run_meta: dict) -> dict:
    from .backends.ollama import OllamaVerifier
    from .config import load_config

    cfg = load_config()
    host = cfg.get("ollama_host", "http://127.0.0.1:11434")
    models = cfg.get("models", {})
    ollama_ok = run_meta.get("ollama_available", OllamaVerifier.is_available(host))

    def ollama_label(key: str, fallback: str) -> str:
        tag = models.get(key, key)
        if ollama_ok and OllamaVerifier.has_model(tag, host):
            return f"Ollama {tag} (Metal / M4)"
        return fallback

    return {
        "solo_llama": ollama_label("llama_local", "Heurístico (Llama)"),
        "solo_deepseek": ollama_label("deepseek_local", "Heurístico (DeepSeek-R1 14B)"),
        "solo_deepseek_coder": ollama_label(
            "deepseek_frontier", "Heurístico (DeepSeek-Coder-V2)"
        ),
        "solo_claude": (
            "API Claude"
            if run_meta.get("anthropic_available")
            else "Heurístico (sin API)"
        ),
        "solo_gpt4o": (
            "API GPT-4o"
            if run_meta.get("openai_available")
            else "Heurístico (sin API)"
        ),
        "triton_heal_dual": (
            ollama_label("llama_local", "Llama")
            + " + veto "
            + ollama_label("deepseek_frontier", "DeepSeek-Coder-V2")
        ),
    }


def benchmark_lex_pass_rate() -> dict:
    from .lex_checker import is_lexically_valid
    from .paths import KERNELS_DIR

    paths = sorted(KERNELS_DIR.glob("*.tri"))
    n = len(paths)
    ok = sum(
        1 for p in paths if is_lexically_valid(p.read_text(encoding="utf-8"), p)
    )
    return {"n_kernels": n, "n_lex_ok": ok, "pct_lex_ok": 100 * ok / n if n else 0}


def write_robustness_tex(df: pd.DataFrame, path: Path) -> None:
    """MSR and latency mean ± std across run_id per config."""
    rows = []
    for config in CONFIGS:
        sub = df[df["config"] == config]
        if sub.empty:
            continue
        by_run = []
        for rid in sorted(sub["run_id"].unique()):
            r0 = sub[sub["run_id"] == rid].drop_duplicates("kernel_id")
            by_run.append(msr_from_df(r0) * 100)
        msr_mean = float(np.mean(by_run)) if by_run else float("nan")
        msr_std = float(np.std(by_run)) if len(by_run) > 1 else 0.0
        lat_mean = sub.groupby("run_id")["latency_ms"].median().mean()
        lat_std = sub.groupby("run_id")["latency_ms"].median().std()
        if np.isnan(lat_std):
            lat_std = 0.0
        rows.append({
            "label": CONFIG_LABELS[config],
            "n_runs": len(by_run),
            "msr_mean": msr_mean,
            "msr_std": msr_std,
            "lat_mean": lat_mean,
            "lat_std": lat_std,
        })
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Robustez verificación: MSR y latencia mediana por réplica (\\texttt{run\\_id}).}",
        "\\label{tab:robustness}",
        "\\footnotesize",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Config. & Réplicas & MSR media (\\%) & MSR desv. & Lat. med. (ms) $\\pm$ desv. \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['label']} & {r['n_runs']} & {r['msr_mean']:.1f} & {r['msr_std']:.1f} & "
            f"{r['lat_mean']:.0f} $\\pm$ {r['lat_std']:.0f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    pred_path = RESULTS_DIR / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError("Run run_experiments.py first")
    df = pd.read_csv(pred_path)
    df["valid_json"] = df["valid_json"].astype(bool)
    df_run0 = df[df["run_id"] == 0] if "run_id" in df.columns else df

    desc = descriptive_table(df_run0, df)
    desc_ext = descriptive_extended(df, df_run0)
    lex_stats = benchmark_lex_pass_rate()
    desc.to_csv(RESULTS_DIR / "descriptive.csv", index=False)
    desc_ext.to_csv(RESULTS_DIR / "descriptive_extended.csv", index=False)

    tests = {
        "C1_claude_vs_llama": mcnemar_test(df_run0, "solo_claude", "solo_llama"),
        "C2_gpt_vs_deepseek_coder": mcnemar_test(
            df_run0, "solo_gpt4o", "solo_deepseek_coder"
        ),
        "C3_dual_vs_llama_fnr": mcnemar_test(df_run0, "triton_heal_dual", "solo_llama"),
        "C5_latency": wilcoxon_latency(df, "solo_llama", "solo_claude"),
        "C6_friedman": friedman_f1(df_run0),
    }
    pvals = {k: v["p"] for k, v in tests.items() if "p" in v}
    holm = holm_adjust(pvals)

    inf_rows_raw = []
    for name, res in tests.items():
        inf_rows_raw.append({"test": name, **res, "p_holm": holm.get(name, res.get("p"))})
    pd.DataFrame(inf_rows_raw).to_csv(RESULTS_DIR / "inference.csv", index=False)

    inf_display = build_inference_rows(tests, holm, desc_ext)

    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    to_latex_table(desc, REPORT_TABLES / "descriptive.tex", "Resultados descriptivos por configuración.", "tab:desc")
    write_descriptive_extended_tex(desc_ext, REPORT_TABLES / "descriptive_extended.tex")
    write_inference_tex(inf_display, REPORT_TABLES / "inference.tex")
    if "run_id" in df.columns:
        write_robustness_tex(df, REPORT_TABLES / "robustness.tex")

    run_meta_path = RESULTS_DIR / "run_meta.json"
    run_meta = {}
    if run_meta_path.exists():
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

    n_valid_json = int(df["valid_json"].astype(bool).sum())
    run_summary = {
        "n_kernels": int(df_run0["kernel_id"].nunique()),
        "n_evaluations_total": len(df),
        "n_valid_json_total": n_valid_json,
        "valid_json_pct_global": 100 * n_valid_json / len(df) if len(df) else 0,
        "n_safe": int((df_run0.drop_duplicates("kernel_id")["label"] == "safe").sum()),
        "n_unsafe": int((df_run0.drop_duplicates("kernel_id")["label"] == "unsafe").sum()),
        "verifier_mode": run_meta.get("verifier_mode", "unknown"),
        "ollama_available": run_meta.get("ollama_available", False),
        "configs_backend": _detect_configs_backend(run_meta),
        "descriptive": desc.to_dict(orient="records"),
        "descriptive_extended": desc_ext.to_dict(orient="records"),
        "benchmark_lex": lex_stats,
        "inference": inf_rows_raw,
        "inference_display": inf_display,
        "holm": holm,
        "git_commit": "2c027b5",
    }
    (RESULTS_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, default=float), encoding="utf-8"
    )
    (RESULTS_DIR / "stats_summary.json").write_text(
        json.dumps(run_summary, indent=2, default=float), encoding="utf-8"
    )
    print("Stats written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
