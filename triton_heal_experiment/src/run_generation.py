#!/usr/bin/env python3
"""Run kernel generation experiment for all configurations."""
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compile_checker import check_compile, check_reference_baseline
from src.config import load_config
from src.kernel_generator import DualGenerator, GenerateResult, get_generator
from src.paths import BENCHMARK_DIR, KERNELS_DIR, RESULTS_DIR
from src.verify import CONFIG_LABELS, CONFIGS

TASKS_PATH = BENCHMARK_DIR / "generation_tasks.jsonl"


def load_tasks():
    if not TASKS_PATH.exists():
        from src.build_generation_tasks import main as build_tasks
        build_tasks()
    rows = []
    with open(TASKS_PATH, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def run_single(config: str, task: dict, run_id: int, cfg: dict) -> dict:
    spec = task["spec"]
    ref_id = task["reference_kernel_id"]
    ref_code = (KERNELS_DIR / f"{ref_id}.tri").read_text(encoding="utf-8")
    baseline_ms = check_reference_baseline(ref_code)

    gen = get_generator(config, cfg)
    retries = 0
    if config == "triton_heal_dual" and isinstance(gen, DualGenerator):
        out, retries = gen.generate_with_repair(spec)
    else:
        out = gen.generate(spec)

    code = out.code or ""
    cr = check_compile(code)
    gen_pipeline_ms = out.latency_ms + cr.check_ms
    speedup_proxy = (
        baseline_ms / gen_pipeline_ms if cr.compile_ok and gen_pipeline_ms > 0 else float("nan")
    )
    gpu_eff = (
        (1.0 / out.latency_ms * 1000) if out.latency_ms > 0 and cr.compile_ok else 0.0
    )

    return {
        "task_id": task["task_id"],
        "category": task["category"],
        "config": config,
        "config_label": CONFIG_LABELS[config],
        "reference_kernel_id": ref_id,
        "seed": task.get("seed", 42),
        "run_id": run_id,
        "generated_code_len": len(code),
        "gen_latency_ms": out.latency_ms,
        "check_ms": cr.check_ms,
        "gen_pipeline_ms": gen_pipeline_ms,
        "baseline_check_ms": baseline_ms,
        "ast_ok": cr.ast_ok,
        "lex_ok": cr.lex_ok,
        "compile_ok": cr.compile_ok,
        "triton_import_ok": cr.triton_import_ok,
        "compile_error": cr.error,
        "levels_passed": cr.levels_passed,
        "retries": retries,
        "speedup_proxy": speedup_proxy,
        "gpu_efficiency_proxy": gpu_eff,
        "backend": out.backend,
        "gen_error": out.error,
    }


def main():
    import os

    load_dotenv()
    cfg = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    max_tasks = int(os.getenv("GENERATION_MAX_TASKS", len(tasks)))
    tasks = tasks[:max_tasks]
    repeats = int(os.getenv("GENERATION_REPEATS", cfg.get("generation_repeats", 3)))
    skip = {s.strip() for s in os.getenv("GENERATION_SKIP_CONFIGS", "").split(",") if s.strip()}
    rows = []

    baselines = {}
    for task in tasks:
        ref_code = (KERNELS_DIR / f"{task['reference_kernel_id']}.tri").read_text(encoding="utf-8")
        baselines[task["task_id"]] = check_reference_baseline(ref_code)

    out = RESULTS_DIR / "generation.csv"

    for config in CONFIGS:
        if config in skip:
            print(f"Skipping {config} (GENERATION_SKIP_CONFIGS)", flush=True)
            continue
        print(f"Generation config: {config}", flush=True)
        for task in tasks:
            for run_id in range(repeats):
                r = run_single(config, task, run_id, cfg)
                r["baseline_check_ms"] = baselines[task["task_id"]]
                if r["compile_ok"] and r["gen_pipeline_ms"] > 0:
                    r["speedup_proxy"] = baselines[task["task_id"]] / r["gen_pipeline_ms"]
                rows.append(r)
                if run_id == 0:
                    print(
                        f"  {task['task_id']} compile_ok={r['compile_ok']} "
                        f"lat={r['gen_latency_ms']:.0f}ms",
                        flush=True,
                    )
        pd.DataFrame(rows).to_csv(out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    meta = {
        "n_tasks": len(tasks),
        "n_configs": len(CONFIGS),
        "repeats": repeats,
        "n_rows": len(df),
        "compile_rate_global": float(df["compile_ok"].mean()) if len(df) else 0,
    }
    (RESULTS_DIR / "generation_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
