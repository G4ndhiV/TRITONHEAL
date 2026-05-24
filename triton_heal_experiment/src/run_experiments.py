#!/usr/bin/env python3
"""Run all verifier configurations on the benchmark."""
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Allow running as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.paths import GROUND_TRUTH_PATH, KERNELS_DIR, RESULTS_DIR
from src.triton_heal import DualVerifyResult
from src.verify import CONFIG_LABELS, CONFIGS, get_verifier


def load_ground_truth() -> pd.DataFrame:
    rows = []
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def run_row(verifier, code: str, config: str, kernel_id: str, run_id: int) -> dict:
    out = verifier.verify(code)
    if isinstance(out, DualVerifyResult):
        return {
            "kernel_id": kernel_id,
            "config": config,
            "config_label": CONFIG_LABELS[config],
            "model": "triton_heal_dual",
            "pred_safe": out.safe,
            "local_safe": out.local_safe,
            "frontier_safe": out.frontier_safe,
            "disagreement": out.disagreement,
            "veto_applied": out.veto_applied,
            "reason": out.reason,
            "latency_ms": out.latency_ms,
            "valid_json": out.valid_json,
            "run_id": run_id,
        }
    return {
        "kernel_id": kernel_id,
        "config": config,
        "config_label": CONFIG_LABELS[config],
        "model": getattr(out, "backend", config),
        "pred_safe": out.safe,
        "local_safe": None,
        "frontier_safe": None,
        "disagreement": None,
        "veto_applied": None,
        "reason": out.reason,
        "latency_ms": out.latency_ms,
        "valid_json": out.valid_json,
        "run_id": run_id,
    }


def main():
    load_dotenv()
    cfg = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not GROUND_TRUTH_PATH.exists():
        from src.benchmark_build import main as build
        build()

    gt = load_ground_truth()
    repeats = cfg.get("latency_repeats_local", 3)
    rows = []

    for config in CONFIGS:
        print(f"Config: {config}")
        verifier = get_verifier(config, cfg)
        for _, item in gt.iterrows():
            kid = item["id"]
            path = KERNELS_DIR / f"{kid}.tri"
            code = path.read_text(encoding="utf-8")
            for run_id in range(repeats):
                r = run_row(verifier, code, config, kid, run_id)
                r["label"] = item["label"]
                r["category"] = item["category"]
                r["error_type"] = item["error_type"]
                rows.append(r)
                if run_id == 0:
                    print(f"  {kid} -> safe={r['pred_safe']}", flush=True)

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "predictions.csv"
    df.to_csv(out_path, index=False)
    import os
    from .backends.ollama import OllamaVerifier
    from .backends.anthropic_api import AnthropicVerifier
    from .backends.openai_api import OpenAIVerifier

    meta = {
        "n_kernels": len(gt),
        "configs": CONFIGS,
        "verifier_mode": os.getenv("VERIFIER_MODE", cfg.get("verifier_mode")),
        "repeats": repeats,
        "ollama_available": OllamaVerifier.is_available(),
        "openai_available": OpenAIVerifier.is_available(),
        "anthropic_available": AnthropicVerifier.is_available(),
        "note": "Si Ollama/API no disponibles, se usan verificadores heuristicos por tier.",
    }
    (RESULTS_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
