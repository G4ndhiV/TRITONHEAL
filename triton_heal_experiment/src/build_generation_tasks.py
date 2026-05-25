#!/usr/bin/env python3
"""Build generation_tasks.jsonl from benchmark safe kernels (20 tasks)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import BENCHMARK_DIR, GROUND_TRUTH_PATH, KERNELS_DIR

SPECS = {
    "elementwise": "Write a Triton elementwise kernel that doubles input values (x * 2) with masked load/store.",
    "softmax": "Write a Triton softmax-style kernel over a 1D row with masked tl.load and tl.store.",
    "matmul": "Write a simplified Triton matmul-style kernel using tl.load/store and program_id tiling.",
    "reduction": "Write a Triton reduction kernel that loads a block and stores per-program result.",
    "flash_attention": "Write a simplified Triton flash-attention style kernel (q,k,v loads, masked store).",
}

OUT = BENCHMARK_DIR / "generation_tasks.jsonl"


def main():
    rows = []
    seen_cat = {c: 0 for c in SPECS}
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["label"] != "safe":
                continue
            cat = item["category"]
            if cat not in SPECS or seen_cat[cat] >= 4:
                continue
            tid = f"gen_{len(rows):04d}"
            rows.append({
                "task_id": tid,
                "category": cat,
                "spec": SPECS[cat],
                "reference_kernel_id": item["id"],
                "seed": 42 + len(rows),
            })
            seen_cat[cat] += 1
            if len(rows) >= 20:
                break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} tasks to {OUT}")


if __name__ == "__main__":
    main()
