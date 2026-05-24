#!/usr/bin/env python3
"""Build Triton Kernel Safety Benchmark from Lex seeds + procedural variants."""
import json
import random
import re
from pathlib import Path

from .config import load_config
from .lex_checker import is_lexically_valid
from .paths import BENCHMARK_DIR, GROUND_TRUTH_PATH, KERNELS_DIR, LEX_DIR

CATEGORIES = ["elementwise", "softmax", "matmul", "reduction", "flash_attention"]

SAFE_TEMPLATES = {
    "elementwise": '''import triton
import triton.language as tl

@triton.jit
def elem_{id}(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)
''',
    "softmax": '''import triton
import triton.language as tl

@triton.jit
def softmax_{id}(out_ptr, inp_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(inp_ptr + offs, mask=mask)
    x = tl.exp(x - tl.max(x, axis=0))
    tl.store(out_ptr + offs, x / tl.sum(x, axis=0), mask=mask)
''',
    "matmul": '''import triton
import triton.language as tl

@triton.jit
def matmul_{id}(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK + tl.arange(0, BLOCK)
    rn = pid_n * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, K, BLOCK):
        a = tl.load(a_ptr + rm[:, None] * K + k + tl.arange(0, BLOCK), mask=rm[:, None] < M)
        b = tl.load(b_ptr + (k + tl.arange(0, BLOCK)) * N + rn[None, :], mask=rn[None, :] < N)
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
''',
    "reduction": '''import triton
import triton.language as tl

@triton.jit
def reduce_{id}(inp_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    val = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    s = tl.sum(val, axis=0)
    if pid == 0:
        tl.store(out_ptr, s)
''',
    "flash_attention": '''import triton
import triton.language as tl

@triton.jit
def flash_{id}(q_ptr, k_ptr, v_ptr, out_ptr, seq_len, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < seq_len
    q = tl.load(q_ptr + offs, mask=mask)
    k = tl.load(k_ptr + offs, mask=mask)
    v = tl.load(v_ptr + offs, mask=mask)
    scores = q * k
    tl.store(out_ptr + offs, scores * v, mask=mask)
''',
}

UNSAFE_MUTATIONS = [
    ("oob_index", lambda s: re.sub(
        r"tl\.load\(([^)]+)\)",
        r"tl.load(\1 + 999999)",
        s,
        count=1,
    ), "memory_oob"),
    ("no_jit", lambda s: s.replace("@triton.jit\n", ""), "missing_decorator"),
    ("bad_block_size", lambda s: re.sub(
        r"BLOCK_SIZE: tl\.constexpr|BLOCK: tl\.constexpr",
        "BLOCK_SIZE: tl.constexpr  # BLOCK=48 non-power-of-two",
        s,
    ).replace("tl.arange(0, BLOCK_SIZE)", "tl.arange(0, 48)").replace(
        "tl.arange(0, BLOCK)", "tl.arange(0, 48)"
    ) if "BLOCK" in s else s, "sram_tiling_misaligned"),
    ("sram_accum", lambda s: s + "\n# UNSAFE: excessive tile accumulation for SRAM\n" + (
        "acc = tl.zeros((128, 128), dtype=tl.float32)\n" * 3
    ), "sram_overflow"),
    ("mixed_precision", lambda s: s.replace(
        "dtype=tl.float32", "dtype=tl.float16"
    ).replace("tl.float32", "tl.float16") + "\n# mixed FP16/FP32 without cast\nx_fp32 = tl.load(x_ptr)\n",
               "mixed_precision"),
    ("illegal_store", lambda s: re.sub(
        r"tl\.store\([^)]+\)",
        "tl.store(out_ptr + offs)",
        s,
        count=1,
    ), "memory_unmasked_store"),
]


def load_lex_seeds() -> list[dict]:
    entries = []
    for p in sorted(LEX_DIR.glob("*.tri")):
        text = p.read_text(encoding="utf-8")
        label = "unsafe" if "bad" in p.name or "error" in p.name else "safe"
        err = "lex_seed_unsafe" if label == "unsafe" else "none"
        entries.append({
            "source": "lex",
            "file": p.name,
            "code": text,
            "label": label,
            "category": "elementwise" if "add" in text else "matmul" if "matmul" in text else "reduction",
            "error_type": err,
        })
    return entries


def generate_benchmark(n_total: int = 80, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    entries: list[dict] = []
    entries.extend(load_lex_seeds())

    kid = len(entries)
    per_cat = max(1, (n_total - len(entries)) // (2 * len(CATEGORIES)))

    for cat in CATEGORIES:
        for _ in range(per_cat):
            code = SAFE_TEMPLATES[cat].format(id=kid)
            if is_lexically_valid(code):
                entries.append({
                    "id": f"k_{kid:04d}",
                    "label": "safe",
                    "category": cat,
                    "error_type": "none",
                    "source": "generated",
                    "code": code,
                })
                kid += 1

        for mut_name, mut_fn, err_type in UNSAFE_MUTATIONS:
            if len(entries) >= n_total:
                break
            base = SAFE_TEMPLATES[cat].format(id=kid)
            try:
                mutated = mut_fn(base)
            except Exception:
                continue
            if mutated != base and is_lexically_valid(mutated):
                entries.append({
                    "id": f"k_{kid:04d}",
                    "label": "unsafe",
                    "category": cat,
                    "error_type": err_type,
                    "source": "mutated",
                    "code": mutated,
                })
                kid += 1

    # Balance safe/unsafe
    safe = [e for e in entries if e["label"] == "safe"]
    unsafe = [e for e in entries if e["label"] == "unsafe"]
    half = n_total // 2
    rng.shuffle(safe)
    rng.shuffle(unsafe)
    selected = safe[:half] + unsafe[:half]
    if len(selected) < n_total:
        extra = safe[half:] + unsafe[half:]
        rng.shuffle(extra)
        selected.extend(extra[: n_total - len(selected)])

    for i, e in enumerate(selected[:n_total]):
        e["id"] = f"k_{i:04d}"
    return selected[:n_total]


def main():
    cfg = load_config()
    n = cfg.get("benchmark_size", 80)
    seed = cfg.get("seed", 42)
    KERNELS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    entries = generate_benchmark(n, seed)
    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        for e in entries:
            path = KERNELS_DIR / f"{e['id']}.tri"
            path.write_text(e["code"], encoding="utf-8")
            row = {
                "id": e["id"],
                "label": e["label"],
                "category": e["category"],
                "error_type": e["error_type"],
                "source": e.get("source", "generated"),
                "path": str(path.relative_to(BENCHMARK_DIR.parent)),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_safe = sum(1 for e in entries if e["label"] == "safe")
    print(f"Benchmark: {len(entries)} kernels ({n_safe} safe, {len(entries)-n_safe} unsafe)")
    print(f"Written to {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
