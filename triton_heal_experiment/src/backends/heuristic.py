"""Rule-based verifiers simulating local (lenient) vs frontier (strict) detection tiers."""
import re
import time
from dataclasses import dataclass


@dataclass
class VerifyResult:
    safe: bool
    reason: str
    line_of_error: int
    latency_ms: float
    valid_json: bool = True
    backend: str = "heuristic"


class HeuristicVerifier:
    """Deterministic static checks; tier=local is lenient, tier=frontier is strict."""

    def __init__(self, tier: str = "local", model_name: str = "heuristic"):
        self.tier = tier
        self.model_name = model_name

    def verify(self, code: str) -> VerifyResult:
        t0 = time.perf_counter()
        issues = self._analyze(code)
        safe = len(issues) == 0
        reason = "; ".join(issues) if issues else "no issues detected"
        line = self._first_issue_line(code, issues)
        ms = (time.perf_counter() - t0) * 1000 + (5.0 if self.tier == "local" else 80.0)
        return VerifyResult(
            safe=safe,
            reason=reason,
            line_of_error=line,
            latency_ms=ms,
            backend=f"heuristic_{self.tier}",
        )

    def _analyze(self, code: str) -> list[str]:
        issues: list[str] = []
        lines = code.splitlines()

        if "@triton.jit" not in code and "def " in code:
            issues.append("missing_jit_decorator")

        if self.tier == "local":
            # Lenient: only obvious problems
            if re.search(r"\+\s*999999|999999", code):
                issues.append("memory_oob")
            if code.count("(") != code.count(")"):
                issues.append("unbalanced_parens")
            return issues

        # Frontier / strict tier
        if re.search(r"\+\s*999999|999999", code):
            issues.append("memory_oob")
        if re.search(r"tl\.store\([^)]*\+ offs\)", code) and "mask=" not in code:
            issues.append("memory_unmasked_store")
        for m in re.finditer(r"tl\..arange\(0,\s*(\d+)\)", code):
            bs = int(m.group(1))
            if bs > 0 and (bs & (bs - 1)) != 0:
                issues.append("sram_tiling_misaligned")
        if "non-power-of-two" in code.lower() or "BLOCK=48" in code:
            issues.append("sram_tiling_misaligned")
        if code.count("tl.zeros") >= 3 and "128" in code:
            issues.append("sram_overflow")
        if "mixed FP16/FP32" in code or (
            "float16" in code and "float32" in code and "cast" not in code.lower()
        ):
            issues.append("mixed_precision")
        if "@triton.jit" not in code and "def " in code:
            issues.append("missing_jit_decorator")
        return issues

    def _first_issue_line(self, code: str, issues: list[str]) -> int:
        if not issues:
            return 0
        for i, line in enumerate(code.splitlines(), start=1):
            if any(k in line.lower() for k in ("999999", "float16", "zeros", "unsafe")):
                return i
        return 1
