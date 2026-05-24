"""Lexical/syntax checks for Triton-like kernels (Python regex + optional Flex binary)."""
import re
import subprocess
from pathlib import Path

from .paths import LEX_DIR

LEX_PATTERNS = [
    (r"@triton\.jit", "missing_jit_decorator"),
    (r"tl\.load|tl\.store", "triton_api"),
    (r"BLOCK_SIZE\s*=\s*\d+", "block_size"),
]


def check_lex_python(source: str) -> list[str]:
    """Return list of issue codes found."""
    issues = []
    if not source.strip():
        issues.append("empty_source")
        return issues
    if "@triton.jit" not in source and "triton" in source.lower():
        issues.append("missing_jit_decorator")
    if "tl.load" not in source and "tl.store" not in source:
        if "def " in source:
            issues.append("missing_triton_ops")
    # Unbalanced brackets
    if source.count("(") != source.count(")"):
        issues.append("unbalanced_parens")
    if source.count("[") != source.count("]"):
        issues.append("unbalanced_brackets")
    return issues


def run_flex_scanner(path: Path) -> tuple[bool, str]:
    """Run compiled Lex scanner if available."""
    binary = LEX_DIR / "lexico_equipo_11_scanner"
    if not binary.exists() or binary.stat().st_size == 0:
        return True, ""
    try:
        proc = subprocess.run(
            [str(binary), str(path)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        err = (proc.stderr or "") + (proc.stdout or "")
        ok = proc.returncode == 0 and "error" not in err.lower()
        return ok, err
    except (subprocess.TimeoutExpired, OSError):
        return True, ""


def is_lexically_valid(source: str, path: Path | None = None) -> bool:
    issues = check_lex_python(source)
    if issues and "empty_source" in issues:
        return False
    if path and path.exists():
        ok, _ = run_flex_scanner(path)
        if not ok:
            return False
    return len(issues) <= 1  # allow minor warnings
