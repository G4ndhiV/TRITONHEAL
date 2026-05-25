"""Multi-level compile/syntax checks for generated Triton kernels (M4, no CUDA required)."""
import ast
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .lex_checker import check_lex_python, is_lexically_valid


@dataclass
class CompileResult:
    ast_ok: bool
    lex_ok: bool
    triton_surface_ok: bool
    triton_import_ok: bool
    compile_ok: bool
    check_ms: float
    error: str
    levels_passed: str


def _triton_surface_ok(source: str) -> bool:
    s = source.lower()
    if "import triton" not in s and "from triton" not in s:
        return False
    if "@triton.jit" not in source:
        return False
    if "triton.language" not in s and " as tl" not in s:
        return False
    if "tl.load" not in source and "tl.store" not in source:
        return False
    return True


def _try_triton_import(timeout: float = 10.0) -> tuple[bool, str]:
    script = (
        "import triton\n"
        "import triton.language as tl\n"
        "print('ok')\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return True, ""
        return False, (proc.stderr or proc.stdout or "import failed")[:200]
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)[:200]


def _try_parse_kernel(source: str, timeout: float = 10.0) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(source)
        path = f.name
    script = f"import ast; ast.parse(open({path!r}).read())"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        ok = proc.returncode == 0
        err = (proc.stderr or "")[:200] if not ok else ""
        return ok, err
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)[:200]
    finally:
        Path(path).unlink(missing_ok=True)


def check_compile(source: str, path: Path | None = None) -> CompileResult:
    t0 = time.perf_counter()
    err_parts = []

    ast_ok = False
    try:
        ast.parse(source)
        ast_ok = True
    except SyntaxError as e:
        err_parts.append(f"ast:{e}")

    lex_issues = check_lex_python(source)
    lex_ok = is_lexically_valid(source, path) and "empty_source" not in lex_issues
    if not lex_ok:
        err_parts.append(f"lex:{','.join(lex_issues)}")

    surface = _triton_surface_ok(source)
    if not surface:
        err_parts.append("triton_surface")

    import_ok, import_err = _try_triton_import()
    if not import_ok:
        err_parts.append(f"triton_import:{import_err}")

    # L1-L3 mandatory on M4; L4 (triton import) reported but not required for compile_ok
    compile_ok = ast_ok and lex_ok and surface

    passed = []
    if ast_ok:
        passed.append("L1")
    if lex_ok:
        passed.append("L2")
    if surface:
        passed.append("L3")
    if import_ok:
        passed.append("L4")

    ms = (time.perf_counter() - t0) * 1000
    return CompileResult(
        ast_ok=ast_ok,
        lex_ok=lex_ok,
        triton_surface_ok=surface,
        triton_import_ok=import_ok,
        compile_ok=compile_ok,
        check_ms=ms,
        error="; ".join(err_parts)[:500],
        levels_passed=",".join(passed),
    )


def check_reference_baseline(reference_code: str) -> float:
    return check_compile(reference_code).check_ms
