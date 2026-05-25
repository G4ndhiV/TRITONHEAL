"""Extract Python kernel code from LLM responses."""
import re


def extract_python_code(text: str) -> str:
    if not text or not text.strip():
        return ""
    # Prefer fenced python blocks
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # Strip common prefixes
    lines = text.strip().splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(("import ", "from ", "@triton")):
            start = i
            break
    return "\n".join(lines[start:]).strip()
