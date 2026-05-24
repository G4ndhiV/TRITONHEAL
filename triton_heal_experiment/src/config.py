import json
from pathlib import Path

from .paths import CONFIG_PATH


def load_config(path: Path | None = None) -> dict:
    p = path or CONFIG_PATH
    with open(p, encoding="utf-8") as f:
        return json.load(f)
