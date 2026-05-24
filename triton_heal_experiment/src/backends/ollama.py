"""Ollama API backend for local LLM verification."""
import json
import re
import time
from dataclasses import dataclass

import requests

from ..paths import PROMPTS_DIR


@dataclass
class VerifyResult:
    safe: bool
    reason: str
    line_of_error: int
    latency_ms: float
    valid_json: bool = True
    backend: str = "ollama"


class OllamaVerifier:
    def __init__(
        self,
        model: str,
        host: str = "http://127.0.0.1:11434",
        timeout: int = 30,
        prompt_file: str = "verifier_v1.txt",
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._prompt_template = (PROMPTS_DIR / prompt_file).read_text(encoding="utf-8")

    @staticmethod
    def is_available(host: str = "http://127.0.0.1:11434") -> bool:
        try:
            r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    @staticmethod
    def has_model(model: str, host: str = "http://127.0.0.1:11434") -> bool:
        try:
            r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=5)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return any(model in n or n.startswith(model) for n in names)
        except requests.RequestException:
            return False

    def verify(self, code: str) -> VerifyResult:
        prompt = self._prompt_template.replace("{kernel_code}", code)
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 512},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")
        except requests.RequestException as e:
            ms = (time.perf_counter() - t0) * 1000
            return VerifyResult(
                safe=False,
                reason=f"ollama_error: {e}",
                line_of_error=0,
                latency_ms=ms,
                valid_json=False,
                backend="ollama",
            )
        ms = (time.perf_counter() - t0) * 1000
        parsed = self._parse_json(text)
        if parsed is None:
            return VerifyResult(
                safe=True,
                reason="invalid_json_response",
                line_of_error=0,
                latency_ms=ms,
                valid_json=False,
                backend="ollama",
            )
        return VerifyResult(
            safe=bool(parsed.get("safe", True)),
            reason=str(parsed.get("reason", "")),
            line_of_error=int(parsed.get("line_of_error", 0) or 0),
            latency_ms=ms,
            valid_json=True,
            backend="ollama",
        )

    def _parse_json(self, text: str) -> dict | None:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{[^{}]*\"safe\"[^{}]*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None
