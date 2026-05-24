"""Anthropic API verifier (optional)."""
import json
import os
import re
import time
from dataclasses import dataclass

from ..paths import PROMPTS_DIR


@dataclass
class VerifyResult:
    safe: bool
    reason: str
    line_of_error: int
    latency_ms: float
    valid_json: bool = True
    backend: str = "anthropic"


class AnthropicVerifier:
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.model = model
        self._prompt_template = (PROMPTS_DIR / "verifier_v1.txt").read_text(encoding="utf-8")
        self._client = None

    @staticmethod
    def is_available() -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def verify(self, code: str) -> VerifyResult:
        prompt = self._prompt_template.replace("{kernel_code}", code)
        t0 = time.perf_counter()
        try:
            msg = self._get_client().messages.create(
                model=self.model,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text if msg.content else ""
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            return VerifyResult(False, str(e), 0, ms, False, "anthropic")
        ms = (time.perf_counter() - t0) * 1000
        parsed = self._parse_json(text)
        if not parsed:
            return VerifyResult(True, "invalid_json", 0, ms, False, "anthropic")
        return VerifyResult(
            bool(parsed.get("safe", True)),
            str(parsed.get("reason", "")),
            int(parsed.get("line_of_error", 0) or 0),
            ms,
            True,
            "anthropic",
        )

    def _parse_json(self, text: str) -> dict | None:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    return None
        return None
