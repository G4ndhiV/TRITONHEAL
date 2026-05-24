"""OpenAI API verifier (optional)."""
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
    backend: str = "openai"


class OpenAIVerifier:
    def __init__(self, model: str = "gpt-4o-2024-05-13"):
        self.model = model
        self._prompt_template = (PROMPTS_DIR / "verifier_v1.txt").read_text(encoding="utf-8")
        self._client = None

    @staticmethod
    def is_available() -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def verify(self, code: str) -> VerifyResult:
        prompt = self._prompt_template.replace("{kernel_code}", code)
        t0 = time.perf_counter()
        try:
            resp = self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Respond only with JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            return VerifyResult(False, str(e), 0, ms, False, "openai")
        ms = (time.perf_counter() - t0) * 1000
        parsed = self._parse_json(text)
        if not parsed:
            return VerifyResult(True, "invalid_json", 0, ms, False, "openai")
        return VerifyResult(
            bool(parsed.get("safe", True)),
            str(parsed.get("reason", "")),
            int(parsed.get("line_of_error", 0) or 0),
            ms,
            True,
            "openai",
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
