"""LLM kernel generators per experimental configuration."""
import os
import time
from dataclasses import dataclass

import requests

from .backends.anthropic_api import AnthropicVerifier
from .backends.openai_api import OpenAIVerifier
from .backends.ollama import OllamaVerifier
from .code_extract import extract_python_code
from .config import load_config
from .paths import PROMPTS_DIR

CATEGORY_TEMPLATES = {
    "elementwise": """import triton
import triton.language as tl

@triton.jit
def generated_elem(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)
""",
    "softmax": """import triton
import triton.language as tl

@triton.jit
def generated_softmax(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)
""",
    "matmul": """import triton
import triton.language as tl

@triton.jit
def generated_matmul(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    a = tl.load(a_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a, mask=mask)
""",
    "reduction": """import triton
import triton.language as tl

@triton.jit
def generated_reduce(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + pid, x, mask=mask)
""",
    "flash_attention": """import triton
import triton.language as tl

@triton.jit
def generated_flash(q_ptr, k_ptr, v_ptr, out_ptr, seq_len, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < seq_len
    q = tl.load(q_ptr + offs, mask=mask)
    k = tl.load(k_ptr + offs, mask=mask)
    v = tl.load(v_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, q * k * v, mask=mask)
""",
}


@dataclass
class GenerateResult:
    code: str
    latency_ms: float
    backend: str
    error: str = ""


class OllamaGenerator:
    def __init__(self, model: str, host: str, timeout: int, prompt_file: str = "generator_v1.txt"):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._template = (PROMPTS_DIR / prompt_file).read_text(encoding="utf-8")
        self._repair_template = (PROMPTS_DIR / "generator_repair_v1.txt").read_text(encoding="utf-8")

    def generate(self, task_spec: str) -> GenerateResult:
        prompt = self._template.replace("{task_spec}", task_spec)
        return self._call(prompt, "ollama")

    def repair(self, task_spec: str, broken_code: str, compile_error: str) -> GenerateResult:
        prompt = (
            self._repair_template.replace("{task_spec}", task_spec)
            .replace("{broken_code}", broken_code[:4000])
            .replace("{compile_error}", compile_error[:500])
        )
        return self._call(prompt, "ollama_repair")

    def _call(self, prompt: str, backend: str) -> GenerateResult:
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 1024},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")
        except requests.RequestException as e:
            ms = (time.perf_counter() - t0) * 1000
            return GenerateResult("", ms, backend, str(e))
        ms = (time.perf_counter() - t0) * 1000
        return GenerateResult(extract_python_code(text), ms, backend)


class HeuristicGenerator:
    def __init__(self, category: str = "elementwise"):
        self.category = category

    def generate(self, task_spec: str) -> GenerateResult:
        t0 = time.perf_counter()
        cat = "elementwise"
        for c in CATEGORY_TEMPLATES:
            if c in task_spec.lower():
                cat = c
                break
        code = CATEGORY_TEMPLATES.get(cat, CATEGORY_TEMPLATES["elementwise"])
        ms = (time.perf_counter() - t0) * 1000 + 50.0
        return GenerateResult(code, ms, "heuristic_template")

    def repair(self, task_spec: str, broken_code: str, compile_error: str) -> GenerateResult:
        return self.generate(task_spec)


class OpenAIGenerator:
    def __init__(self, model: str):
        self.model = model
        self._template = (PROMPTS_DIR / "generator_v1.txt").read_text(encoding="utf-8")

    def generate(self, task_spec: str) -> GenerateResult:
        if not OpenAIVerifier.is_available():
            return HeuristicGenerator().generate(task_spec)
        from openai import OpenAI

        prompt = self._template.replace("{task_spec}", task_spec)
        t0 = time.perf_counter()
        try:
            resp = OpenAI().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Output only Python Triton kernel code."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            return GenerateResult("", ms, "openai", str(e))
        ms = (time.perf_counter() - t0) * 1000
        return GenerateResult(extract_python_code(text), ms, "openai")

    def repair(self, task_spec: str, broken_code: str, compile_error: str) -> GenerateResult:
        gen = OllamaGenerator("gpt-4o", "http://127.0.0.1:11434", 60)
        gen._repair_template = (PROMPTS_DIR / "generator_repair_v1.txt").read_text(encoding="utf-8")
        if not OpenAIVerifier.is_available():
            return HeuristicGenerator().generate(task_spec)
        return gen.repair(task_spec, broken_code, compile_error)


class AnthropicGenerator:
    def __init__(self, model: str):
        self.model = model
        self._template = (PROMPTS_DIR / "generator_v1.txt").read_text(encoding="utf-8")

    def generate(self, task_spec: str) -> GenerateResult:
        if not AnthropicVerifier.is_available():
            return HeuristicGenerator().generate(task_spec)
        import anthropic

        prompt = self._template.replace("{task_spec}", task_spec)
        t0 = time.perf_counter()
        try:
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text if msg.content else ""
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            return GenerateResult("", ms, "anthropic", str(e))
        ms = (time.perf_counter() - t0) * 1000
        return GenerateResult(extract_python_code(text), ms, "anthropic")

    def repair(self, task_spec: str, broken_code: str, compile_error: str) -> GenerateResult:
        if not AnthropicVerifier.is_available():
            return HeuristicGenerator().generate(task_spec)
        return self.generate(task_spec)


def get_generator(config_name: str, cfg: dict | None = None):
    cfg = cfg or load_config()
    host = os.getenv("OLLAMA_HOST", cfg.get("ollama_host", "http://127.0.0.1:11434"))
    models = cfg["models"]
    timeout = cfg.get("timeout_seconds", 120)
    mode = os.getenv("VERIFIER_MODE", cfg.get("verifier_mode", "hybrid"))
    ollama_ok = OllamaVerifier.is_available(host)

    def ollama_gen(key: str):
        tag = models.get(key, key)
        if mode in ("ollama", "hybrid") and ollama_ok and OllamaVerifier.has_model(tag, host):
            return OllamaGenerator(tag, host, timeout)
        return HeuristicGenerator()

    if config_name == "solo_llama":
        return ollama_gen("llama_local")
    if config_name == "solo_deepseek":
        return ollama_gen("deepseek_local")
    if config_name == "solo_deepseek_coder":
        return ollama_gen("deepseek_frontier")
    if config_name == "solo_claude":
        if mode in ("api", "hybrid") and AnthropicVerifier.is_available():
            return AnthropicGenerator(models["claude_frontier"])
        return HeuristicGenerator()
    if config_name == "solo_gpt4o":
        if mode in ("api", "hybrid") and OpenAIVerifier.is_available():
            return OpenAIGenerator(models["gpt_frontier"])
        return HeuristicGenerator()
    if config_name == "triton_heal_dual":
        local = ollama_gen("llama_local")
        coder_tag = models.get("deepseek_frontier", "deepseek-coder-v2")
        if mode in ("ollama", "hybrid") and ollama_ok and OllamaVerifier.has_model(coder_tag, host):
            repair = OllamaGenerator(coder_tag, host, timeout)
        else:
            repair = HeuristicGenerator()
        return DualGenerator(local, repair)
    raise ValueError(config_name)


class DualGenerator:
    """Generate with local SLM, repair with frontier if compile fails."""

    def __init__(self, local, repair, max_retries: int = 2):
        self.local = local
        self.repair = repair
        self.max_retries = max_retries

    def generate(self, task_spec: str) -> GenerateResult:
        return self.generate_with_repair(task_spec)

    def generate_with_repair(self, task_spec: str) -> tuple[GenerateResult, int]:
        t0 = time.perf_counter()
        out = self.local.generate(task_spec)
        retries = 0
        total_ms = out.latency_ms
        code = out.code
        for _ in range(self.max_retries):
            from .compile_checker import check_compile

            cr = check_compile(code)
            if cr.compile_ok:
                break
            retries += 1
            rep = self.repair.repair(task_spec, code, cr.error)
            total_ms += rep.latency_ms
            if rep.code:
                code = rep.code
        ms = (time.perf_counter() - t0) * 1000
        return (
            GenerateResult(code, total_ms, "triton_heal_dual_gen"),
            retries,
        )

    def repair(self, task_spec: str, broken_code: str, compile_error: str) -> GenerateResult:
        return self.repair.repair(task_spec, broken_code, compile_error)
