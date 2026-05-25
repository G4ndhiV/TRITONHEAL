"""Factory for verifier instances by configuration."""
import os

from dotenv import load_dotenv

from .backends.anthropic_api import AnthropicVerifier
from .backends.heuristic import HeuristicVerifier
from .backends.ollama import OllamaVerifier
from .backends.openai_api import OpenAIVerifier
from .config import load_config
from .triton_heal import TritonHealDual


def _mode() -> str:
    load_dotenv()
    cfg = load_config()
    return os.getenv("VERIFIER_MODE", cfg.get("verifier_mode", "hybrid"))


def get_verifier(config_name: str, cfg: dict | None = None):
    cfg = cfg or load_config()
    mode = _mode()
    host = os.getenv("OLLAMA_HOST", cfg.get("ollama_host", "http://127.0.0.1:11434"))
    models = cfg["models"]
    ollama_ok = OllamaVerifier.is_available(host)

    def _heuristic_local():
        return HeuristicVerifier(tier="local", model_name="llama_heuristic")

    def _heuristic_frontier():
        return HeuristicVerifier(tier="frontier", model_name="claude_heuristic")

    def _ollama_or_heuristic(model_key: str, tier: str):
        tag = models.get(model_key, model_key)
        if mode in ("ollama", "hybrid") and ollama_ok and OllamaVerifier.has_model(tag, host):
            return OllamaVerifier(tag, host=host, timeout=cfg.get("timeout_seconds", 30))
        return HeuristicVerifier(tier=tier, model_name=f"{model_key}_heuristic")

    if config_name == "solo_llama":
        return _ollama_or_heuristic("llama_local", "local")
    if config_name == "solo_deepseek":
        return _ollama_or_heuristic("deepseek_local", "local")
    if config_name == "solo_deepseek_coder":
        return _ollama_or_heuristic("deepseek_frontier", "frontier")
    if config_name == "solo_claude":
        if mode in ("api", "hybrid") and AnthropicVerifier.is_available():
            return AnthropicVerifier(models["claude_frontier"])
        return _heuristic_frontier()
    if config_name == "solo_gpt4o":
        if mode in ("api", "hybrid") and OpenAIVerifier.is_available():
            return OpenAIVerifier(models["gpt_frontier"])
        return _heuristic_frontier()
    if config_name == "triton_heal_dual":
        local = _ollama_or_heuristic("llama_local", "local")
        if mode in ("api", "hybrid") and AnthropicVerifier.is_available():
            frontier = AnthropicVerifier(models["claude_frontier"])
        elif mode in ("api", "hybrid") and OpenAIVerifier.is_available():
            frontier = OpenAIVerifier(models["gpt_frontier"])
        else:
            coder_tag = models.get("deepseek_frontier", "deepseek-coder-v2")
            if mode in ("ollama", "hybrid") and ollama_ok and OllamaVerifier.has_model(coder_tag, host):
                frontier = OllamaVerifier(
                    coder_tag,
                    host=host,
                    timeout=cfg.get("timeout_seconds", 30),
                    prompt_file="verifier_strict_v1.txt",
                )
            else:
                llama_tag = models.get("llama_local", "llama3.1:8b")
                if mode in ("ollama", "hybrid") and ollama_ok and OllamaVerifier.has_model(llama_tag, host):
                    frontier = OllamaVerifier(
                        llama_tag,
                        host=host,
                        timeout=cfg.get("timeout_seconds", 30),
                        prompt_file="verifier_strict_v1.txt",
                    )
                else:
                    frontier = _heuristic_frontier()
        return TritonHealDual(local, frontier)
    raise ValueError(f"Unknown config: {config_name}")


CONFIGS = [
    "solo_llama",
    "solo_deepseek",
    "solo_deepseek_coder",
    "solo_claude",
    "solo_gpt4o",
    "triton_heal_dual",
]

CONFIG_LABELS = {
    "solo_llama": "Llama-3.1-8B (local)",
    "solo_deepseek": "DeepSeek-R1-Distill-Qwen-14B (local)",
    "solo_deepseek_coder": "DeepSeek-Coder-V2 (frontera)",
    "solo_claude": "Claude 3.5 Sonnet (frontera)",
    "solo_gpt4o": "GPT-4o (frontera)",
    "triton_heal_dual": "Triton Heal (dual + veto)",
}
