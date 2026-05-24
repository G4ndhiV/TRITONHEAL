from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
BENCHMARK_DIR = ROOT / "benchmark"
PROMPTS_DIR = ROOT / "prompts"
LEX_DIR = ROOT / "lex"
RESULTS_DIR = PROJECT / "results"
FIGURES_DIR = PROJECT / "figures"
REPORT_TABLES = PROJECT / "report" / "tables"
CONFIG_PATH = ROOT / "experiment_config.json"
GROUND_TRUTH_PATH = BENCHMARK_DIR / "ground_truth.jsonl"
KERNELS_DIR = BENCHMARK_DIR / "kernels"
