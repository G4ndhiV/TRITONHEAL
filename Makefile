PYTHON ?= python3
ROOT := $(CURDIR)
VENV := $(ROOT)/triton_heal_experiment/.venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: all venv benchmark experiments generation stats generation-stats stats-all figures pdf clean

all: venv benchmark experiments generation stats-all figures pdf

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -q -r triton_heal_experiment/requirements.txt

benchmark:
	cd triton_heal_experiment && $(PY) -m src.benchmark_build
	cd triton_heal_experiment && $(PY) -m src.build_generation_tasks

experiments:
	cd triton_heal_experiment && VERIFIER_MODE=ollama $(PY) -m src.run_experiments

generation:
	cd triton_heal_experiment && VERIFIER_MODE=ollama $(PY) -m src.run_generation

stats:
	cd triton_heal_experiment && $(PY) -m src.stats_analysis

generation-stats:
	cd triton_heal_experiment && $(PY) -m src.generation_stats

stats-all: stats generation-stats

figures:
	cd triton_heal_experiment && $(PY) -m src.plot_results

pdf: figures
	cd report && /opt/homebrew/bin/tectonic main.tex
	@test -f report/main.pdf && echo "PDF: report/main.pdf"

clean:
	rm -rf triton_heal_experiment/.venv results/* figures/* report/*.aux report/*.log report/*.out report/*.fls report/*.fdb_latexmk
