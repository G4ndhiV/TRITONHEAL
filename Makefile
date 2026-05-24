PYTHON ?= python3
VENV := triton_heal_experiment/.venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: all venv benchmark experiments stats figures pdf clean

all: venv benchmark experiments stats figures pdf

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -q -r triton_heal_experiment/requirements.txt

benchmark:
	cd triton_heal_experiment && $(PY) -m src.benchmark_build

experiments:
	cd triton_heal_experiment && VERIFIER_MODE=hybrid $(PY) -m src.run_experiments

stats:
	cd triton_heal_experiment && $(PY) -m src.stats_analysis

figures:
	cd triton_heal_experiment && $(PY) -m src.plot_results

pdf:
	cd report && tectonic main.tex && mv -f main.pdf main.pdf 2>/dev/null; true
	@test -f report/main.pdf && echo "PDF: report/main.pdf"

clean:
	rm -rf triton_heal_experiment/.venv results/* figures/* report/*.aux report/*.log report/*.out report/*.fls report/*.fdb_latexmk
