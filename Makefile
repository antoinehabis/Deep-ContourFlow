# Deep ContourFlow — dataset + benchmark pipeline.
#
#   make reproduce      download the datasets, format them, run the full benchmark
#   make help           list every target
#
# Uses ./.venv/bin/python when it exists, otherwise python3. Override with:
#   make eval PYTHON=/path/to/python

PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
EVAL    = $(PYTHON) -m experiments.foreground_extraction.eval

.PHONY: help install download format datasets verify eval quick smoke reproduce clean-results

help:  ## List the available targets
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk -F':.*?## ' '{printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package + benchmark dependencies (editable)
	$(PYTHON) -m pip install -e ".[benchmark]"

download:  ## Download the raw datasets (~1.3 GiB) into datasets/raw/
	$(PYTHON) scripts/download_datasets.py

format:  ## Convert the raw datasets into datasets/formatted/
	$(PYTHON) scripts/format_datasets.py

datasets: download format  ## download + format, in one step

verify:  ## Check the downloaded datasets are complete
	$(PYTHON) scripts/download_datasets.py --verify-only

smoke:  ## 60-second sanity check (4 ECSSD images, 15 steps)
	$(EVAL) 'datasets=[ecssd]' data.max_samples=4 dcf.n_epochs=15

quick:  ## Benchmark on a seeded 200-image subset of each dataset (~7 min on a GPU)
	$(EVAL) data.max_samples=200

eval:  ## Full benchmark on all 17 788 images (~3-4 h on one modern GPU)
	$(EVAL)

reproduce: datasets eval  ## Everything: download, format and run the full benchmark

clean-results:  ## Remove previous evaluation runs (datasets are left untouched)
	rm -rf experiments/foreground_extraction/results
