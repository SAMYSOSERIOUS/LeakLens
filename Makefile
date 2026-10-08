PY ?= python

.PHONY: install sample prepare audit report replay replay-loop test serve clean

install:            ## install the package and test tools
	$(PY) -m pip install -e ".[dev]"

sample:             ## make the small made-up dataset (KKBox format)
	$(PY) -m leaklens make-sample

prepare:            ## raw KKBox CSVs in data/kkbox/raw -> data/kkbox/prepared
	$(PY) -m leaklens prepare --raw data/kkbox/raw --out data/kkbox/prepared

audit:              ## steps 1-6: re-test notebooks, honest models, euros, reports
	$(PY) -m leaklens audit

report:             ## rewrite README results + manager summary from reports/audit.json
	$(PY) -m leaklens report

replay:             ## one run of the weekly job
	$(PY) -m leaklens replay-step

replay-loop:        ## run 15 months of the replay at once (fills the monitor)
	$(PY) -m leaklens replay-run --steps 15

test:
	$(PY) -m pytest

serve:              ## open the monitor at http://localhost:8000
	cd site && $(PY) -m http.server 8000

clean:
	rm -rf state site/data/monitor.json .pytest_cache
