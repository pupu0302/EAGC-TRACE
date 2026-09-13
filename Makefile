PYTHON ?= python3
export PYTHONDONTWRITEBYTECODE := 1

# Default output directory for reproduce/reproduce_table1.py. Override on the
# command line, e.g.: make reproduce-table1 OUTPUT_DIR=/tmp/eagc-reproduced
# Never defaults into this package's own source tree.
OUTPUT_DIR ?= $(CURDIR)/reproduced_results

.PHONY: validate-data test reproduce-table1 reproduce-full-stats reproduce-primary-channel all

validate-data:
	$(PYTHON) reproduce/validate_data.py

test:
	$(PYTHON) -m pytest tests/ -q -p no:cacheprovider

reproduce-table1:
	$(PYTHON) reproduce/reproduce_table1.py --output-dir "$(OUTPUT_DIR)"

reproduce-full-stats:
	$(PYTHON) reproduce/reproduce_table1.py --full-stats --output-dir "$(OUTPUT_DIR)"

reproduce-primary-channel:
	$(PYTHON) reproduce/reproduce_primary_channel_metrics.py --output-dir "$(OUTPUT_DIR)"

all: validate-data test reproduce-table1 reproduce-primary-channel
