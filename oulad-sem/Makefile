PYTHON ?= python3
CONFIG ?= config/config.yaml
STAGES = S00 S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 S11 S12

sim:
	@RUN_ID="sim-$$(date -u +%Y%m%dT%H%M%SZ)-$$(openssl rand -hex 3)"; \
	echo "$$RUN_ID" > .last_sim_run; echo "simulation run $$RUN_ID"; \
	for S in $(STAGES); do \
	  PYTHONHASHSEED=0 $(PYTHON) scripts/run_stage.py --stage $$S --mode simulation \
	    --run-id $$RUN_ID --config $(CONFIG) || exit 1; \
	done

full:
	@RUN_ID="run-$$(date -u +%Y%m%dT%H%M%SZ)-$$(openssl rand -hex 3)"; \
	echo "$$RUN_ID" > .last_full_run; echo "full run $$RUN_ID"; \
	for S in $(STAGES); do \
	  PYTHONHASHSEED=0 $(PYTHON) scripts/run_stage.py --stage $$S --mode full \
	    --run-id $$RUN_ID --config $(CONFIG) || exit 1; \
	done

validate:
	@RUN=$${RUN_ID:-$$(cat .last_full_run 2>/dev/null || cat .last_sim_run)}; \
	echo "validating $$RUN"; $(PYTHON) scripts/validate_run.py --run-id $$RUN

test:
	$(PYTHON) -m pytest -q

.PHONY: sim full validate test
