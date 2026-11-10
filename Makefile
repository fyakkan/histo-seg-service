# Common commands for the project. Run `make <target>`.
# Targets that print "TODO" are filled in by later parts of the project.

PYTHON ?= python3.12
VENV   := .venv
PY     := $(VENV)/bin/python

# Print progress immediately, also when output goes to a log file.
export PYTHONUNBUFFERED := 1

.PHONY: setup data data-download train evaluate mlflow-ui fixtures export test test-all lint security \
        docker-build smoke up down demo-drift

setup:  ## create the virtual environment and install the project
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements/train.txt -r requirements/dev.txt
	$(PY) -m pip install --no-deps -e .

# Resumable. Use `make data-download LIMIT=2` for a quick test with 2 ROIs.
data-download:
	$(PY) -m histoseg.data.download --config configs/data.yaml $(if $(LIMIT),--limit $(LIMIT))

# Download (skips what is already there), split patients, cut tiles, write the data report.
data: data-download
	$(PY) -m histoseg.data.split --config configs/data.yaml
	$(PY) -m histoseg.data.tiling --config configs/data.yaml
	$(PY) -m histoseg.data.report --config configs/data.yaml

# Extra options go in ARGS, e.g. `make train ARGS="--epochs 1 --no-register"` to time one epoch.
train:
	$(PY) -m histoseg.train --config configs/train.yaml $(ARGS)

evaluate:
	$(PY) -m histoseg.evaluate --config configs/train.yaml

mlflow-ui:
	$(VENV)/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

fixtures:
	$(PY) scripts/make_fixtures.py --config configs/data.yaml

export:
	@echo "TODO: export the model to ONNX"

test:
	$(PY) -m pytest -m "not train"

test-all:
	$(PY) -m pytest

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

security:
	@echo "TODO: bandit, pip-audit, gitleaks, trivy"

docker-build:
	@echo "TODO: docker build"

smoke:
	@echo "TODO: container smoke test"

up:
	@echo "TODO: docker compose up"

down:
	@echo "TODO: docker compose down"

demo-drift:
	@echo "TODO: stain-drift demo"
