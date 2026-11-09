# Common commands for the project. Run `make <target>`.
# Targets that print "TODO" are filled in by later parts of the project.

PYTHON ?= python3.12
VENV   := .venv
PY     := $(VENV)/bin/python

.PHONY: setup data train evaluate export test test-all lint security \
        docker-build smoke up down demo-drift

setup:  ## create the virtual environment
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

data:
	@echo "TODO: download BCSS and cut tiles"

train:
	@echo "TODO: train the model"

evaluate:
	@echo "TODO: evaluate the champion model"

export:
	@echo "TODO: export the model to ONNX"

test:
	@echo "TODO: pytest -m 'not train'"

test-all:
	@echo "TODO: pytest (all tests)"

lint:
	@echo "TODO: ruff check + ruff format --check"

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
