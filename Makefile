# Common commands for the project. Run `make <target>`.
# Targets that print "TODO" are filled in by later parts of the project.

PYTHON ?= python3.12
VENV   := .venv
PY     := $(VENV)/bin/python
IMAGE  ?= histo-seg-service:local
MODEL_RELEASE := model-v1
# Scanner images, pinned to versions verified with `docker pull` (same as CI).
TRIVY_IMAGE    := aquasec/trivy:0.76.1
GITLEAKS_IMAGE := zricethezav/gitleaks:v8.30.1

# Print progress immediately, also when output goes to a log file.
export PYTHONUNBUFFERED := 1

.PHONY: setup data data-download train evaluate mlflow-ui fixtures export model test test-all \
        lint security docker-build smoke up down demo-drift

setup:  ## create the virtual environment and install the project
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements/train.txt -r requirements/serve.txt -r requirements/dev.txt
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
	$(PY) -m histoseg.export_onnx --config configs/train.yaml

# Download the released model instead of training it, and verify its checksum.
model:
	gh release download $(MODEL_RELEASE) --dir artifacts --clobber \
		--pattern model.onnx --pattern model.onnx.sha256 --pattern model_card.json
	cd artifacts && shasum -a 256 -c model.onnx.sha256

test:
	$(PY) -m pytest -m "not train"

test-all:
	$(PY) -m pytest

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

# The same four scanners as CI. Trivy needs the image from `make docker-build`.
security:
	$(VENV)/bin/bandit -r src -ll
	$(VENV)/bin/pip-audit -r requirements/serve.txt -r requirements/dev.txt
	$(VENV)/bin/pip-audit --no-deps -r requirements/train.txt
	docker run --rm -v "$(CURDIR):/repo" $(GITLEAKS_IMAGE) git /repo --redact
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		-v "$(HOME)/.cache/trivy:/root/.cache/trivy" $(TRIVY_IMAGE) \
		image --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 $(IMAGE)

docker-build:
	docker build -t $(IMAGE) .
	@docker image ls $(IMAGE) --format 'Image size: {{.Size}}'

smoke:
	bash scripts/smoke_test.sh $(IMAGE)

up:
	@echo "TODO: docker compose up"

down:
	@echo "TODO: docker compose down"

demo-drift:
	@echo "TODO: stain-drift demo"
