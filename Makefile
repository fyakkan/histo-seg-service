# Common commands for the project. Run `make <target>`.

PYTHON ?= python3.12
VENV   := .venv
PY     := $(VENV)/bin/python
IMAGE  ?= histo-seg-service:local
MODEL_RELEASE := model-v1
# Scanner images, pinned to versions verified with `docker pull` (same as CI).
TRIVY_IMAGE    := aquasec/trivy:0.76.1
# Local Kubernetes (kind v0.33.0 in .venv/bin; node image digest from the kind release notes).
KIND         := $(VENV)/bin/kind
KIND_CLUSTER := histo-seg
KIND_NODE    := kindest/node:v1.37.0@sha256:a1ed56cfb0e7b93589bdf97c8cd566405a265939e3620fc4f5de89adff580ae5
KUBECTL      := kubectl --context kind-$(KIND_CLUSTER)
# Must match the image in k8s/deployment.yaml, whatever IMAGE is set to (CI sets its own).
K8S_IMAGE    := histo-seg-service:local
GITLEAKS_IMAGE := zricethezav/gitleaks:v8.30.1
# Same Prometheus image as docker-compose.yml; its promtool tests the alert rules.
PROMETHEUS_IMAGE := prom/prometheus:v3.15.1

# Print progress immediately, also when output goes to a log file.
export PYTHONUNBUFFERED := 1

.PHONY: setup data data-download train evaluate mlflow-ui fixtures export model test test-all \
        test-alerts lint security docker-build smoke up down demo-drift k8s-up k8s-down

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

# Unit tests for monitoring/alert_rules.yml (monitoring/alert_rules_test.yml), via Docker.
test-alerts:
	docker run --rm -v "$(CURDIR)/monitoring:/monitoring:ro" --entrypoint promtool \
		$(PROMETHEUS_IMAGE) test rules /monitoring/alert_rules_test.yml

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

# API on 127.0.0.1:8000, Prometheus on 127.0.0.1:9090. Needs artifacts/model.onnx (make model).
up:
	docker compose up -d --build

down:
	docker compose down

# Normal tiles, then stain-shifted tiles, until the StainDriftHigh alert fires (~3 min).
demo-drift:
	$(PY) scripts/demo_drift.py

# Local kind cluster (not a production deployment): build, load the image, deploy 2 replicas,
# wait for the rollout, then smoke-test through a port-forward on 127.0.0.1:8080.
k8s-up:
	$(MAKE) docker-build IMAGE=$(K8S_IMAGE)
	$(KIND) get clusters | grep -qx $(KIND_CLUSTER) || \
		$(KIND) create cluster --name $(KIND_CLUSTER) --image $(KIND_NODE) --wait 120s
	$(KIND) load docker-image $(K8S_IMAGE) --name $(KIND_CLUSTER)
	$(KUBECTL) apply -f k8s/
	$(KUBECTL) rollout restart deployment/histo-seg
	$(KUBECTL) rollout status deployment/histo-seg --timeout=180s
	$(KUBECTL) get pods -l app=histo-seg -o wide
	$(KUBECTL) port-forward service/histo-seg 8080:8000 >/dev/null 2>&1 & pf=$$!; \
		SMOKE_URL=http://127.0.0.1:8080 bash scripts/smoke_test.sh; rc=$$?; kill $$pf; exit $$rc

k8s-down:
	$(KIND) delete cluster --name $(KIND_CLUSTER)
