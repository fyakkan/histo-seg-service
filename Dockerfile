# Two stages: the builder installs the Python packages, the runtime image only copies the result,
# so pip, build caches and other install-time files never reach the final image.

FROM python:3.12.16-slim-trixie AS builder
RUN python -m venv /opt/venv
COPY requirements/serve.txt /tmp/serve.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/serve.txt


FROM python:3.12.16-slim-trixie

# Run as an unprivileged user, never as root.
RUN useradd --system --uid 10001 --no-create-home app

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY src/ src/
COPY configs/serve.yaml configs/serve.yaml
COPY artifacts/model.onnx artifacts/model_card.json artifacts/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000

# The slim image has no curl, so the health check uses Python's standard library.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "histoseg.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
