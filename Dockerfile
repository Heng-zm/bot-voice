FROM python:3.12.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libopus-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --disable-pip-version-check --no-cache-dir -r requirements.txt \
    && python -m pip check

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8080
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -m app.healthcheck

CMD ["python", "-m", "app.main"]
