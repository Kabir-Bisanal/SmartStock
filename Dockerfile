FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY pyproject.toml requirements-app.txt README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-app.txt

COPY app ./app
COPY config ./config
COPY sql ./sql

RUN mkdir -p \
      data/interim \
      data/processed/production \
      data/processed/inventory \
      data/simulated \
      models && \
    groupadd --system smartstock && \
    useradd --system --gid smartstock --home-dir /app smartstock && \
    chown -R smartstock:smartstock /app

USER smartstock

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py"]
