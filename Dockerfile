FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8791

# 默认只绑本机回环；对外暴露前请自行加认证与限流
CMD ["python", "-m", "uvicorn", "agentloom.api.app:app", "--host", "127.0.0.1", "--port", "8791"]
