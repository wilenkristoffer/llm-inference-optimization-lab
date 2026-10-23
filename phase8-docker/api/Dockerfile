# Phase 4 API in a container. Build context: phase4-observability/
#   docker build -f phase8-docker/api/Dockerfile -t lab/api phase4-observability
# Standard library only, so no pip install.
FROM python:3.12-slim

WORKDIR /app
COPY api.py servers.py ./

# lab.db is a read-only bind mount at /data, shared with the native scripts.
# The model servers stay native and are reached through host.docker.internal;
# there is nothing to spawn them into from here, so control is off.
ENV LAB_DB=/data/lab.db \
    API_HOST=0.0.0.0 \
    LAB_SERVER_CONTROL=0 \
    OLLAMA_URL=http://host.docker.internal:11434 \
    VLLM_URL=http://host.docker.internal:8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8010
CMD ["python", "api.py"]
