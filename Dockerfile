ARG BASE_IMAGE=asllm:1.10.7-poc-pytorch2.9.0-ubuntu24.04-sail2.1.0-cuda13.0-sglang0.5.10-vllm0.19.0-py312
FROM ${BASE_IMAGE}

WORKDIR /workspace/belief_tracer

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/belief_tracer:/workspace/rllm \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL=

COPY . .

RUN python -m pip install --no-input --index-url "${PIP_INDEX_URL}" -e .

ENTRYPOINT ["bcg", "agent"]
CMD ["run", "gpqa_diamond", "--model", "/data/share/models/Qwen3-14B"]
