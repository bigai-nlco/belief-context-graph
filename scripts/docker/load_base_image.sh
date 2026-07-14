#!/usr/bin/env bash

set -euo pipefail

BASE_IMAGE_TAR="${BASE_IMAGE_TAR:-/data/docker/images/asllm_1.10.7-poc-pytorch2.9.0-ubuntu24.04-sail2.1.0-cuda13.0-sglang0.5.10-vllm0.19.0-py312.tar}"
BASE_IMAGE="${BASE_IMAGE:-asllm:1.10.7-poc-pytorch2.9.0-ubuntu24.04-sail2.1.0-cuda13.0-sglang0.5.10-vllm0.19.0-py312}"

if docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    echo "[BeliefTracer] Base image already present: ${BASE_IMAGE}"
    exit 0
fi

if [[ ! -f "${BASE_IMAGE_TAR}" ]]; then
    echo "[BeliefTracer] Base image tar not found: ${BASE_IMAGE_TAR}" >&2
    exit 1
fi

echo "[BeliefTracer] Loading base image from ${BASE_IMAGE_TAR}"
docker load -i "${BASE_IMAGE_TAR}"
docker image inspect "${BASE_IMAGE}" >/dev/null
echo "[BeliefTracer] Loaded base image: ${BASE_IMAGE}"
