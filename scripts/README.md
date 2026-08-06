# BCG service scripts and runtime configuration

Covers `scripts/start_vllm.sh`, `scripts/start_sglang_server.sh` and
`scripts/start_tonggraph_server.sh`. All scripts load the repository `.env`
(see `.env.example`), validate inputs and fail fast on missing/invalid
settings. Shared helpers live in `scripts/lib/common.sh`
(`bcg_load_root_env`, `bcg_require_env`, `bcg_validate_port`,
`bcg_check_port_free`, `bcg_maybe_dry_run`).

## Environment variable inventory

| Variable | Required | Default | Secret | Used by |
|---|---|---|---|---|
| `VLLM_MODEL` | yes (vLLM) | — | no | start_vllm |
| `VLLM_PORT` | no | 8001 | no | start_vllm |
| `VLLM_HOST` | no | 0.0.0.0 | no | start_vllm |
| `VLLM_MAX_MODEL_LEN` | no | 65536 | no | start_vllm |
| `VLLM_GPU_MEMORY_UTILIZATION` | no | 0.88 | no | start_vllm |
| `VLLM_VISIBLE_GPUS` | no | 0 | no | start_vllm |
| `VLLM_BIN` | no | `<repo>/.venv/bin/vllm` | no | start_vllm |
| `SGLANG_MODEL` | yes (SGLang) | falls back to `VLLM_MODEL` | no | start_sglang_server |
| `SGLANG_PORT` | no | 8003 (falls back to `VLLM_PORT`) | no | start_sglang_server |
| `SGLANG_HOST` | no | 0.0.0.0 (falls back to `VLLM_HOST`) | no | start_sglang_server |
| `SGLANG_VISIBLE_GPUS` | no | 1 (falls back to `VLLM_VISIBLE_GPUS`/`CUDA_VISIBLE_DEVICES`) | no | start_sglang_server |
| `SGLANG_TP` / `SGLANG_DP` | no | 1 | no | start_sglang_server |
| `SGLANG_MEM_FRACTION_STATIC` | no | 0.90 (falls back to `VLLM_GPU_MEMORY_UTILIZATION`) | no | start_sglang_server |
| `SGLANG_DTYPE` | no | auto (falls back to `VLLM_DTYPE`) | no | start_sglang_server |
| `SGLANG_CONTEXT_LENGTH` | no | SGLang auto | no | start_sglang_server |
| `SGLANG_LOG` | no | `<repo>/var/logs/sglang-*.log` | no | start_sglang_server |
| `TONGGRAPH_SERVER_BIN` | no | `tonggraph-server` on PATH | no | start_tonggraph_server |
| `TONGGRAPH_CONFIG` | no | `<repo>/deploy/tonggraph-server.yml` | no | start_tonggraph_server |
| `TONGGRAPH_DATA_DIR` | no | `<repo>/var/tonggraph` | no | start_tonggraph_server |
| `TONGGRAPH_PORT` | no | 8719 | no | start_tonggraph_server |
| `TONGGRAPH_ADMIN_TOKEN` | yes (TongGraph) | — | **yes** | start_tonggraph_server |
| `TONGGRAPH_AGENT_WRITER_TOKEN` | yes (TongGraph) | — | **yes** | start_tonggraph_server |
| `TONGGRAPH_AGENT_READER_TOKEN` | yes (TongGraph) | — | **yes** | start_tonggraph_server |

Rules:

- Secrets are only read from the environment (`.env` / exported vars); they
  never appear in checked-in configs or generated YAML.
- Ports are validated as integers 1-65535; invalid values fail fast.
- Start/stop operations only target processes confirmed by port **and**
  process characteristics (see the SGLang script's `--stop`).

## Compatibility aliases

`VLLM_*` variables are accepted by the SGLang script as aliases
(`SGLANG_*` wins). These aliases map onto the same internal configuration
and will be removed after one release cycle; no startup logic is duplicated
for them.

## Deploy YAML

`deploy/tonggraph-server.yml` is a portable template: `data_dir` is a
relative default and `TONGGRAPH_DATA_DIR` is injected into a runtime copy
(`var/tonggraph-config.generated.yml`) by `start_tonggraph_server.sh`.
`scripts/check_deploy_yaml.py` smoke-checks the template structure and
verifies tokens are referenced via `token_env` only.

## External components required for production

- vLLM or SGLang serving the construction model (not bundled; model weights
  must be provisioned separately).
- TongGraph server binary with the `server` extra installed
  (`pip install tonggraph[server]` or `TONGGRAPH_SERVER_BIN`).
- A Graph store database directory (default `<repo>/var/tonggraph`).
