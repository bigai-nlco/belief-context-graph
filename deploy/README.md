# deploy/

Server deployment templates and runtime configuration.

- `tonggraph-server.yml` — portable TongGraph server template; the effective
  `data_dir` is injected by `scripts/start_tonggraph_server.sh` from
  `TONGGRAPH_DATA_DIR` (see `scripts/README.md` for the full environment
  variable inventory). `scripts/check_deploy_yaml.py` validates its
  structure and that tokens are `token_env` references only.
