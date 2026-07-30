UV ?= uv
BCG ?= bcg
TEST_ARGS ?=
AGENT_ARGS ?=
CONSTRUCT_ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help
help:
	@printf '%s\n' 'BCG Make targets:'
	@printf '%s\n' '  make install              Create/update the uv environment'
	@printf '%s\n' '  make install-tool         Install the Python bcg command with uv tool'
	@printf '%s\n' '  make agent                Start Graph Construction and open the Agent TUI'
	@printf '%s\n' '  make construct            Run bcg construct with CONSTRUCT_ARGS'
	@printf '%s\n' '  make test                 Run the Python test suite'
	@printf '%s\n' '  make vllm-server          Start the configured construction-model server'
	@printf '%s\n' '  make sglang-server        Start the configured construction-model server'

.PHONY: install
install:
	@$(UV) sync
	@npm --prefix agent-cli install
	@npm --prefix agent-cli run build

.PHONY: install-tool
install-tool:
	@$(UV) tool install .
	@npm install -g ./agent-cli

.PHONY: agent
agent:
	@$(BCG) $(AGENT_ARGS)

.PHONY: construct
construct:
	@$(BCG) construct $(CONSTRUCT_ARGS)

.PHONY: test
test:
	@$(UV) run pytest $(TEST_ARGS)

.PHONY: vllm-server
vllm-server:
	@scripts/start_vllm.sh

.PHONY: sglang-server
sglang-server:
	@scripts/start_sglang_server.sh
