UV ?= uv
NPM ?= npm
BCG ?= bcg
TEST_ARGS ?=
AGENT_ARGS ?=
CONSTRUCT_ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help
help:
	@printf '%s\n' 'BCG Make targets:'
	@printf '%s\n' '  make install              Install locked Python, Agent, and Dashboard dependencies'
	@printf '%s\n' '  make install-tool         Install the Python bcg command with uv tool'
	@printf '%s\n' '  make agent                Start Graph Construction and open the Agent TUI'
	@printf '%s\n' '  make construct            Run bcg construct with CONSTRUCT_ARGS'
	@printf '%s\n' '  make test                 Run Python and Agent tests'
	@printf '%s\n' '  make check                Run all repository checks and builds'
	@printf '%s\n' '  make test-python          Run the Python test suite'
	@printf '%s\n' '  make build-agent          Build Agent workspace packages and CLI'
	@printf '%s\n' '  make test-agent           Build the Agent and run its tests'
	@printf '%s\n' '  make build-dashboard      Type-check and build the Dashboard'
	@printf '%s\n' '  make check-shell          Syntax-check repository shell scripts'
	@printf '%s\n' '  make check-repository     Check tracked files for local/generated artifacts'
	@printf '%s\n' '  make vllm-server          Start the configured construction-model server'
	@printf '%s\n' '  make sglang-server        Start the configured construction-model server'

.PHONY: install
install:
	@$(UV) sync --locked --all-groups
	@$(NPM) --prefix agent-cli ci
	@$(NPM) --prefix dashboard ci
	@$(NPM) --prefix agent-cli run build

.PHONY: install-tool
install-tool:
	@$(UV) tool install .
	@$(NPM) install -g ./agent-cli

.PHONY: agent
agent:
	@$(BCG) $(AGENT_ARGS)

.PHONY: construct
construct:
	@$(BCG) construct $(CONSTRUCT_ARGS)

.PHONY: lint-python compile-python test-python build-agent test-agent build-dashboard check-shell check-repository test check
lint-python:
	@$(UV) run ruff check .
	@$(UV) run ruff format --check .

compile-python:
	@$(UV) run python -m compileall -q bcg scripts tests

test-python:
	@$(UV) run pytest $(TEST_ARGS)

build-agent:
	@$(NPM) --prefix agent-cli run build

test-agent: build-agent
	@$(NPM) --prefix agent-cli test

build-dashboard:
	@$(NPM) --prefix dashboard run build

check-shell:
	@bash -n install.sh scripts/*.sh

check-repository:
	@scripts/check_repository_hygiene.sh

test: test-python test-agent

check: lint-python compile-python test build-dashboard check-shell check-repository

.PHONY: vllm-server
vllm-server:
	@scripts/start_vllm.sh

.PHONY: sglang-server
sglang-server:
	@scripts/start_sglang_server.sh
