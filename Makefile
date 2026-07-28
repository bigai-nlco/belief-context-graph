PYTHON ?= python
PIP ?= pip
BT ?= bcg agent

BT_CONTAINER ?= belief_tracer
BT_UI_HOST ?= 172.25.10.2
BT_UI_PORT ?= 23456
BT_ARTIFACTS_DIR ?= artifacts/belief_tracer
BT_WORKDIR ?=
BT_UI_LOG ?= /tmp/belief_tracer-ui.log

RUN_ARGS ?=
UI_ARGS ?=
CHECK_THINKING_ARGS ?=
TEST_ARGS ?=
ROLLOUT_ARGS ?=
GRAPH_ARGS ?=
DOCKER_ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help
help:
	@printf '%s\n' 'BeliefTracer Make targets:'
	@printf '%s\n' '  make install              Install package in editable mode'
	@printf '%s\n' '  make version              Print BeliefTracer version'
	@printf '%s\n' '  make tasks                List supported benchmark tasks'
	@printf '%s\n' '  make run RUN_ARGS="..."   Run bcg agent run with arguments'
	@printf '%s\n' '  make run-help             Show rollout help'
	@printf '%s\n' '  make ui                   Start local UI in the foreground'
	@printf '%s\n' '  make ui-help              Show UI help'
	@printf '%s\n' '  make restart-ui           Restart detached UI in Docker container'
	@printf '%s\n' '  make ui-log               Tail detached UI log from Docker container'
	@printf '%s\n' '  make check-thinking CHECK_THINKING_ARGS="..."'
	@printf '%s\n' '  make test                 Run pytest'
	@printf '%s\n' '  make tonggraph-server     Start TongGraph Server for graph persistence'
	@printf '%s\n' '  make tonggraph-sync GRAPH_ARGS="path/to/final_graph.json"'
	@printf '%s\n' '  make rollout ROLLOUT_ARGS="..."'
	@printf '%s\n' '  make docker-build'
	@printf '%s\n' '  make docker-up'
	@printf '%s\n' '  make docker-run DOCKER_ARGS="run math500 --model MODEL"'
	@printf '%s\n' '  make docker-ui            Start UI through docker compose run'
	@printf '%s\n' '  make docker-shell         Open a shell in the running container'
	@printf '%s\n' '  make docker-ps            Show BeliefTracer Docker container status'

.PHONY: install
install:
	@$(PIP) install -e .

.PHONY: version
version:
	@$(BT) --version

.PHONY: tasks
tasks:
	@$(BT) tasks

.PHONY: run
run:
	@$(BT) run $(RUN_ARGS)

.PHONY: run-help
run-help:
	@$(BT) run --help

.PHONY: ui
ui:
	@$(BT) ui \
		--host "$(BT_UI_HOST)" \
		--port "$(BT_UI_PORT)" \
		--artifacts-dir "$(BT_ARTIFACTS_DIR)" \
		$(UI_ARGS)

.PHONY: ui-help
ui-help:
	@$(BT) ui --help

.PHONY: restart-ui
restart-ui:
	@BT_CONTAINER="$(BT_CONTAINER)" \
	BT_UI_HOST="$(BT_UI_HOST)" \
	BT_UI_PORT="$(BT_UI_PORT)" \
	BT_ARTIFACTS_DIR="$(BT_ARTIFACTS_DIR)" \
	BT_UI_LOG="$(BT_UI_LOG)" \
	BT_WORKDIR="$(BT_WORKDIR)" \
	scripts/restart_ui.sh

.PHONY: ui-log
ui-log:
	@docker exec "$(BT_CONTAINER)" sh -lc 'tail -n 100 -f "$(BT_UI_LOG)"'

.PHONY: check-thinking
check-thinking:
	@$(BT) check-thinking $(CHECK_THINKING_ARGS)

.PHONY: test
test:
	@$(PYTHON) -m pytest $(TEST_ARGS)

.PHONY: tonggraph-server
tonggraph-server:
	@scripts/start_tonggraph_server.sh

.PHONY: tonggraph-sync
tonggraph-sync:
	@$(BT) tonggraph-sync $(GRAPH_ARGS)

.PHONY: rollout
rollout:
	@scripts/rollout.sh $(ROLLOUT_ARGS)

.PHONY: docker-build
docker-build:
	@docker compose build

.PHONY: docker-up
docker-up:
	@docker compose up --build

.PHONY: docker-run
docker-run:
	@docker compose run --rm belief_tracer $(DOCKER_ARGS)

.PHONY: docker-ui
docker-ui:
	@docker compose run --rm belief_tracer ui \
		--host "$(BT_UI_HOST)" \
		--port "$(BT_UI_PORT)" \
		--artifacts-dir "$(BT_ARTIFACTS_DIR)" \
		$(UI_ARGS)

.PHONY: docker-shell
docker-shell:
	@docker exec -it "$(BT_CONTAINER)" sh

.PHONY: docker-ps
docker-ps:
	@docker ps -a --filter "name=^/$(BT_CONTAINER)$$"
