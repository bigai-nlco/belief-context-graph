"""Ray-distributed vLLM rollout engine for agent workflow inference.

Spawns ``num_replicas`` Ray actors, each owning its own in-process
``VLLMEngine`` on ``tensor_parallel_size`` GPUs. Implements the rllm
``RolloutEngine`` interface so it drops in wherever ``VLLMEngine`` does —
``AgentWorkflowEngine`` dispatches concurrent ``get_model_response``
calls across the pool, and each replica batches its share internally
via vLLM's ``AsyncLLMEngine``.

Aggregate concurrency = ``num_replicas`` × (per-replica vLLM batching).
``n_parallel_tasks`` in the workflow engine should be sized to keep all
replicas busy.
"""

from __future__ import annotations

import asyncio
import itertools
import os
from typing import Any

import ray

from bcg.agent.tokenizer_compat import (
    configure_parser_for_qwen_thinking,
    load_tokenizer_compat,
)
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser


def _ensure_ray(address: str | None) -> None:
    """Initialize Ray once; respect RAY_ADDRESS or explicit address."""
    if ray.is_initialized():
        return
    # Match Ray's upcoming behavior for zero-GPU driver workers and silence
    # Ray 2.47's FutureWarning without filtering unrelated warnings.
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    kwargs: dict[str, Any] = {"ignore_reinit_error": True, "log_to_driver": False}
    addr = address or os.environ.get("RAY_ADDRESS")
    if addr:
        kwargs["address"] = addr
    ray.init(**kwargs)


def _safe_signal_context():
    """Neutralize ``signal.signal`` calls from non-main threads.

    Ray actor methods run off the main interpreter thread, but rllm /
    vLLM import chains register SIGALRM handlers at module load time —
    which raises ``ValueError: signal only works in main thread`` inside
    the actor. We monkey-patch ``signal.signal`` for the duration of the
    risky import/initialization so those calls become no-ops off-thread.
    """
    import signal
    import threading
    from contextlib import contextmanager

    @contextmanager
    def _mgr():
        original_signal = signal.signal

        def _safe_signal(signum, handler):
            if threading.current_thread() is threading.main_thread():
                try:
                    return original_signal(signum, handler)
                except ValueError:
                    return None
            return None

        signal.signal = _safe_signal
        try:
            yield
        finally:
            signal.signal = original_signal

    return _mgr()


def _build_actor_cls(tensor_parallel_size: int):
    """Build a Ray remote actor class sized for the requested TP degree.

    Each actor receives ``num_gpus=tensor_parallel_size`` so Ray's
    scheduler places replicas on disjoint GPU sets. vLLM's multiprocess
    executor then fans TP workers across those GPUs inside the actor.
    """

    @ray.remote(num_gpus=tensor_parallel_size, num_cpus=1)
    class RolloutActor:
        def __init__(self, engine_kwargs: dict) -> None:
            with _safe_signal_context():
                from bcg.agent.vllm_engine import VLLMEngine

                self.engine = VLLMEngine(**engine_kwargs)

        async def generate(self, messages: list[dict], kwargs: dict) -> dict:
            out: ModelOutput = await self.engine.get_model_response(messages, **kwargs)
            return out.to_dict()

        async def wake_up(self) -> None:
            await self.engine.wake_up()

        async def sleep(self) -> None:
            await self.engine.sleep()

        def shutdown(self) -> None:
            self.engine.shutdown()

        def ping(self) -> bool:
            return True

    return RolloutActor


class RayVLLMEngine(RolloutEngine):
    """Pool of Ray-hosted ``VLLMEngine`` replicas behind the ``RolloutEngine`` API.

    The driver keeps a local tokenizer / chat parser for any caller that
    reads ``.tokenizer`` or ``.chat_parser`` (e.g. rllm's verl transform
    path). Model weights live only inside the actors.
    """

    def __init__(
        self,
        model: str,
        num_replicas: int = 2,
        tensor_parallel_size: int = 1,
        ray_address: str | None = None,
        max_prompt_length: int = 16384,
        max_response_length: int = 8192,
        max_model_length: int | None = None,
        sampling_params: dict | None = None,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.90,
        enforce_eager: bool = False,
        trust_remote_code: bool = True,
        disable_thinking: bool = False,
        engine_kwargs: dict | None = None,
        **_: Any,
    ) -> None:
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")

        _ensure_ray(ray_address)

        self.model = model
        self.num_replicas = num_replicas
        self.tensor_parallel_size = tensor_parallel_size
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

        # Local tokenizer / chat parser so driver-side attribute access works.
        self.tokenizer = load_tokenizer_compat(
            model, trust_remote_code=trust_remote_code
        )
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.processor = None
        self.chat_parser = ChatTemplateParser.get_parser(
            self.tokenizer, disable_thinking=disable_thinking
        )
        configure_parser_for_qwen_thinking(
            self.chat_parser, model, disable_thinking=disable_thinking
        )

        # When TP > 1, force vLLM to the in-actor multiprocess backend so we
        # don't nest Ray placement groups inside an already-Ray-hosted actor.
        merged_engine_kwargs = dict(engine_kwargs or {})
        if tensor_parallel_size > 1:
            merged_engine_kwargs.setdefault("distributed_executor_backend", "mp")

        actor_kwargs = dict(
            model=model,
            tokenizer=None,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            max_model_length=max_model_length,
            sampling_params=dict(sampling_params or {}),
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
            disable_thinking=disable_thinking,
            engine_kwargs=merged_engine_kwargs,
        )

        ActorCls = _build_actor_cls(tensor_parallel_size)
        self.actors = [ActorCls.remote(actor_kwargs) for _ in range(num_replicas)]
        # Block until every replica has finished loading weights so the
        # first wave of rollout tasks doesn't race a still-initializing actor.
        ray.get([a.ping.remote() for a in self.actors])

        self._rr = itertools.count()
        self.validate = False

    def _next_actor(self):
        idx = next(self._rr) % self.num_replicas
        return self.actors[idx]

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        actor = self._next_actor()
        ref = actor.generate.remote(messages, kwargs)
        payload: dict = await ref
        return ModelOutput.from_dict(payload)

    async def wake_up(self) -> None:
        await asyncio.gather(
            *[asyncio.ensure_future(a.wake_up.remote()) for a in self.actors]
        )

    async def sleep(self) -> None:
        await asyncio.gather(
            *[asyncio.ensure_future(a.sleep.remote()) for a in self.actors]
        )

    def shutdown(self) -> None:
        for a in self.actors:
            try:
                ray.get(a.shutdown.remote(), timeout=30)
            except Exception:
                pass
            try:
                ray.kill(a)
            except Exception:
                pass
        self.actors = []


__all__ = ["RayVLLMEngine"]
