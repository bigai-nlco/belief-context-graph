"""FastAPI app for running AgentHarness inquiries."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent_harness.api.runner import HarnessAPIRunner
from agent_harness.api.schemas import (
    AppConfigResponse,
    HealthResponse,
    InquiryRequest,
    InquiryResponse,
)
from agent_harness.infra.config import AgentHarnessConfig, get_config
from workflows.react_base.profile import load_profile


def create_app() -> FastAPI:
    """Create the AgentHarness API application."""
    runner = HarnessAPIRunner()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await runner.start()
        try:
            yield
        finally:
            await runner.stop()

    app = FastAPI(
        title="AgentHarness API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", runner_ready=runner.ready)

    @app.get("/api/config", response_model=AppConfigResponse)
    async def config() -> AppConfigResponse:
        cfg = get_config(force_reload=True)
        model = _configured_model(cfg)
        profile = load_profile("default")
        return AppConfigResponse(
            llm_provider=cfg.llm_provider,
            default_model=model,
            default_profile="default",
            supported_models=[model] if model else [],
            generation=_generation_config(profile),
        )

    @app.post("/api/inquiries", response_model=InquiryResponse)
    async def create_inquiry(request: InquiryRequest) -> InquiryResponse:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must not be empty")

        return await runner.run_inquiry(
            query=query,
            session_id=request.session_id,
            pipeline_id=request.pipeline_id,
            profile=request.profile,
            model=request.model,
            wall_time_s=request.wall_time_s,
        )

    @app.post("/api/inquiries/stream")
    async def stream_inquiry(request: InquiryRequest) -> StreamingResponse:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must not be empty")

        async def events() -> AsyncIterator[str]:
            async for event in runner.run_inquiry_stream(
                query=query,
                session_id=request.session_id,
                pipeline_id=request.pipeline_id,
                profile=request.profile,
                model=request.model,
                wall_time_s=request.wall_time_s,
            ):
                event_type = str(event.get("type") or "message")
                payload = event.get("payload") or {}
                yield (
                    f"event: {event_type}\n"
                    f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/inquiries/{session_id}/stop")
    async def stop_inquiry(session_id: str) -> dict:
        session_id = session_id.strip()
        if not session_id:
            raise HTTPException(status_code=422, detail="session_id must not be empty")
        return await runner.stop_inquiry(session_id)

    return app


def _configured_model(config: AgentHarnessConfig) -> str:
    provider = config.llm_provider.lower()
    if provider == "anthropic":
        return config.anthropic_model
    if provider == "qwen":
        return config.qwen_model
    return config.openai_model


def _generation_config(profile: dict) -> dict:
    llm = profile.get("llm") or {}
    agent = profile.get("agent") or {}
    return {
        "temperature": llm.get("temperature"),
        "top_p": llm.get("top_p"),
        "repetition_penalty": llm.get("repetition_penalty"),
        "max_context_length": agent.get("max_input_tokens"),
        "max_tokens": llm.get("max_tokens"),
    }


app = create_app()
