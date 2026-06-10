"""Pydantic models for the AgentHarness web API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class InquiryRequest(BaseModel):
    """Request body for a single frontend inquiry."""

    query: str = Field(..., min_length=1, max_length=20_000)
    session_id: str | None = Field(default=None, max_length=128)
    pipeline_id: str = Field(default="react_base", max_length=128)
    profile: str | None = Field(default="default", max_length=128)
    model: str | None = Field(default=None, max_length=128)
    wall_time_s: int | None = Field(default=None, ge=1, le=86_400)


class TraceStep(BaseModel):
    """Reader-safe step emitted from the harness trace."""

    index: int
    turn: int | None = None
    title: str
    summary: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | str | None = None
    observation: str = ""
    duration_ms: int | None = None
    status: Literal["completed", "error"] = "completed"


class InquiryResponse(BaseModel):
    """Response body for a completed inquiry."""

    id: str
    session_id: str | None = None
    query: str
    status: Literal["completed", "failed"]
    final_answer: str = ""
    trace: list[TraceStep] = Field(default_factory=list)
    duration_seconds: float
    pipeline_id: str = "react_base"
    profile: str | None = None
    model: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check payload."""

    status: Literal["ok"]
    runner_ready: bool


class AppConfigResponse(BaseModel):
    """Public frontend configuration derived from environment settings."""

    llm_provider: str
    default_model: str
    default_profile: str
    supported_models: list[str]
    generation: dict[str, Any]
