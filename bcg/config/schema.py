"""Pydantic schema for the unified BCG YAML configuration.

The schema validates YAML-parsed dictionaries; defaults live exclusively in
``bcg/config/defaults.yaml``, so the schema itself must not re-declare
behavioral defaults (Pydantic fields here carry no default except where a
value is structurally required).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1


class _ForbidExtra(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerSettings(_ForbidExtra):
    host: str
    port: int = Field(ge=1, le=65535)


class PricingSettings(_ForbidExtra):
    input_per_1k: float
    output_per_1k: float


class ModelEntry(_ForbidExtra):
    """One model routing entry (chat model or embedding backend).

    ``api_key_env`` names an environment variable that holds the secret;
    inline secrets must never be written to configuration files.
    """

    api_key_env: str | None = None
    base_url: str | None = None
    model: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = None
    temperature: float | None = None
    top_p: float | None = None
    pricing: PricingSettings | None = None
    provider: str | None = None
    device: str | None = None
    dtype: str | None = None
    batch_size: int | None = Field(default=None, ge=1)
    max_length: int | None = Field(default=None, ge=1)
    trust_remote_code: bool | None = None
    input_prefix: str | None = None
    model_kwargs: dict[str, Any] | None = None


class ExtractorSettings(_ForbidExtra):
    enabled: bool | None = None
    provider: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_concurrency: int | None = Field(default=None, ge=1)
    request_timeout: float | None = Field(default=None, ge=0)
    retries: int | None = Field(default=None, ge=0)
    context_scope: str | None = None
    enable_thinking: bool | None = None
    include_turn_content: bool | None = None
    require_excerpt: bool | None = None
    dynamic_node_cap: bool | None = None
    node_cap_unit: str | None = None
    node_cap_ratio: float | None = None
    node_cap_min: int | None = None
    node_cap_max: int | None = None


class StanceLabel(_ForbidExtra):
    description: str


class StanceSettings(_ForbidExtra):
    enabled: bool | None = None
    model_path: str | None = None
    device: str | None = None
    dtype: str | None = None
    batch_size: int | None = Field(default=None, ge=1)
    max_length: int | None = Field(default=None, ge=1)
    local_files_only: bool | None = None
    hypothesis_template: str | None = None
    labels: dict[str, StanceLabel] | None = None


class EdgeGenerationSettings(_ForbidExtra):
    enabled: bool | None = None
    provider: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    retries: int | None = Field(default=None, ge=0)
    enable_thinking: bool | None = None
    fail_on_error: bool | None = None
    search_previous_turns: bool | None = None
    max_previous_windows: int | None = Field(default=None, ge=0)


class RuntimeSettings(_ForbidExtra):
    evidence_mode: Literal["sentence", "excerpt", "chunk"]
    context_chars: int = Field(ge=0)
    min_content_len: int = Field(ge=0)


class IncrementalMergeSettings(_ForbidExtra):
    enabled: bool
    threshold: float = Field(ge=0.0, le=1.0)
    keep_newest_text: bool


class EntitySettings(_ForbidExtra):
    method: str | None = None
    spacy_model: str | None = None
    huggingface_model: str | None = None
    device: str | None = None
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    merge_overlapping: bool | None = None
    include_standard_types: bool | None = None
    fallback_methods: list[str] | None = None
    patterns: list[Any] | None = None


class RelationPropagationSettings(_ForbidExtra):
    default_relation_weight: float | None = None
    input_confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    min_confidence_delta: float | None = Field(default=None, ge=0.0)
    max_iterations: int | None = Field(default=None, ge=1)


class ConfidenceSettings(_ForbidExtra):
    initial_method: str | None = None
    evidence_method: str | None = None
    source_weight: float | None = None
    stance_weight: float | None = None
    default_source_reliability: float | None = Field(default=None, ge=0.0, le=1.0)
    default_stance_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    source_reliability: dict[str, float] | None = None
    stance_quality: dict[str, float] | None = None
    relation_propagation: RelationPropagationSettings | None = None


class ChunkingSettings(_ForbidExtra):
    enabled: bool | None = None
    breakpoint_percentile_threshold: float | None = Field(
        default=None, ge=0.0, le=100.0
    )
    buffer_size: int | None = Field(default=None, ge=0)
    min_chunk_sentences: int | None = Field(default=None, ge=1)
    isolate_tool_calls: bool | None = None


class PipelineSettings(_ForbidExtra):
    extractor: ExtractorSettings | None = None
    stance: StanceSettings | None = None
    edge_generation: EdgeGenerationSettings | None = None
    runtime: RuntimeSettings | None = None
    incremental_merge: IncrementalMergeSettings | None = None
    entities: EntitySettings | None = None
    confidence: ConfidenceSettings | None = None
    chunking: ChunkingSettings | None = None


class RunnerSettings(_ForbidExtra):
    """SDK-side run defaults (BCGRunner/RunOptions)."""

    evidence_mode: Literal["sentence", "excerpt", "chunk"]
    incremental_merge: bool
    incremental_merge_threshold: float = Field(ge=0.0, le=1.0)
    verify_merge: bool
    context_chars: int = Field(ge=0)
    io_context_chars: int = Field(ge=0)
    min_content_len: int = Field(ge=0)


class BCGSettings(_ForbidExtra):
    """Top-level unified configuration."""

    schema_version: int
    backend: Literal["unified", "hybrid"]
    model_key: str = Field(min_length=1)
    embedding_key: str = Field(min_length=1)
    server: ServerSettings
    models: dict[str, ModelEntry]
    pipeline: PipelineSettings
    runner: RunnerSettings

    @model_validator(mode="after")
    def _check_version(self) -> BCGSettings:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version}; "
                f"this version of BCG expects {SCHEMA_VERSION}"
            )
        return self


__all__ = [
    "BCGSettings",
    "SCHEMA_VERSION",
]
