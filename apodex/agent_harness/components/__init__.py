"""Components — concrete implementations of pluggable runtime extensions.

- ``middleware/`` — LLM-call middleware (e.g. SummarizationMiddleware).
- ``observers/`` — loop observers (budget guard, trajectory file, SSE, ...).

Layering: ``core/`` ← ``components/`` ← ``workflows/``. Nothing here may
import from ``workflows/``.
"""
