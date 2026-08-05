"""Step 8: unified exception hierarchy tests."""

from __future__ import annotations

import pytest

from bcg.core.errors import (
    BCGArtifactError,
    BCGBackendError,
    BCGConfigError,
    BCGError,
    BCGUsageError,
)


def test_hierarchy_subclasses_standard_exceptions() -> None:
    assert issubclass(BCGError, Exception)
    assert issubclass(BCGConfigError, ValueError)
    assert issubclass(BCGUsageError, RuntimeError)
    assert issubclass(BCGArtifactError, RuntimeError)
    assert issubclass(BCGBackendError, RuntimeError)


def test_loader_uses_bcg_config_error(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    from bcg.config import load_settings

    with pytest.raises(BCGConfigError):
        load_settings(explicit=str(bad), home=tmp_path / "no-home")

    # legacy handlers catching ValueError still work
    with pytest.raises(ValueError):
        load_settings(explicit=str(bad), home=tmp_path / "no-home")
