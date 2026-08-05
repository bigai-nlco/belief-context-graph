"""Unified YAML configuration for BCG (schema, loader, defaults)."""

from bcg.config.loader import (
    DEFAULTS_PATH,
    PROJECT_CONFIG_NAMES,
    defaults_dict,
    load_settings,
    locate_config_files,
)
from bcg.config.schema import SCHEMA_VERSION, BCGSettings

__all__ = [
    "BCGSettings",
    "DEFAULTS_PATH",
    "PROJECT_CONFIG_NAMES",
    "SCHEMA_VERSION",
    "defaults_dict",
    "load_settings",
    "locate_config_files",
]
