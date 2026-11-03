"""Unified YAML configuration for BCG (schema, loader, defaults)."""

from bcg.config.loader import (
    DEFAULTS_PATH,
    PROJECT_CONFIG_NAMES,
    defaults_dict,
    load_settings,
    locate_config_files,
)
from bcg.config.runtime import (
    LEGACY_CONFIG_PATH,
    RuntimeConfig,
    load_construct_config,
    resolve_runtime_config,
    settings_to_construct_config,
)
from bcg.config.schema import SCHEMA_VERSION, BCGSettings

__all__ = [
    "BCGSettings",
    "DEFAULTS_PATH",
    "LEGACY_CONFIG_PATH",
    "PROJECT_CONFIG_NAMES",
    "RuntimeConfig",
    "SCHEMA_VERSION",
    "defaults_dict",
    "load_construct_config",
    "load_settings",
    "locate_config_files",
    "resolve_runtime_config",
    "settings_to_construct_config",
]
