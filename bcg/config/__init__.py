"""Unified YAML configuration for BCG (schema, loader, defaults)."""

from bcg.config.loader import (
    DEFAULTS_PATH,
    PROJECT_CONFIG_NAMES,
    defaults_dict,
    load_settings,
    locate_config_files,
)
from bcg.config.migration import (
    find_legacy_configs,
    legacy_settings,
    migrate_model_config,
    migrate_to_yaml,
    migrate_user_config,
)
from bcg.config.schema import SCHEMA_VERSION, BCGSettings

__all__ = [
    "BCGSettings",
    "DEFAULTS_PATH",
    "PROJECT_CONFIG_NAMES",
    "SCHEMA_VERSION",
    "defaults_dict",
    "find_legacy_configs",
    "legacy_settings",
    "load_settings",
    "locate_config_files",
    "migrate_model_config",
    "migrate_to_yaml",
    "migrate_user_config",
]
