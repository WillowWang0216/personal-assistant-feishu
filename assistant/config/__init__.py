"""Configuration module for personal-assistant."""

from assistant.config.loader import load_config, get_config_path
from assistant.config.schema import Config

__all__ = ["Config", "load_config", "get_config_path"]
