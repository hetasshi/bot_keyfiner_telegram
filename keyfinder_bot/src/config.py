"""Application configuration utilities for the keyfinder bot."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

MAX_FILE_MB: Final[int] = 70
"""Default maximum file size allowed for uploads."""

SHOW_CLOSE_KEY: Final[bool] = False
"""Whether to show close key alternatives by default."""


@dataclass(frozen=True)
class Config:
    """Container for application settings."""

    bot_token: str
    max_file_mb: int = MAX_FILE_MB
    show_close_key: bool = SHOW_CLOSE_KEY


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    logger.warning("Unrecognized boolean value '%s', using default=%s", value, default)
    return default


def load_config() -> Config:
    """Load configuration from environment variables.

    Returns:
        Config: Loaded configuration object.

    Raises:
        RuntimeError: If the Telegram bot token is missing.
    """

    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Environment variable TELEGRAM_BOT_TOKEN is required but not provided."
        )

    show_close_key_env = _parse_bool(os.getenv("SHOW_CLOSE_KEY"), default=SHOW_CLOSE_KEY)

    config = Config(bot_token=token, show_close_key=show_close_key_env)
    logger.debug("Configuration loaded: show_close_key=%s", config.show_close_key)
    return config
