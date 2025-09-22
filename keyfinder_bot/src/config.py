"""Application configuration loading."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file if present.
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MAX_FILE_MB = 70
DEFAULT_SHOW_CLOSE_KEY = False
DEFAULT_ANALYSIS_CONCURRENCY = 2
DEFAULT_TEMP_DIR = Path.cwd() / "temp"


def _bool_from_env(value: Optional[str], default: bool) -> bool:
    """Parse boolean flag from environment variables."""

    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.warning("Unknown boolean env value '%s', using default=%s", value, default)
    return default


def _int_from_env(value: Optional[str], default: int, *, minimum: int | None = None) -> int:
    """Parse integer configuration value from environment."""

    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Invalid integer env value '%s', using default=%s", value, default)
        return default
    if minimum is not None and parsed < minimum:
        logger.warning(
            "Configuration value %s below minimum %s, using minimum", parsed, minimum
        )
        return minimum
    return parsed


@dataclass(slots=True)
class Settings:
    """Dataclass storing runtime configuration."""

    telegram_bot_token: str
    max_file_mb: int = DEFAULT_MAX_FILE_MB
    show_close_key: bool = DEFAULT_SHOW_CLOSE_KEY
    analysis_concurrency: int = DEFAULT_ANALYSIS_CONCURRENCY
    temp_dir: Path = field(default_factory=lambda: DEFAULT_TEMP_DIR)

    @property
    def max_file_bytes(self) -> int:
        """Return maximum allowed file size in bytes."""

        return self.max_file_mb * 1024 * 1024


def load_settings() -> Settings:
    """Load and validate application settings from environment variables."""

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Please create a .env file with the bot token."
        )

    show_close_key = _bool_from_env(os.getenv("SHOW_CLOSE_KEY"), DEFAULT_SHOW_CLOSE_KEY)
    max_file_mb = _int_from_env(os.getenv("MAX_FILE_MB"), DEFAULT_MAX_FILE_MB, minimum=1)
    concurrency = _int_from_env(
        os.getenv("ANALYSIS_CONCURRENCY"), DEFAULT_ANALYSIS_CONCURRENCY, minimum=1
    )

    temp_dir_env = os.getenv("KEYFINDER_TEMP")
    temp_dir = Path(temp_dir_env) if temp_dir_env else DEFAULT_TEMP_DIR

    settings = Settings(
        telegram_bot_token=token,
        max_file_mb=max_file_mb,
        show_close_key=show_close_key,
        analysis_concurrency=concurrency,
        temp_dir=temp_dir,
    )
    return settings


__all__ = ["Settings", "load_settings"]
