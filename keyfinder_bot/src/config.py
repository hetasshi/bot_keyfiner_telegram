"""Application configuration loading utilities."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Final

from dotenv import load_dotenv

MAX_FILE_MB: Final[int] = 70
DEFAULT_SHOW_CLOSE_KEY: Final[bool] = False
DEFAULT_ANALYSIS_WORKERS: Final[int] = 2


@dataclass(frozen=True)
class Settings:
    """Container for application settings."""

    bot_token: str
    max_file_mb: int = MAX_FILE_MB
    show_close_key: bool = DEFAULT_SHOW_CLOSE_KEY
    analysis_workers: int = DEFAULT_ANALYSIS_WORKERS

    @property
    def max_file_bytes(self) -> int:
        """Return the maximum permitted file size in bytes."""

        return self.max_file_mb * 1024 * 1024


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    """Load application settings from environment variables."""

    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set. Please configure .env before running the bot.")

    show_close_key = _parse_bool(os.getenv("SHOW_CLOSE_KEY"), default=DEFAULT_SHOW_CLOSE_KEY)
    try:
        workers = int(os.getenv("ANALYSIS_WORKERS", str(DEFAULT_ANALYSIS_WORKERS)))
    except ValueError:
        logging.warning("Invalid ANALYSIS_WORKERS value, falling back to %s", DEFAULT_ANALYSIS_WORKERS)
        workers = DEFAULT_ANALYSIS_WORKERS
    if workers < 1:
        logging.warning("ANALYSIS_WORKERS must be >= 1; using %s", DEFAULT_ANALYSIS_WORKERS)
        workers = DEFAULT_ANALYSIS_WORKERS

    return Settings(
        bot_token=token,
        max_file_mb=MAX_FILE_MB,
        show_close_key=show_close_key,
        analysis_workers=workers,
    )


settings = load_settings()
