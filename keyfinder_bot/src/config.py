"""Application configuration management."""
from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv

MAX_FILE_MB = 70
SHOW_CLOSE_KEY = False
DEFAULT_ANALYSIS_CONCURRENCY = 2


@dataclass(slots=True)
class Settings:
    """Runtime settings for the bot."""

    bot_token: str
    max_file_mb: int = MAX_FILE_MB
    show_close_key: bool = SHOW_CLOSE_KEY
    analysis_concurrency: int = DEFAULT_ANALYSIS_CONCURRENCY


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer value for {name!r}: {raw!r}") from exc
    return max(1, parsed)


def load_settings() -> Settings:
    """Load settings from environment variables."""
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Environment variable TELEGRAM_BOT_TOKEN is required. "
            "Populate it in the .env file."
        )

    show_close = _env_bool("SHOW_CLOSE_KEY", SHOW_CLOSE_KEY)
    concurrency = _env_int("ANALYSIS_CONCURRENCY", DEFAULT_ANALYSIS_CONCURRENCY)

    return Settings(
        bot_token=token,
        max_file_mb=MAX_FILE_MB,
        show_close_key=show_close,
        analysis_concurrency=concurrency,
    )


__all__ = ["Settings", "load_settings", "MAX_FILE_MB", "SHOW_CLOSE_KEY"]
