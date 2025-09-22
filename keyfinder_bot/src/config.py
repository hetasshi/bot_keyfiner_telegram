"""Application configuration management."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MAX_FILE_MB = 70
DEFAULT_SHOW_CLOSE_KEY = False
DEFAULT_ANALYSIS_CONCURRENCY = 2
DEFAULT_TEMP_DIR = Path(".temp")


@dataclass(slots=True)
class Settings:
    """Runtime settings for the bot."""

    bot_token: str
    max_file_mb: int = DEFAULT_MAX_FILE_MB
    show_close_key: bool = DEFAULT_SHOW_CLOSE_KEY
    analysis_concurrency: int = DEFAULT_ANALYSIS_CONCURRENCY
    temp_dir: Path = DEFAULT_TEMP_DIR


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer value for {name!r}: {raw!r}") from exc
    return max(minimum, parsed)


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    base = default if raw is None or not raw.strip() else Path(raw.strip())
    return base.expanduser()


def load_settings() -> Settings:
    """Load settings from environment variables."""
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Environment variable TELEGRAM_BOT_TOKEN is required. "
            "Populate it in the .env file."
        )

    max_file_mb = _env_int("MAX_FILE_MB", DEFAULT_MAX_FILE_MB)
    show_close = _env_bool("SHOW_CLOSE_KEY", DEFAULT_SHOW_CLOSE_KEY)
    concurrency = _env_int(
        "ANALYSIS_CONCURRENCY", DEFAULT_ANALYSIS_CONCURRENCY, minimum=1
    )
    temp_dir = _env_path("KEYFINDER_TEMP", DEFAULT_TEMP_DIR)

    return Settings(
        bot_token=token,
        max_file_mb=max_file_mb,
        show_close_key=show_close,
        analysis_concurrency=concurrency,
        temp_dir=temp_dir,
    )


__all__ = [
    "Settings",
    "load_settings",
    "DEFAULT_MAX_FILE_MB",
    "DEFAULT_SHOW_CLOSE_KEY",
    "DEFAULT_ANALYSIS_CONCURRENCY",
    "DEFAULT_TEMP_DIR",
]
