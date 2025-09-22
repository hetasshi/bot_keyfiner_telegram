"""Entrypoint for the Keyfinder Telegram bot."""
from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from .config import Settings, load_settings
from .bot.handlers import setup_handlers
from .utils.files import ensure_temp_dir


logger = logging.getLogger(__name__)


async def _run_bot(settings: Settings) -> None:
    """Start polling loop with the provided settings."""
    ensure_temp_dir()

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    setup_handlers(dp, settings)

    logger.info("Bot started. Waiting for updates...")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


async def main() -> None:
    """Configure logging, load settings and run the bot."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = load_settings()
    await _run_bot(settings)


if __name__ == "__main__":
    asyncio.run(main())
