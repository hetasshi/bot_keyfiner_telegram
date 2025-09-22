"""Entry point for the KeyFinder Telegram bot."""
from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode

from .config import settings
from .bot.handlers import router
from .utils.files import ensure_temp_dir


async def main() -> None:
    """Configure and run the Telegram bot."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("aiogram").setLevel(logging.INFO)

    temp_dir = ensure_temp_dir()
    logging.info("Using temporary directory: %s", temp_dir)

    bot = Bot(token=settings.bot_token, parse_mode=ParseMode.HTML)
    dp = Dispatcher()
    dp.include_router(router)

    logging.info("Bot started. Waiting for messages...")
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.getLogger(__name__).info("Bot stopped.")
