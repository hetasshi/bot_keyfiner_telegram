"""Entry point for the keyfinder Telegram bot."""

from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher

from .bot import register_handlers
from .config import load_settings


async def main() -> None:
    """Configure and run the Telegram bot."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        settings = load_settings()
    except RuntimeError as error:
        logging.error("Configuration error: %s", error)
        raise

    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()
    register_handlers(dispatcher, settings)

    logging.info("Bot started. Listening for updates...")
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
        sys.exit(0)
