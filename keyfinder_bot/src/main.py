"""Entry point for the keyfinder Telegram bot."""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher

from .config import load_config
from .bot.handlers import create_router


async def main() -> None:
    """Initialize and run the Telegram bot polling loop."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    config = load_config()
    logger.info("Starting bot with close-key hints %s", "enabled" if config.show_close_key else "disabled")

    bot = Bot(token=config.bot_token, parse_mode="HTML")
    dp = Dispatcher()
    dp.include_router(create_router(config))

    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.getLogger(__name__).info("Bot stopped")
