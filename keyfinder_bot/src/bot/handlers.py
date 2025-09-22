"""Telegram message handlers."""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction
from aiogram.filters import CommandStart
from aiogram.types import Audio, Document, Message, Voice

from ..audio_processing import analyze_file
from ..config import Settings
from ..utils import (
    detect_extension,
    ensure_temp_dir,
    is_supported_audio,
    remove_silent,
    safe_join_temp,
)
from . import messages

logger = logging.getLogger(__name__)


def register_handlers(router: Router, settings: Settings) -> None:
    """Register bot message handlers."""

    ensure_temp_dir(settings.temp_dir)
    analysis_semaphore = asyncio.Semaphore(settings.analysis_concurrency)
    max_file_size = settings.max_file_bytes

    @router.message(CommandStart())
    async def cmd_start(message: Message) -> None:
        logger.info("Received /start from %s", message.from_user.id if message.from_user else "user")
        await message.answer(messages.START_MESSAGE)

    @router.message(F.audio)
    async def handle_audio(message: Message, bot: Bot) -> None:
        if not message.audio:
            return
        await _process_file(
            bot=bot,
            message=message,
            file=message.audio,
            fallback_prefix="audio",
            max_size=max_file_size,
            semaphore=analysis_semaphore,
            show_close_key=settings.show_close_key,
        )

    @router.message(F.voice)
    async def handle_voice(message: Message, bot: Bot) -> None:
        if not message.voice:
            return
        await _process_file(
            bot=bot,
            message=message,
            file=message.voice,
            fallback_prefix="voice",
            max_size=max_file_size,
            semaphore=analysis_semaphore,
            show_close_key=settings.show_close_key,
        )

    @router.message(F.document)
    async def handle_document(message: Message, bot: Bot) -> None:
        document = message.document
        if document is None:
            return
        if not is_supported_audio(document.file_name, document.mime_type):
            logger.info("Unsupported document received: %s", document.file_name)
            await message.answer(messages.UNSUPPORTED_FORMAT_MESSAGE)
            return
        await _process_file(
            bot=bot,
            message=message,
            file=document,
            fallback_prefix="document",
            max_size=max_file_size,
            semaphore=analysis_semaphore,
            show_close_key=settings.show_close_key,
        )


async def _process_file(
    *,
    bot: Bot,
    message: Message,
    file: Audio | Voice | Document,
    fallback_prefix: str,
    max_size: int,
    semaphore: asyncio.Semaphore,
    show_close_key: bool,
) -> None:
    """Download, analyze and respond for a Telegram file object."""

    file_size = getattr(file, "file_size", None)
    if file_size and file_size > max_size:
        logger.info("File rejected due to size: %s bytes", file_size)
        await message.answer(messages.FILE_TOO_LARGE_MESSAGE)
        return

    filename = getattr(file, "file_name", None)
    mime_type = getattr(file, "mime_type", None)
    extension = detect_extension(filename, mime_type) or ".ogg"

    if not is_supported_audio(filename, mime_type):
        logger.info("Unsupported format: name=%s mime=%s", filename, mime_type)
        await message.answer(messages.UNSUPPORTED_FORMAT_MESSAGE)
        return

    if not filename:
        filename = f"{fallback_prefix}_{getattr(file, 'file_unique_id', 'audio')}{extension}"

    temp_path = safe_join_temp(extension)
    try:
        await message.answer_chat_action(ChatAction.TYPING)
    except Exception:  # pragma: no cover - not critical
        pass

    try:
        file_info = await bot.get_file(file.file_id)
        await bot.download(file_info, destination=temp_path)
    except Exception as exc:
        logger.exception("Failed to download file: %s", exc)
        await message.answer(messages.ANALYSIS_ERROR_MESSAGE)
        remove_silent(temp_path)
        return

    try:
        async with semaphore:
            result = await asyncio.to_thread(
                analyze_file,
                temp_path,
                want_close_key=show_close_key,
            )
        result["filename"] = filename
        reply_text = messages.format_analysis_result(result)
        await message.answer(reply_text)
    except Exception as exc:  # pragma: no cover - runtime errors
        logger.exception("Audio analysis failed: %s", exc)
        await message.answer(messages.ANALYSIS_ERROR_MESSAGE)
    finally:
        remove_silent(temp_path)


__all__ = ["register_handlers"]
