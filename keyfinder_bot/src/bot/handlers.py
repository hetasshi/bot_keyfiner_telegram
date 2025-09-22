"""Telegram handlers for the keyfinder bot."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Tuple

from aiogram import F, Router
from aiogram.enums import ChatAction, ContentType
from aiogram.filters import CommandStart
from aiogram.types import Message

from ..audio_processing.analyzer import analyze_file
from ..config import Config
from ..utils.files import display_name, get_extension, is_supported_audio, remove_silent, safe_join_temp
from . import messages

logger = logging.getLogger(__name__)

ANALYSIS_CONCURRENCY = 2
_analysis_semaphore = asyncio.Semaphore(ANALYSIS_CONCURRENCY)


def create_router(config: Config) -> Router:
    """Create and configure the router with all bot handlers."""

    router = Router()

    @router.message(CommandStart())
    async def handle_start(message: Message) -> None:
        await message.answer(messages.start_message(config.max_file_mb))

    @router.message(
        (F.content_type == ContentType.AUDIO)
        | (F.content_type == ContentType.VOICE)
        | (F.content_type == ContentType.DOCUMENT)
    )
    async def handle_audio(message: Message) -> None:
        meta = _extract_file_meta(message)
        if meta is None:
            await message.answer(messages.unsupported_format())
            return

        file_id, original_name, mime_type, file_size = meta

        extension = get_extension(original_name, mime_type)
        if extension is None:
            await message.answer(messages.unsupported_format())
            return

        max_bytes = config.max_file_mb * 1024 * 1024
        if file_size and file_size > max_bytes:
            await message.answer(messages.file_too_large(config.max_file_mb))
            return

        temp_path = safe_join_temp(extension)

        try:
            file = await message.bot.get_file(file_id)
            if file.file_size and file.file_size > max_bytes:
                await message.answer(messages.file_too_large(config.max_file_mb))
                return

            await message.bot.download(file, destination=temp_path)
            await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

            async with _analysis_semaphore:
                analysis = await asyncio.to_thread(
                    analyze_file,
                    temp_path,
                    want_close_key=config.show_close_key,
                )

            human_name = display_name(original_name, extension, fallback="audio")
            analysis["filename"] = human_name

            reply = messages.format_analysis_result(
                analysis, include_close=config.show_close_key
            )
            await message.answer(reply)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.exception("Failed to process audio: %s", exc)
            await message.answer(messages.processing_error())
        finally:
            remove_silent(temp_path)

    @router.message()
    async def handle_other(message: Message) -> None:
        await message.answer(messages.start_message(config.max_file_mb))

    return router


def _extract_file_meta(message: Message) -> Optional[Tuple[str, Optional[str], Optional[str], Optional[int]]]:
    """Extract TG file metadata (file_id, filename, mime_type, size)."""

    if message.audio:
        audio = message.audio
        return audio.file_id, audio.file_name, audio.mime_type, audio.file_size

    if message.voice:
        voice = message.voice
        fallback_name = f"voice_{message.message_id}.ogg"
        return voice.file_id, fallback_name, voice.mime_type, voice.file_size

    if message.document:
        document = message.document
        if not is_supported_audio(document.file_name, document.mime_type):
            return None
        return document.file_id, document.file_name, document.mime_type, document.file_size

    return None
