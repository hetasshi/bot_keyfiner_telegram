"""Telegram handlers for processing user messages."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import ContentType, Message
from aiogram.utils.chat_action import ChatActionSender
from aiogram import Dispatcher

from ..audio_processing import analyzer
from ..audio_processing.analyzer import AnalysisError
from ..config import Settings
from ..utils import files
from . import messages

logger = logging.getLogger(__name__)

router = Router()

_config: Optional[Settings] = None
_semaphore: Optional[asyncio.Semaphore] = None


@dataclass(slots=True)
class MediaMeta:
    file_id: str
    extension: str
    display_name: str
    size: Optional[int]
    mime_type: Optional[str]


def setup_handlers(dp: Dispatcher, settings: Settings) -> None:
    """Register handlers and prepare runtime context."""
    global _config, _semaphore
    _config = settings
    _semaphore = asyncio.Semaphore(settings.analysis_concurrency)
    dp.include_router(router)
    logger.info(
        "Handlers registered (limit=%s concurrent analyses)",
        settings.analysis_concurrency,
    )


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(messages.START_MESSAGE)


@router.message(F.content_type.in_({ContentType.AUDIO, ContentType.VOICE, ContentType.DOCUMENT}))
async def handle_audio(message: Message) -> None:
    if _config is None or _semaphore is None:
        raise RuntimeError("Handlers are not configured. Call setup_handlers first.")

    media = _extract_media_meta(message)
    if media is None:
        await message.answer(messages.unsupported_file())
        return

    limit_bytes = _config.max_file_mb * 1024 * 1024
    if media.size and media.size > limit_bytes:
        await message.answer(messages.file_too_large(_config.max_file_mb))
        return

    bot = message.bot
    temp_path = files.safe_join_temp(media.extension)

    try:
        file_info = await bot.get_file(media.file_id)
        file_size = file_info.file_size
        if file_size and file_size > limit_bytes:
            await message.answer(messages.file_too_large(_config.max_file_mb))
            return

        await bot.download_file(file_info.file_path, destination=temp_path)

        async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
            async with _semaphore:
                analysis = await asyncio.to_thread(
                    analyzer.analyze_file,
                    temp_path,
                    want_close_key=_config.show_close_key,
                )
        analysis["filename"] = media.display_name

        await message.answer(messages.format_analysis_result(analysis))

    except AnalysisError as exc:
        logger.warning("Failed to analyse %s: %s", media.display_name, exc)
        await message.answer(messages.processing_error())
    except Exception as exc:  # pragma: no cover - unexpected failure
        logger.exception("Unexpected error while processing %s", media.display_name)
        await message.answer(messages.processing_error())
    finally:
        files.remove_silent(temp_path)


def _extract_media_meta(message: Message) -> Optional[MediaMeta]:
    if message.audio:
        audio = message.audio
        ext = files.resolve_extension(audio.file_name, audio.mime_type)
        if not ext:
            return None
        display = files.pick_filename(audio.file_name, ext)
        return MediaMeta(
            file_id=audio.file_id,
            extension=ext,
            display_name=display,
            size=audio.file_size,
            mime_type=audio.mime_type,
        )

    if message.voice:
        voice = message.voice
        ext = files.resolve_extension(None, voice.mime_type) or ".ogg"
        display = files.pick_filename("voice_message.ogg", ext)
        return MediaMeta(
            file_id=voice.file_id,
            extension=ext,
            display_name=display,
            size=voice.file_size,
            mime_type=voice.mime_type,
        )

    if message.document:
        document = message.document
        ext = files.resolve_extension(document.file_name, document.mime_type)
        if not ext:
            return None
        display = files.pick_filename(document.file_name, ext)
        return MediaMeta(
            file_id=document.file_id,
            extension=ext,
            display_name=display,
            size=document.file_size,
            mime_type=document.mime_type,
        )

    return None


__all__ = ["setup_handlers", "router"]
