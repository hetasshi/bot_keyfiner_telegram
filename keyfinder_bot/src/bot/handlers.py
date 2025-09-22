"""Telegram message handlers for the KeyFinder bot."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from aiogram import F, Router
from aiogram.enums import ContentType
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from ..audio_processing import analyze_file
from ..config import settings
from ..utils import determine_extension, is_supported_audio, remove_silent, safe_join_temp
from . import messages

logger = logging.getLogger(__name__)

router = Router(name="keyfinder")
ANALYSIS_SEMAPHORE = asyncio.Semaphore(settings.analysis_workers)


@dataclass(slots=True)
class FileMeta:
    """Container with metadata for files downloaded from Telegram."""

    file_id: str
    file_name: str
    mime_type: Optional[str]
    file_size: Optional[int]


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    """Send a brief instruction on how to use the bot."""

    await message.answer(messages.START_MESSAGE)


@router.message(F.content_type.in_({ContentType.AUDIO, ContentType.VOICE, ContentType.DOCUMENT}))
async def handle_audio_message(message: Message) -> None:
    """Handle audio, voice and document uploads."""

    analysis_result: dict[str, Any] | None = None
    file_meta = _extract_file_meta(message)
    if file_meta is None:
        logger.debug("Received unsupported message type: %s", message.content_type)
        return

    if file_meta.file_size and file_meta.file_size > settings.max_file_bytes:
        await message.answer(messages.file_too_large(settings.max_file_mb))
        return

    if not is_supported_audio(file_meta.file_name, file_meta.mime_type):
        await message.answer(messages.unsupported_format())
        return

    extension = determine_extension(file_meta.file_name, file_meta.mime_type)
    if extension is None:
        await message.answer(messages.unsupported_format())
        return

    download_path = safe_join_temp(extension)
    try:
        tg_file = await message.bot.get_file(file_meta.file_id)
        try:
            await message.bot.download(tg_file, destination=download_path)
        except (TelegramAPIError, OSError) as exc:
            logger.warning("Download failed for %s: %s", file_meta.file_name, exc)
            await message.answer(messages.download_failed())
            return

        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            try:
                async with ANALYSIS_SEMAPHORE:
                    analysis_result = await asyncio.to_thread(
                        analyze_file,
                        download_path,
                        want_close_key=settings.show_close_key,
                    )
            except Exception:  # pragma: no cover - runtime safeguard
                logger.exception("Analysis failed for %s", download_path)
                await message.answer(messages.analysis_failed())
                return
    except TelegramAPIError as exc:  # pragma: no cover - network dependent
        logger.warning("Failed to obtain file info: %s", exc)
        await message.answer(messages.download_failed())
        return
    finally:
        remove_silent(download_path)

    if analysis_result is None:
        logger.error("Analysis result missing for %s", file_meta.file_name)
        await message.answer(messages.analysis_failed())
        return

    analysis_result["filename"] = file_meta.file_name
    if not settings.show_close_key:
        analysis_result.pop("close_key", None)

    text = messages.format_analysis_message(analysis_result)
    await message.answer(text)


def _extract_file_meta(message: Message) -> Optional[FileMeta]:
    """Extract metadata from different media types."""

    if message.audio:
        audio = message.audio
        file_name = audio.file_name or f"audio_{audio.file_unique_id}.mp3"
        return FileMeta(
            file_id=audio.file_id,
            file_name=file_name,
            mime_type=audio.mime_type,
            file_size=audio.file_size,
        )
    if message.voice:
        voice = message.voice
        file_name = f"voice_{voice.file_unique_id}.ogg"
        return FileMeta(
            file_id=voice.file_id,
            file_name=file_name,
            mime_type=voice.mime_type or "audio/ogg",
            file_size=voice.file_size,
        )
    if message.document:
        document = message.document
        if document.file_name:
            file_name = document.file_name
        else:
            suffix = determine_extension(None, document.mime_type) or ".audio"
            file_name = f"document_{document.file_unique_id}{suffix}"
        return FileMeta(
            file_id=document.file_id,
            file_name=file_name,
            mime_type=document.mime_type,
            file_size=document.file_size,
        )
    return None
