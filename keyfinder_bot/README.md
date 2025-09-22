# Keyfinder Telegram Bot

Асинхронный Telegram-бот на базе `aiogram 3`, который принимает аудио-файлы и определяет основные музыкальные характеристики: тональность (с энгармонической альтернативой), темп (BPM, половинный и удвоенный) и длительность трека.

## Возможности

- Приём `audio`, `voice` и `document` сообщений с аудиоконтентом (`mp3`, `wav`, `m4a`, `ogg/opus`).
- Ограничение на размер файла — до 70 MB (проверяется до скачивания).
- DSP-пайплайн: HPSS → извлечение хрома-признаков → алгоритм Krumhansl–Schmuckler для определения тональности.
- Расчёт темпа по перкуссионной компоненте с использованием `librosa.beat.tempo`.
- Асинхронное выполнение тяжёлых операций через `asyncio.to_thread` с семафором для ограничения конкуренции.
- Логирование и дружественные сообщения об ошибках.

## Требования

- Python 3.11 или новее.
- Установленный FFmpeg (см. инструкции ниже).

## Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Установка FFmpeg

- **Ubuntu/Debian**:
  ```bash
  sudo apt-get update
  sudo apt-get install ffmpeg
  ```
- **macOS (Homebrew)**:
  ```bash
  brew install ffmpeg
  ```
- **Windows**:
  1. Скачайте готовые сборки с [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/).
  2. Распакуйте архив и добавьте путь к папке `bin` в переменную окружения `PATH`.

## Конфигурация

1. Скопируйте файл `.env.example` в `.env`:
   ```bash
   cp .env.example .env
   ```
2. Укажите ваш `TELEGRAM_BOT_TOKEN` (его можно получить у [@BotFather](https://t.me/BotFather)).
3. При необходимости можно включить отображение близкой тональности, добавив `SHOW_CLOSE_KEY=1` в `.env`.

## Запуск

```bash
python -m src.main
```

Бот начнёт long polling и будет готов принимать аудиофайлы.

## Формат ответа

```
Тональность бита: <имя_файла>
Тон: <Note><Maj|min> (<Enharmonic><Maj|min> при наличии)
Темп: <BPM> BPM (<BPM*2> / <BPM/2>)
Длительность: mm:ss
```

При активированном режиме отображения «близкой тональности» будет добавлена строка:
```
Близкий вариант: <...>
```

## Примечания

- На коротких семплах или чисто перкуссионных лупах точность может снижаться.
- Убедитесь, что отправляете сжатый файл (например, `mp3`) если исходный материал крупнее 70 MB.
- Временные файлы автоматически удаляются после анализа.

## Полезные ссылки

- [Документация aiogram v3](https://docs.aiogram.dev/)
- [Librosa documentation](https://librosa.org/doc/latest/index.html)
