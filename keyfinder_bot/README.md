# Keyfinder Telegram Bot

Телеграм-бот, который принимает аудио-файлы и определяет их тональность (Key) и темп (BPM). Бот использует `librosa` для аудиоаналитики и `aiogram` v3 для взаимодействия с Telegram.

## Возможности

- Определение тональности бита и отображение энгармонической альтернативы.
- Вычисление темпа (BPM), а также половинного и удвоенного значения.
- Подсчёт длительности трека.
- Поддержка форматов: MP3, WAV, M4A, OGG, OPUS.
- Ограничение на размер файла: до 70 МБ.

## Требования

- Python 3.11+
- Установленный FFmpeg (для декодирования аудио через `audioread`).

### Установка FFmpeg

- **Ubuntu / Debian**:

  ```bash
  sudo apt-get update
  sudo apt-get install ffmpeg
  ```

- **macOS (Homebrew)**:

  ```bash
  brew install ffmpeg
  ```

- **Windows**:

  1. Скачайте сборку с [ffmpeg.org](https://ffmpeg.org/download.html) или из [репозитория gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
  2. Распакуйте архив, добавьте путь к `bin/` в переменную окружения `PATH`.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Настройка окружения

1. Создайте файл `.env` на основе `.env.example`:

   ```bash
   cp .env.example .env
   ```

2. Укажите токен Telegram-бота в файле `.env`:

   ```env
   TELEGRAM_BOT_TOKEN=ваш_токен
   ```

По желанию можно включить отображение близких тональностей, добавив переменную `SHOW_CLOSE_KEY=true`.

## Запуск

```bash
python -m src.main
```

После запуска бот начнёт поллинг обновлений. Пришлите боту аудио-файл (mp3/wav/m4a/ogg/opus) размером до 70 МБ, и он ответит сообщением вида:

```
Тональность бита: track.mp3
Тон: A#min (Bbmin)
Темп: 140 BPM (280 / 70)
Длительность: 03:25
```

Если алгоритм обнаружит близкую альтернативную тональность (при включённом флаге `SHOW_CLOSE_KEY`), будет добавлена строка `Близкий вариант: ...`.

## Примечания

- На коротких клипах и треках, состоящих преимущественно из ударных, результаты могут быть менее точными.
- Временные файлы удаляются после завершения обработки.
- Бот не использует базы данных и хранит файлы только на время анализа.

## Разработка

Структура проекта:

```
keyfinder_bot/
  README.md
  requirements.txt
  .env.example
  src/
    main.py
    config.py
    bot/
      __init__.py
      handlers.py
      messages.py
    audio_processing/
      __init__.py
      analyzer.py
      profiles.py
    utils/
      __init__.py
      files.py
```

## Лицензия

Проект распространяется по лицензии MIT. Используйте и адаптируйте под свои задачи.
