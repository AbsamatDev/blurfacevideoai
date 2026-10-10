"""Telegram-бот для размытия лиц на видео (python-telegram-bot, async).

Логика:
* /start, /help — справка;
* видео или видеодокумент -> скачивание -> обработка (очередь) -> отправка;
* статусы: скачивание -> очередь -> обработка (с прогрессом) -> отправка;
* лимит параллельных задач настраивается (MAX_PARALLEL_JOBS);
* временные файлы удаляются после завершения в любом случае.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path
from urllib.parse import unquote

import httpx

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from src.config import Settings
from src.bot import stats as botstats
from src.video.pipeline import ProcessResult, VideoProcessingError, process_video
from src.video.tempfiles import cleanup_stale_jobs, job_directory, result_path, safe_input_name

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".3gp"}

HELP_TEXT = (
    "🤖 *Помощь — как это работает*\n\n"
    "1️⃣ Отправьте мне видео (сообщением или файлом)\n"
    "2️⃣ Дождитесь обработки — покажу прогресс\n"
    "3️⃣ Получите готовое видео с размытыми лицами 🎉\n\n"
    "🎥 *Что поддерживается:*\n"
    "• форматы: MP4, MOV, MKV, WebM, AVI, 3GP\n"
    "• разрешение: любое, вплоть до 4K\n"
    "• длительность: до {dur:.0f} минут\n"
    "• вход до {max_in} МБ · выход до {max_out} МБ\n"
    "• аудио и частота кадров сохраняются как есть\n\n"
    "⏱ *Скорость:* видео длиной до минуты обрабатывается примерно за 1–3 минуты "
    "(4K — чуть дольше).\n\n"
    "{mode_note}\n\n"
    "⚙️ *Режимы размытия:*\n"
    "• _strong_ — сильное (по умолчанию)\n"
    "• _normal_ — умеренное\n"
    "• _pixelate_ — мозаика\n\n"
    "💡 Если видео не распознаётся как видео — перешлите его как файл 📁"
)

MODE_NOTE_CLOUD = "☁️ Лимиты облачного Bot API: скачивание до 20 МБ, отправка до 50 МБ."
MODE_NOTE_LOCAL = (
    "🚀 Подключён локальный Bot API сервер — лимиты повышены: "
    "приём файлов до {max_in} МБ, отправка до {max_out} МБ."
)

WELCOME_TEXT = (
    "👋 Привет! Я размываю лица на видео 🎥\n\n"
    "Перешлите мне видео — и получите его же, но со скрытыми лицами 😌 "
    "Удобно для влогов, съёмок на улице и кадров с прохожими.\n\n"
    "🎬 *Что поддерживается:*\n"
    "• форматы: MP4, MOV, MKV, WebM, AVI\n"
    "• разрешение: любое, вплоть до 4K\n"
    "• длительность: до {dur:.0f} минут\n"
    "• размер файла: до {max_in} МБ\n\n"
    "⏱ *Обработка:* обычно 1–3 минуты (для 4K — чуть дольше)\n"
    "🛡 Аудио, разрешение и частота кадров сохраняются\n\n"
    "👉 Просто отправьте видео в этот чат.\n"
    "Подробнее — команда /help"
)


def _looks_like_video(file_name: str | None, mime_type: str | None) -> bool:
    if mime_type and mime_type.startswith("video/"):
        return True
    if file_name:
        return Path(file_name).suffix.lower() in VIDEO_EXTS
    return False


def _progress_text(pct: int, done: int, total: int) -> str:
    """Статус с прогресс-баром визуального вида ▰▰▰▱▱."""
    filled = max(0, min(10, round(pct / 10)))
    bar = "▰" * filled + "▱" * (10 - filled)
    return (
        f"🔍 *Обработка видео…*\n\n"
        f"`{bar}` *{pct}%*\n"
        f"Кадры: {done} из {total}\n\n"
        f"Это займёт ещё пару минут ⏱"
    )


class BlurFaceBot:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sem = asyncio.Semaphore(max(1, settings.max_parallel_jobs))
        settings.ensure_temp_dir()
        cleanup_stale_jobs(settings.temp_dir, keep=settings.keep_temp)

    # ------------------------------------------------------------------
    def build(self) -> Application:
        if self.settings.telegram_api_base:
            # httpx учитывает системный прокси Windows даже для localhost:
            # запросы к локальному серверу уходят в прокси и получают 503.
            # Запрещаем прокси для loopback (побочно отключает системный
            # прокси и для остальных запросов — в локальном режиме бот всё
            # равно общается только с localhost).
            os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
            os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")
        # Таймауты PTB по умолчанию (5 с) слишком малы для аплоада видео:
        # файл успевает дойти до сервера, но клиент не дожидается ответа и
        # падает с TimedOut — видео приходит, а бот показывает «❌».
        request = HTTPXRequest(
            read_timeout=30.0,
            write_timeout=300.0,
            connect_timeout=30.0,
            pool_timeout=5.0,
            media_write_timeout=300.0,
        )
        builder = Application.builder().token(self.settings.bot_token)
        if self.settings.telegram_api_base:
            builder = (
                builder.base_url(f"{self.settings.telegram_api_base}/bot")
                .base_file_url(f"{self.settings.telegram_api_base}/file/bot")
                .local_mode(True)
            )
        app = (
            builder
            .request(request)
            .build()
        )
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_help))
        if self.settings.admin_user_id:
            app.add_handler(CommandHandler("stats", self.cmd_stats))
        app.add_handler(MessageHandler(filters.VIDEO, self.handle_video))
        app.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        app.add_error_handler(self.on_error)
        return app

    def run(self):
        if not self.settings.bot_token:
            raise SystemExit(
                "Токен бота не задан. Создайте бота через @BotFather и укажите "
                "TELEGRAM_BOT_TOKEN в файле .env (см. .env.example)."
            )
        app = self.build()
        logger.info("Бот запущен, long polling...")
        app.run_polling(allowed_updates=Update.ALL_TYPES)

    # ------------------------------------------------------------------
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self._record_user(update, "start")
        s = self.settings
        await update.message.reply_text(
            WELCOME_TEXT.format(dur=s.max_duration_sec / 60, max_in=s.max_input_size_mb),
            parse_mode="Markdown",
        )

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user or user.id != self.settings.admin_user_id:
            return
        days = 1.0
        if context.args:
            try:
                days = max(0.01, float(context.args[0]))
            except ValueError:
                pass
        text = botstats.summarize(self.settings.stats_file, days)
        text += "\n\n" + botstats.top_users(self.settings.stats_file, days)
        await update.message.reply_text(text)

    def _record_user(self, update: Update, event: str, **extra):
        """Пишет событие в статистику (ошибки записи игнорируются)."""
        user = update.effective_user
        botstats.record(
            self.settings.stats_file,
            event,
            user_id=user.id if user else 0,
            name=(user.full_name if user else "") or "",
            chat_id=update.effective_chat.id if update.effective_chat else 0,
            **extra,
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        s = self.settings
        mode_note = (
            MODE_NOTE_LOCAL.format(max_in=s.max_input_size_mb, max_out=s.max_send_size_mb)
            if s.telegram_api_base
            else MODE_NOTE_CLOUD
        )
        await update.message.reply_text(
            HELP_TEXT.format(
                max_in=s.max_input_size_mb,
                dur=s.max_duration_sec / 60,
                max_out=s.max_send_size_mb,
                mode_note=mode_note,
            ),
            parse_mode="Markdown",
        )

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        doc = update.message.document
        if not _looks_like_video(doc.file_name, doc.mime_type):
            await update.message.reply_text(
                "🤔 *Это не похоже на видеофайл*\n\n"
                "Подойдут форматы: MP4, MOV, MKV, WebM, AVI, 3GP\n"
                "Отправьте видео заново — я разберусь 😌",
                parse_mode="Markdown",
            )
            return
        await self._handle_file(update, context, doc, doc.file_name)

    async def handle_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        video = update.message.video
        await self._handle_file(update, context, video, video.file_name)

    # ------------------------------------------------------------------
    async def _handle_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE, tg_file, file_name):
        message = update.effective_message
        if not tg_file:
            return
        self._record_user(update, "received", mb=round((tg_file.file_size or 0) / 1048576, 2))
        size = tg_file.file_size
        if size and size > self.settings.max_input_size_bytes:
            await message.reply_text(
                f"❌ *Файл слишком большой*\n\n"
                f"Размер: {size / 1024 / 1024:.1f} МБ\n"
                f"Лимит: {self.settings.max_input_size_mb} МБ — это ограничение "
                f"Telegram Bot API на скачивание.\n\n"
                f"💡 Можно уменьшить видео перед отправкой или обработать его "
                f"локально: `python main.py process --input <файл> --output <файл>`",
                parse_mode="Markdown",
            )
            return

        status = await message.reply_text("⏳ *Скачиваю видео…*", parse_mode="Markdown")
        base_name = safe_input_name(file_name)
        s = self.settings
        loop = asyncio.get_running_loop()

        async with self.sem:
            try:
                with job_directory(s.temp_dir, keep=s.keep_temp) as job_dir:
                    await self._edit(status, "⏳ *Скачиваю видео…*", parse_mode="Markdown")

                    input_path = job_dir / base_name
                    try:
                        tg = await context.bot.get_file(tg_file.file_id)
                        await self._download_tg_file(tg, input_path)
                    except TelegramError as exc:
                        await self._edit(
                            status,
                            f"❌ *Не удалось скачать файл*\n\n{self._short(exc)}\n\n"
                            f"💡 Попробуйте отправить видео повторно.",
                            parse_mode="Markdown",
                        )
                        return

                    actual_size = input_path.stat().st_size
                    if actual_size > s.max_input_size_bytes:
                        await self._edit(
                            status,
                            f"❌ *Файл слишком большой*\n\n"
                            f"Размер: {actual_size / 1024 / 1024:.1f} МБ\n"
                            f"Лимит: {s.max_input_size_mb} МБ.",
                            parse_mode="Markdown",
                        )
                        return

                    await self._edit(
                        status,
                        "🔍 *Обрабатываю видео…*\n\n"
                        "Ищу лица в каждом кадре и размываю их.\n"
                        "Обычно занимает 1–3 минуты ⏱",
                        parse_mode="Markdown",
                    )
                    # Результат всегда пишем в mp4 — см. result_path()
                    out_path = result_path(job_dir, input_path)

                    progress_state = {"last": 0.0}

                    def on_progress(done: int, total: int):
                        now = time.monotonic()
                        if total and done < total and now - progress_state["last"] < s.progress_interval_sec:
                            return
                        progress_state["last"] = now
                        pct = int(done * 100 / total) if total else 0
                        asyncio.run_coroutine_threadsafe(
                            self._edit(status, _progress_text(pct, done, total)),
                            loop,
                        )

                    result = await asyncio.to_thread(
                        process_video, input_path, out_path, s, None, on_progress
                    )

                    out_size = out_path.stat().st_size
                    if out_size > s.max_send_size_bytes:
                        await self._edit(
                            status,
                            f"❌ *Результат слишком большой*\n\n"
                            f"Размер: {out_size / 1024 / 1024:.1f} МБ\n"
                            f"Лимит отправки: {s.max_send_size_mb} МБ\n\n"
                            f"💡 Обработайте видео локально:\n"
                            f"`python main.py process --input <файл> --output <файл>`",
                            parse_mode="Markdown",
                        )
                        return

                    await self._edit(status, "📤 *Отправляю результат…*", parse_mode="Markdown")
                    await context.bot.send_chat_action(message.chat_id, ChatAction.UPLOAD_VIDEO)
                    caption = self._caption(result)
                    with open(out_path, "rb") as f:
                        await message.reply_video(
                            video=f,
                            caption=caption,
                            parse_mode="Markdown",
                            supports_streaming=True,
                            filename=result.output_path.name,
                            width=result.out_width,
                            height=result.out_height,
                            duration=int(result.out_duration),
                        )
                    await self._edit(
                        status,
                        f"✅ *Готово!*\n\n"
                        f"⏱ Обработка: {result.stats.elapsed_sec:.0f} с "
                        f"({result.stats.frames_processed} кадров)\n"
                        f"⚡ Режим: {result.stats.compute_mode}\n\n"
                        f"Видео выше ☝️ Можно публиковать 🎉",
                        parse_mode="Markdown",
                    )
                    self._record_user(
                        update, "processed",
                        proc_sec=round(result.stats.elapsed_sec, 1),
                        frames=result.stats.frames_processed,
                        out_mb=round(out_size / 1048576, 2),
                    )
            except VideoProcessingError as exc:
                logger.warning("Ошибка обработки: %s", exc)
                self._record_user(update, "error", stage="process")
                await self._edit(
                    status,
                    f"❌ *Не получилось обработать видео*\n\n{exc}\n\n"
                    f"💡 Попробуйте другое видео или отправьте его файлом 📁",
                    parse_mode="Markdown",
                )
            except TelegramError as exc:
                logger.warning("Ошибка Telegram: %s", exc)
                self._record_user(update, "error", stage="telegram")
                await self._edit(
                    status,
                    f"❌ *Ошибка Telegram*\n\n{self._short(exc)}\n\n"
                    f"💡 Попробуйте отправить видео ещё раз.",
                    parse_mode="Markdown",
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Непредвиденная ошибка")
                self._record_user(update, "error", stage="internal")
                await self._edit(
                    status,
                    f"❌ *Внутренняя ошибка*\n\n{self._short(exc)}\n\n"
                    f"💡 Попробуйте позже или напишите /start заново.",
                    parse_mode="Markdown",
                )

    # ------------------------------------------------------------------
    async def _download_tg_file(self, tg_file, dest: Path):
        """Скачивает файл в dest.

        Облако: качаем средствами PTB (он сам строит URL из file_path).
        Локальный сервер (--local): getFile возвращает абсолютный путь на
        ДИСКЕ СЕРВЕРА — читаем примонтированный том напрямую. HTTP-выдача
        /file/bot<token>/... в актуальных версиях сервера отсутствует
        (HTTP-слой понимает только /bot<token>/<метод>), поэтому она лишь
        запасной вариант для старых сборок.
        """
        fp = tg_file.file_path or ""
        prefix = self.settings.telegram_server_files_prefix
        # Облако, либо URL без серверного префикса — качаем средствами PTB.
        if not self.settings.telegram_api_base or (
            fp.startswith(("http://", "https://")) and not (prefix and prefix in fp)
        ):
            await tg_file.download_to_drive(str(dest))
            return

        # Локальный сервер: file_path — путь на диске СЕРВЕРА. PTB оборачивает
        # его в URL (base_file_url + путь), т.к. по нему не существует файла
        # на машине бота — вычленяем серверный путь обратно по префиксу work-dir.
        raw = fp[fp.index(prefix):] if prefix and prefix in fp else fp
        raw = unquote(raw)
        if await asyncio.to_thread(self._copy_from_server_volume, raw, dest):
            return

        url = (
            f"{self.settings.telegram_api_base}"
            f"/file/bot{self.settings.bot_token}/{raw.lstrip('/')}"
        )
        try:
            await asyncio.to_thread(self._download_via_http, url, dest)
            return
        except httpx.HTTPError:
            dest.unlink(missing_ok=True)
        raise TelegramError(f"файл недоступен на локальном сервере: {self._short_path(fp)}")

    def _copy_from_server_volume(self, fp: str, dest: Path) -> bool:
        """Копирует файл локального сервера из тома в dest. True = успех.

        file_path ответа getFile (/var/lib/telegram-bot-api/...) отображается
        на каталог тома на стороне бота. Docker Desktop при bind-монтировании
        Windows-папки заменяет запрещённые в именах NTFS символы на символ из
        PUA-диапазона U+F000 + код (':' в имени каталога бота превращается в
        U+F03A), поэтому пробуем и такой вариант пути.
        """
        s = self.settings
        prefix = s.telegram_server_files_prefix
        local_dir = s.telegram_local_files_dir
        if not (local_dir and prefix and fp.startswith(prefix)):
            return False
        rel = fp[len(prefix):].lstrip("/\\")
        candidates = [local_dir / rel]
        mangled = rel.translate({ord(c): chr(0xF000 + ord(c)) for c in ':*?"<>|'})
        if mangled != rel:
            candidates.append(local_dir / mangled)
        for src in candidates:
            if src.is_file():
                shutil.copyfile(src, dest)
                return True
        return False

    @staticmethod
    def _download_via_http(url: str, dest: Path):
        with httpx.Client(timeout=httpx.Timeout(60.0, read=600.0, write=60.0, pool=10.0)) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_bytes(1 << 20):
                        f.write(chunk)

    @staticmethod
    def _short_path(fp: str) -> str:
        return fp[-80:] if len(fp) > 80 else fp

    # ------------------------------------------------------------------
    @staticmethod
    async def _edit(status, text: str, parse_mode=None):
        try:
            await status.edit_text(text, parse_mode=parse_mode)
        except (BadRequest, TelegramError):
            pass  # сообщение не изменилось / rate limit — не критично

    @staticmethod
    def _short(exc) -> str:
        text = str(exc).strip().splitlines()
        line = text[0] if text else "неизвестная ошибка"
        return line[:300]

    @staticmethod
    def _caption(result: ProcessResult) -> str:
        st = result.stats
        return (
            f"🎭 *Готово! Лица размыты*\n\n"
            f"👥 Найдено лиц: {st.faces_detected} "
            f"(в {st.frames_with_faces} кадрах из {st.frames_processed})\n"
            f"🎬 Разрешение: {result.in_width}x{result.in_height}\n"
            f"⏱ Обработка: {st.elapsed_sec:.0f} с\n"
            f"⚡ {st.compute_mode}\n\n"
            f"📤 Готово к публикации — можно заливать в соцсети!"
        )

    async def on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        logger.error("Ошибка при обработке обновления", exc_info=context.error)


def run_bot(settings: Settings | None = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = settings or Settings.load()
    BlurFaceBot(settings).run()