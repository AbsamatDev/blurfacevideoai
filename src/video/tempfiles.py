"""Безопасная работа с временными файлами заданий."""
from __future__ import annotations

import logging
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

SAFE_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".3gp"}


def safe_input_name(original_name: str | None) -> str:
    """Безопасное имя входного файла: только проверенное расширение + случайный суффикс."""
    ext = ""
    if original_name:
        candidate = Path(original_name).suffix.lower()
        if candidate in SAFE_EXTS:
            ext = candidate
    return f"input_{uuid.uuid4().hex[:12]}{ext}"


def result_path(job_dir: Path, input_path: Path) -> Path:
    """Путь итогового файла задания: всегда mp4 (H.264+AAC — самый
    совместимый контейнер), с уникальным именем, не зависящим от имени
    входного файла (Telegram может не передать имя вовсе — регрессия:
    путь без расширения ломал muxer ffmpeg).
    """
    input_path = Path(input_path)
    token = input_path.stem.split("_", 1)[-1] if "_" in input_path.stem else ""
    token = token or uuid.uuid4().hex[:8]
    return job_dir / f"result_{token}.mp4"


@contextmanager
def job_directory(temp_dir: Path, keep: bool = False):
    """Контекст временного каталога задания.

    Создаёт уникальный подкаталог в ``temp_dir`` и гарантированно удаляет
    его после завершения (в том числе при исключении), если ``keep=False``.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    job_dir = Path(tempfile.mkdtemp(prefix="job_", dir=str(temp_dir)))
    logger.debug("создан временный каталог задания: %s", job_dir)
    try:
        yield job_dir
    finally:
        if keep:
            logger.info("файлы задания сохранены (KEEP_TEMP=1): %s", job_dir)
        else:
            shutil.rmtree(job_dir, ignore_errors=True)
            logger.debug("временный каталог задания удалён: %s", job_dir)


def cleanup_stale_jobs(temp_dir: Path, keep: bool = False) -> int:
    """Удаляет осиротевшие подкаталоги заданий в temp_dir (например, после падения).

    Вызывать только когда нет активных заданий (например, на старте бота).
    """
    if not temp_dir.exists():
        return 0
    removed = 0
    for child in temp_dir.iterdir():
        if child.is_dir() and child.name.startswith("job_"):
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    if removed and not keep:
        logger.info("удалено осиротевших временных каталогов: %d", removed)
    return removed