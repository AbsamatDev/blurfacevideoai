"""Точка входа: локальная обработка видео, запуск бота, проверка окружения.

Локальная обработка:
    python main.py process --input input.mp4 --output output.mp4
    (необязательно: --mode strong|normal|pixelate, --strength, --encoder auto|nvenc|x264)

Бот:
    python main.py bot

Проверка окружения:
    python main.py check
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import BLUR_MODES, ENCODER_MODES, Settings  # noqa: E402


def cmd_process(args: argparse.Namespace) -> int:
    from src.video.pipeline import VideoProcessingError, process_video

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    overrides = {}
    if args.mode:
        overrides["blur_mode"] = args.mode
    if args.strength:
        overrides["blur_strength"] = args.strength
    if args.padding:
        overrides["face_padding"] = args.padding
    if args.encoder:
        overrides["encoder"] = args.encoder

    settings = Settings.load(**overrides)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"❌ Входной файл не найден: {input_path}", file=sys.stderr)
        return 1
    if output_path.exists() and not args.force:
        print(
            f"❌ Выходной файл уже существует: {output_path} (используйте --force для перезаписи)",
            file=sys.stderr,
        )
        return 1

    print(f"Обрабатываю: {input_path}")
    try:
        result = process_video(input_path, output_path, settings)
    except VideoProcessingError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    st = result.stats
    print()
    print("=" * 60)
    print(f"Исходное разрешение:        {result.in_width}x{result.in_height}")
    print(f"Итоговое разрешение:        {result.out_width}x{result.out_height}")
    print(f"Исходная продолжительность: {result.in_duration:.2f} с")
    print(f"Итоговая продолжительность: {result.out_duration:.2f} с")
    print(f"Обработано кадров:          {st.frames_processed}")
    print(f"Обнаружено лиц (по кадрам): {st.faces_detected} (кадров с лицами: {st.frames_with_faces})")
    print(f"Режим размытия:             {settings.blur_mode} (сила {settings.blur_strength:g})")
    print(f"Режим вычислений:           {st.compute_mode} ({st.encoder})")
    print(f"Время обработки:            {st.elapsed_sec:.1f} с ({result.processing_speed_fps:.1f} кадров/с)")
    print(f"Путь к итоговому файлу:     {result.output_path.resolve()}")
    print("=" * 60)
    return 0


def cmd_bot(args: argparse.Namespace) -> int:
    from src.bot.telegram_bot import run_bot

    run_bot()
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from check_env import run_checks

    return run_checks()


def cmd_stats(args: argparse.Namespace) -> int:
    from src.bot import stats as botstats
    from src.config import Settings

    s = Settings.load()
    days = args.days
    print(botstats.summarize(s.stats_file, days))
    print()
    print(botstats.top_users(s.stats_file, days))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blurface",
        description="Размытие лиц на видео (локально или через Telegram-бота).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_process = sub.add_parser("process", help="обработать видеофайл локально")
    p_process.add_argument("--input", "-i", required=True, help="входной видеофайл")
    p_process.add_argument("--output", "-o", required=True, help="итоговый видеофайл")
    p_process.add_argument("--mode", choices=BLUR_MODES, help="режим размытия (по умолчанию из .env: strong)")
    p_process.add_argument("--strength", type=float, help="сила размытия (множитель, по умолчанию 1.0)")
    p_process.add_argument("--padding", type=float, help="запас вокруг лица (доля, по умолчанию 0.30)")
    p_process.add_argument("--encoder", choices=ENCODER_MODES, help="принудительный выбор кодировщика")
    p_process.add_argument("--force", "-f", action="store_true", help="перезаписать выходной файл")
    p_process.set_defaults(func=cmd_process)

    p_bot = sub.add_parser("bot", help="запустить Telegram-бота")
    p_bot.set_defaults(func=cmd_bot)

    p_check = sub.add_parser("check", help="проверить окружение (Python, FFmpeg, GPU, модель)")
    p_check.set_defaults(func=cmd_check)

    p_stats = sub.add_parser("stats", help="показать статистику использования бота")
    p_stats.add_argument("--days", type=float, default=1.0, help="за сколько дней (по умолчанию 1)")
    p_stats.set_defaults(func=cmd_stats)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())