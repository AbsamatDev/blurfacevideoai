"""Генерация синтетических тестовых видео без скачивания чужих материалов.

Примеры:
    python make_test_video.py --dir tests_data
    python make_test_video.py --kind moving_face --out_dir tests_data
    python make_test_video.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.video import synth


def build(out_dir: Path, w=640, h=360, fps=15, n_frames=45, kinds=None) -> dict:
    from src.video.ffmpeg_tools import FFmpegTool

    ff = FFmpegTool()
    out_dir.mkdir(parents=True, exist_ok=True)

    makers = {
        "no_face": lambda: synth.make_no_face_video(out_dir / "no_face.mp4", w, h, fps, n_frames, ff),
        "one_face": lambda: synth.make_face_video(out_dir / "one_face.mp4", n_faces=1, w=w, h=h, fps=fps, n_frames=n_frames, ff=ff),
        "two_faces": lambda: synth.make_face_video(out_dir / "two_faces.mp4", n_faces=2, w=w, h=h, fps=fps, n_frames=n_frames, ff=ff),
        "moving_face": lambda: synth.make_face_video(out_dir / "moving_face.mp4", moving=True, w=w, h=h, fps=fps, n_frames=n_frames, ff=ff),
        "with_audio": lambda: synth.add_sine_audio(out_dir / "one_face.mp4", out_dir / "with_audio.mp4", ff),
        "hd720": lambda: synth.make_face_video(out_dir / "hd720.mp4", n_faces=1, w=1280, h=720, fps=fps, n_frames=n_frames, ff=ff),
        "rotated90": lambda: synth.make_portrait_rotated_video(out_dir / "rotated90.mp4", n_frames=n_frames, fps=fps, ff=ff),
        "corrupted": lambda: synth.make_corrupted_video(out_dir / "corrupted.mp4", ff),
        "garbage": lambda: synth.make_garbage_file(out_dir / "garbage.mp4"),
    }

    # with_audio строится поверх one_face — добавляем зависимость при выборочном создании
    if kinds and "with_audio" in kinds:
        kinds = kinds | {"one_face"}

    made = {}
    for name, make in makers.items():
        if kinds and name not in kinds:
            continue
        made[name] = make()
    return made


def main() -> int:
    parser = argparse.ArgumentParser(description="Создать синтетические тестовые видео.")
    parser.add_argument("--dir", default="tests_data", help="каталог для результатов")
    parser.add_argument("--kind", help="создать только указанный вид (иначе все)")
    parser.add_argument("--list", action="store_true", help="показать список видео")
    args = parser.parse_args()

    if args.list:
        print("Виды видео: " + ", ".join([
            "no_face", "one_face", "two_faces", "moving_face", "with_audio",
            "hd720", "rotated90", "corrupted", "garbage"]))
        return 0

    out_dir = Path(args.dir)
    kinds = {args.kind} if args.kind else None
    made = build(out_dir, kinds=kinds)
    if not made:
        print(f"Неизвестный вид: {args.kind} (см. --list)")
        return 1
    for name, path in made.items():
        print(f"создано: {name:15s} -> {path}")
    print(f"\nВсе видео лежат в {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())