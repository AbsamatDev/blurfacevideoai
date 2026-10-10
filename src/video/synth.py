"""Создание синтетических тестовых видео (без скачивания чужих материалов).

Все видео генерируются локально: кадры рисуются с помощью
``src.face_detection.synface`` и кодируются bundled/system FFmpeg.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT
from src.face_detection.synface import make_face_frame
from src.video.ffmpeg_tools import FFmpegTool, _popen_kwargs


def _open_encoder(ff: FFmpegTool, out_path: Path, w: int, h: int, fps: int) -> subprocess.Popen:
    cmd = [
        str(ff.ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, **_popen_kwargs())


def write_frames_video(
    out_path: Path,
    frame_fn,
    n_frames: int,
    w: int = 640,
    h: int = 360,
    fps: int = 15,
    ff: FFmpegTool | None = None,
) -> Path:
    """Кодирует видео из кадров, получаемых вызовами ``frame_fn(i)`` (BGR ndarray)."""
    ff = ff or FFmpegTool()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = _open_encoder(ff, out_path, w, h, fps)
    try:
        for i in range(n_frames):
            frame = frame_fn(i)
            assert frame.shape[:2] == (h, w), f"размер кадра {frame.shape[:2]} != {(h, w)}"
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        rc = proc.wait(timeout=120)
        if rc != 0:
            err = proc.stderr.read().decode("utf-8", "replace")
            raise RuntimeError(f"FFmpeg не смог записать тестовое видео: {err.strip()}")
    finally:
        if proc.poll() is None:
            proc.kill()
    return out_path


def add_sine_audio(src: Path, dst: Path, ff: FFmpegTool | None = None) -> Path:
    """Копирует видеопоток и добавляет аудиодорожку 440 Гц."""
    ff = ff or FFmpegTool()
    cmd = [
        str(ff.ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-f", "lavfi", "-i", "sine=frequency=440:beep_factor=2:duration=3600",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", "-shortest",
        str(dst),
    ]
    res = subprocess.run(cmd, capture_output=True, timeout=120, **_popen_kwargs())
    if res.returncode != 0:
        raise RuntimeError(f"Не удалось добавить аудио: {res.stderr.decode('utf-8', 'replace').strip()}")
    return dst


def set_rotation(src: Path, dst: Path, degrees: int, ff: FFmpegTool | None = None) -> Path:
    """Записывает метаданные поворота (display matrix) копированием потоков."""
    ff = ff or FFmpegTool()
    cmd = [
        str(ff.ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-display_rotation", str(degrees), "-i", str(src),
        "-c", "copy", str(dst),
    ]
    res = subprocess.run(cmd, capture_output=True, timeout=120, **_popen_kwargs())
    if res.returncode != 0:
        raise RuntimeError(f"Не удалось записать поворот: {res.stderr.decode('utf-8', 'replace').strip()}")
    return dst


def make_no_face_video(out_path: Path, w: int = 640, h: int = 360, fps: int = 15,
                       n_frames: int = 45, ff: FFmpegTool | None = None) -> Path:
    """Видео без лиц (шумный фон)."""
    ff = ff or FFmpegTool()
    rng = np.random.default_rng(1)

    def frame_fn(i: int):
        img = np.zeros((h, w, 3), np.uint8)
        for c in range(3):
            img[:, :, c] = int(60 + 30 * c)
        n = rng.normal(0, 10, (h, w, 1)).repeat(3, axis=2)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)

    return write_frames_video(out_path, frame_fn, n_frames, w, h, fps, ff)


def make_face_video(out_path: Path, n_faces: int = 1, moving: bool = False,
                    w: int = 640, h: int = 360, fps: int = 15, n_frames: int = 45,
                    ff: FFmpegTool | None = None) -> Path:
    """Видео с одним/несколькими (статичными или движущимися) лицами."""
    ff = ff or FFmpegTool()
    seed = 7 if moving else 42

    def frame_fn(i: int):
        t = i / max(1, n_frames - 1)
        if moving:
            from src.face_detection.synface import face_path_at_t
            cx, cy, fw = face_path_at_t(w, h, t)
            positions = [(cx, cy, fw)]
        elif n_faces == 1:
            positions = [(w // 2, h // 2, int(h * 0.62))]
        else:
            # Размер подбран так, чтобы YuNet стабильно находил оба лица
            # (230 px на холсте 640x360 попадает в «слепую зону» модели).
            fw = int(h * 0.53)
            xs = [int(w * 0.27), int(w * 0.73), w // 2]
            ys = [h // 2, h // 2, int(h * 0.8)]
            positions = [(xs[k], ys[k], fw) for k in range(n_faces)]
        return make_face_frame(w, h, seed=seed + i, positions=positions)

    return write_frames_video(out_path, frame_fn, n_frames, w, h, fps, ff)


def make_portrait_rotated_video(out_path: Path, n_frames: int = 30, fps: int = 15,
                                   ff: FFmpegTool | None = None) -> Path:
    """Видео, снятое «на телефон»: кадр 640x360 с rotation metadata 90°,
    отображается как портрет 360x640 с вертикальным лицом.

    В кодированном кадре лицо лежит на боку — так устроены реальные файлы
    с метаданными поворота. Итоговый путь: временный файл -> out_path.
    """
    import cv2

    ff = ff or FFmpegTool()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_coded = out_path.parent / (out_path.stem + "_coded.mp4")

    w_coded, h_coded = 640, 360  # кадр как в файле
    w_disp, h_disp = 360, 640    # как отображается после поворота

    def frame_fn(i: int):
        portrait = make_face_frame(w_disp, h_disp, seed=100 + i,
                                   positions=[(w_disp // 2, h_disp // 2, int(h_disp * 0.40))])
        # поворачиваем в ориентацию хранения: rot90 по часовой (k=3);
        # display_rotation=90 (против часовой) возвращает кадр в исходный вид
        coded = np.rot90(portrait, k=3)
        coded = np.ascontiguousarray(coded)
        assert coded.shape[:2] == (h_coded, w_coded), f"{coded.shape} != {(h_coded, w_coded)}"
        return coded

    write_frames_video(tmp_coded, frame_fn, n_frames, w_coded, h_coded, fps, ff)
    try:
        set_rotation(tmp_coded, out_path, 90, ff)
    finally:
        tmp_coded.unlink(missing_ok=True)
    return out_path


def make_corrupted_video(out_path: Path, ff: FFmpegTool | None = None) -> Path:
    """Повреждённый файл: обрезанный MP4 без завершающих атомов."""
    ff = ff or FFmpegTool()
    tmp = out_path.parent / (out_path.stem + "_src.mp4")
    make_no_face_video(tmp, n_frames=30, ff=ff)
    data = tmp.read_bytes()
    # Оставляем начало файла (атомы ftyp/mdat без moov) — файл открывается,
    # но не декодируется полностью.
    out_path.write_bytes(data[: len(data) // 2])
    tmp.unlink(missing_ok=True)
    return out_path


def make_garbage_file(out_path: Path) -> Path:
    """Полностью мусорный файл (случайные байты с подписью mp4)."""
    rng = np.random.default_rng(3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + rng.integers(0, 255, 64 * 1024, dtype=np.uint8).tobytes())
    return out_path