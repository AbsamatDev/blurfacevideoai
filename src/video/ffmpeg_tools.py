"""Поиск FFmpeg, запуск, анализ (ffprobe) и проверки видео."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.config import PROJECT_ROOT

# Не открываем консольные окна при запуске ffmpeg из Python на Windows.
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class FFmpegNotFoundError(RuntimeError):
    """FFmpeg или ffprobe не найдены в системе."""


class ProbeError(RuntimeError):
    """Не удалось прочитать информацию о видеофайле."""


def _popen_kwargs() -> dict:
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = CREATE_NO_WINDOW
    return kw


def find_binary(name: str, explicit: Optional[Path] = None) -> Path:
    """Ищет bundled-бинарник, потом системный PATH. Бросает FFmpegNotFoundError."""
    if explicit:
        if explicit.exists():
            return explicit
        raise FFmpegNotFoundError(f"Указанный путь не существует: {explicit}")

    # 1. Бинарники, идущие вместе с проектом.
    bundled = PROJECT_ROOT / "tools" / "bin" / (name + (".exe" if os.name == "nt" else ""))
    if bundled.exists():
        return bundled

    # 2. Системный PATH.
    found = shutil.which(name)
    if found:
        return Path(found)

    raise FFmpegNotFoundError(
        f"FFmpeg не найден ({name}). Установите FFmpeg и добавьте его в PATH "
        f"либо укажите путь в переменной окружения "
        f"{'FFMPEG_PATH' if name == 'ffmpeg' else 'FFPROBE_PATH'} в файле .env."
    )


class FFmpegTool:
    """Обёртка над ffmpeg/ffprobe для нужд проекта."""

    def __init__(self, ffmpeg_path: Optional[Path] = None, ffprobe_path: Optional[Path] = None):
        self.ffmpeg = find_binary("ffmpeg", ffmpeg_path)
        self.ffprobe = find_binary("ffprobe", ffprobe_path)

    # ------------------------------------------------------------------
    def run(self, cmd: list, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, timeout=timeout, **_popen_kwargs())

    def version(self) -> str:
        res = self.run([str(self.ffmpeg), "-version"])
        if res.returncode != 0:
            raise FFmpegNotFoundError("FFmpeg не удалось запустить")
        return res.stdout.decode("utf-8", "replace").splitlines()[0]

    # ------------------------------------------------------------------
    @property
    @lru_cache(maxsize=None)
    def _nvenc_cache(self) -> bool:
        return self._test_nvenc()

    def _test_nvenc(self) -> bool:
        """Реальная пробная кодировка: гарантирует, что NVENC доступен (драйвер и т.п.)."""
        try:
            with tempfile.TemporaryDirectory(prefix="blurface_nvenc_") as td:
                out = Path(td) / "test.mp4"
                cmd = [
                    str(self.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10:duration=0.2",
                    "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", str(out),
                ]
                res = self.run(cmd, timeout=30)
                return res.returncode == 0 and out.exists() and out.stat().st_size > 0
        except Exception:
            return False

    def nvenc_available(self) -> bool:
        return self._nvenc_cache

    # ------------------------------------------------------------------
    @property
    @lru_cache(maxsize=None)
    def _filters_cache(self) -> frozenset:
        try:
            res = self.run([str(self.ffmpeg), "-hide_banner", "-filters"], timeout=30)
        except Exception:
            return frozenset()
        names = set()
        for line in res.stdout.decode("utf-8", "replace").splitlines():
            parts = line.split()
            # Строка вида " .SC zscale  V->V  ..." — первое поле из 3 символов-флагов.
            if len(parts) >= 3 and len(parts[0]) == 3 and parts[0].strip(".CAS ") == "":
                names.add(parts[1])
        return frozenset(names)

    def has_filter(self, name: str) -> bool:
        """Есть ли фильтр в собранном ffmpeg (например, libplacebo для HDR)."""
        return name in self._filters_cache

    # ------------------------------------------------------------------
    def probe(self, path) -> "VideoInfo":
        path = Path(path)
        if not path.exists():
            raise ProbeError(f"Файл не найден: {path}")
        cmd = [
            str(self.ffprobe), "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ]
        res = self.run(cmd, timeout=60)
        if res.returncode != 0:
            tail = res.stderr.decode("utf-8", "replace").strip().splitlines()
            detail = tail[-1] if tail else "неизвестная ошибка"
            raise ProbeError(f"Не удалось прочитать видеофайл: {detail}")
        try:
            data = json.loads(res.stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ProbeError(f"Не удалось разобрать ответ ffprobe: {exc}") from exc

        streams = data.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        if video is None:
            raise ProbeError("В файле нет видеодорожки (это не видео или файл повреждён)")

        rotation = 0
        for sd in video.get("side_data_list", []) or []:
            if "rotation" in sd:
                rotation = int(round(float(sd["rotation"])))
        # Старый вариант хранения: тег rotate
        tag = (video.get("tags") or {}).get("rotate")
        if tag is not None and rotation == 0:
            try:
                rotation = int(round(float(tag)))
            except ValueError:
                rotation = 0

        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
        if width <= 0 or height <= 0:
            raise ProbeError("Не удалось определить размеры видео — файл может быть повреждён")

        swap = rotation in (90, -90, 270, -270)

        def rate_to_float(value: str) -> float:
            try:
                num, _, den = value.partition("/")
                num_f = float(num)
                den_f = float(den) if den else 1.0
                if den_f == 0 or num_f == 0:
                    return 0.0
                return num_f / den_f
            except (ValueError, TypeError):
                return 0.0

        fps = rate_to_float(video.get("avg_frame_rate") or "")
        fps_fraction = video.get("avg_frame_rate") or ""
        if fps <= 0:
            fps = rate_to_float(video.get("r_frame_rate") or "")
            fps_fraction = video.get("r_frame_rate") or ""

        nb_frames = video.get("nb_frames")
        if nb_frames is not None:
            try:
                nb_frames = int(nb_frames)
            except ValueError:
                nb_frames = None

        fmt = data.get("format", {})
        try:
            duration = float(fmt.get("duration") or 0.0)
        except ValueError:
            duration = 0.0

        audios = [s for s in streams if s.get("codec_type") == "audio"]
        audio_duration = None
        if audios:
            try:
                audio_duration = float(audios[0].get("duration") or 0.0) or None
            except ValueError:
                audio_duration = None

        return VideoInfo(
            width=width,
            height=height,
            rotation=rotation,
            display_width=height if swap else width,
            display_height=width if swap else height,
            fps=fps,
            fps_fraction=fps_fraction,
            duration=duration,
            nb_frames=nb_frames,
            video_codec=video.get("codec_name") or "?",
            pix_fmt=video.get("pix_fmt") or "?",
            has_audio=bool(audios),
            audio_codecs=[s.get("codec_name") or "?" for s in audios],
            audio_duration=audio_duration,
            color_range=video.get("color_range") or "unknown",
            color_space=video.get("color_space") or "unknown",
            color_transfer=video.get("color_transfer") or "unknown",
            color_primaries=video.get("color_primaries") or "unknown",
        )

    # ------------------------------------------------------------------
    def verify_decodable(self, path, max_duration: Optional[float] = None) -> tuple[bool, str]:
        """Полная проверка декодируемости. Возвращает (успех, последняя ошибка)."""
        cmd = [str(self.ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-v", "error"]
        if max_duration and max_duration > 0:
            cmd += ["-t", str(max_duration)]
        cmd += ["-i", str(path), "-map", "0:v:0", "-f", "null", "-"]
        try:
            res = self.run(cmd, timeout=600)
        except subprocess.TimeoutExpired:
            return False, "проверка не завершилась вовремя"
        err = res.stderr.decode("utf-8", "replace").strip()
        last = err.splitlines()[-1] if err else ""
        return res.returncode == 0, last

    def extract_frame(self, path, timestamp: float, out_png: Path) -> bool:
        """Извлекает кадр на заданной секунде в PNG (с учётом авторотации)."""
        cmd = [
            str(self.ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{timestamp:.3f}", "-i", str(path),
            "-frames:v", "1", "-update", "1", str(out_png),
        ]
        res = self.run(cmd, timeout=60)
        return res.returncode == 0 and out_png.exists()


@dataclass
class VideoInfo:
    width: int
    height: int
    rotation: int
    display_width: int
    display_height: int
    fps: float
    fps_fraction: str
    duration: float
    nb_frames: Optional[int]
    video_codec: str
    pix_fmt: str
    has_audio: bool
    audio_codecs: list = field(default_factory=list)
    audio_duration: Optional[float] = None
    # Цветовые метаданные (для корректной конверсии HDR и цветовых тегов).
    color_range: str = "unknown"
    color_space: str = "unknown"
    color_transfer: str = "unknown"
    color_primaries: str = "unknown"

    @property
    def is_hdr(self) -> bool:
        """HDR-видео: transfer HLG (arib-std-b67) или HDR10/PQ (smpte2084)."""
        return self.color_transfer in ("arib-std-b67", "smpte2084")