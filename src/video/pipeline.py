"""Основной конвейер обработки видео: FFmpeg (чтение) -> YuNet (детекция) ->
размытие -> FFmpeg (кодирование с сохранением аудио).

Кадры передаются потоково через пайпы, видео никогда не загружается в память
целиком. Разрешение, частота кадров, ориентация (rotation metadata) и
аудиодорожка сохраняются.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from src.config import Settings
from src.face_detection.detector import FaceDetector
from src.video.blur import blur_boxes
from src.video.ffmpeg_tools import FFmpegTool, ProbeError, VideoInfo, _popen_kwargs

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]  # (обработано кадров, оценочный всего)

MP4_LIKE = {".mp4", ".m4v", ".mov"}

# Аудио-кодеки, безопасные для копирования в MP4 без перекодирования.
MP4_SAFE_AUDIO = {"aac", "mp3", "ac3", "eac3"}

# Семейство BT.601 (SD-контент): такие файлы кодируем той же матрицей.
BT601_SPACES = {"bt470bg", "smpte170m", "smpte240m", "fcc"}

# Конвертация HDR (HLG/HDR10/широкая гамма BT.2020) в SDR BT.709 через libplacebo.
_LIBPLACEBO_SDR = (
    "libplacebo=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=limited"
)
# Запасной вариант, если в ffmpeg нет libplacebo.
_ZSCALE_SDR = (
    "zscale=t=linear:npl=100,format=gbrpf32le,"
    "zscale=p=bt709:m=bt709:t=bt709:r=limited,tonemap=hable,format=bgr24"
)


def needs_wide_gamut_convert(info: VideoInfo) -> bool:
    """HDR (HLG/PQ) или широкая гамма BT.2020 — требуют конвертации в SDR."""
    return info.is_hdr or info.color_primaries == "bt2020"


def build_decode_color_filter(ff: FFmpegTool, info: VideoInfo, settings: Settings) -> Optional[str]:
    """Фильтр для декодера, приводящий HDR/BT.2020 к SDR BT.709.

    Возвращает None для обычного SDR-контента (лишних конверсий не делаем).
    """
    if not needs_wide_gamut_convert(info):
        return None
    if not settings.convert_hdr:
        logger.warning(
            "HDR-видео (%s), конвертация отключена (CONVERT_HDR=0) — цвета могут быть блеклыми",
            info.color_transfer,
        )
        return None
    if ff.has_filter("libplacebo"):
        return _LIBPLACEBO_SDR
    logger.warning("фильтр libplacebo недоступен — используем запасную zscale-цепочку")
    return _ZSCALE_SDR


def build_color_out_args(info: VideoInfo, hdr_converted: bool) -> list:
    """Цветовые теги и матрица кодирования.

    Без явных тегов ffmpeg кодирует RGB->YUV матрицей BT.601, а плееры и
    Telegram читают HD-файл как BT.709 — цвета уезжали на каждом видео.
    """
    if hdr_converted or info.color_space not in BT601_SPACES:
        matrix, tag = "bt709", "bt709"
    else:
        matrix, tag = "bt601", "smpte170m"
    return [
        "-vf", f"scale=out_color_matrix={matrix}:out_range=tv",
        "-colorspace", tag, "-color_primaries", tag,
        "-color_trc", tag, "-color_range", "tv",
    ]


class VideoProcessingError(RuntimeError):
    """Ошибка обработки с понятным пользователю описанием."""


@dataclass
class ProcessStats:
    frames_processed: int = 0
    faces_detected: int = 0          # суммарно по кадрам (после NMS)
    raw_detections: int = 0         # суммарно по кадрам (до NMS)
    frames_with_faces: int = 0
    elapsed_sec: float = 0.0
    encoder: str = ""                # h264_nvenc | libx264
    compute_mode: str = ""           # "GPU (NVENC)" | "CPU (libx264)"


@dataclass
class ProcessResult:
    input_path: Path
    output_path: Path
    in_width: int
    in_height: int
    out_width: int
    out_height: int
    in_duration: float
    out_duration: float
    fps: float
    stats: ProcessStats = field(default_factory=ProcessStats)

    @property
    def processing_speed_fps(self) -> float:
        if self.stats.elapsed_sec > 0:
            return self.stats.frames_processed / self.stats.elapsed_sec
        return 0.0


class _StderrDrain(threading.Thread):
    """Собирает stderr дочернего процесса, чтобы пайп не переполнялся."""

    def __init__(self, pipe):
        super().__init__(daemon=True)
        self.pipe = pipe
        self.chunks: list[bytes] = []

    def run(self):
        try:
            for chunk in iter(self.pipe.readline, b""):
                self.chunks.append(chunk)
        except (ValueError, OSError):
            pass
        finally:
            try:
                self.pipe.close()
            except (ValueError, OSError):
                pass

    def text(self, limit: int = 2000) -> str:
        data = b"".join(self.chunks)
        return data.decode("utf-8", "replace").strip()[-limit:]


def _read_exact(pipe, n: int) -> Optional[bytes]:
    """Читает ровно n байт из пайпа; None — чистый EOF между кадрами.

    Частичный кадр — признак повреждённого файла.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            if len(buf) == 0:
                return None
            raise VideoProcessingError(
                "Видеофайл повреждён: поток кадров оборвался посреди кадра."
            )
        buf.extend(chunk)
    return bytes(buf)


def build_encoder_args(settings: Settings, encoder_name: str) -> list:
    """Аргументы кодирования видеопотока (без -c:v, он добавляется здесь же)."""
    args = ["-c:v", encoder_name]
    bitrate = settings.video_bitrate.strip()
    if bitrate:
        return args + ["-b:v", bitrate]
    if encoder_name == "h264_nvenc":
        args += ["-preset", "p5", "-tune", "hq", "-rc", "vbr",
                 "-cq", str(settings.nvenc_cq), "-b:v", "0", "-spatial-aq", "1"]
    else:  # libx264
        args += ["-preset", "medium", "-crf", str(settings.x264_crf)]
    return args


def choose_encoder(settings: Settings, ff: FFmpegTool) -> str:
    """Выбор кодировщика: ENCODER=auto|nvenc|x264."""
    if settings.encoder == "nvenc":
        if not ff.nvenc_available():
            raise VideoProcessingError(
                "GPU-кодирование NVENC недоступно (нет видеокарты NVIDIA или драйвер слишком старый). "
                "Обновите драйвер NVIDIA или установите ENCODER=x264 в .env."
            )
        return "h264_nvenc"
    if settings.encoder == "x264":
        return "libx264"
    return "h264_nvenc" if ff.nvenc_available() else "libx264"


def validate_output(
    ff: FFmpegTool,
    output_path: Path,
    in_info: VideoInfo,
    settings: Settings,
) -> VideoInfo:
    """Проверяет итоговый файл; при расхождении бросает VideoProcessingError."""
    out_info = ff.probe(output_path)
    if (out_info.display_width, out_info.display_height) != (in_info.display_width, in_info.display_height):
        raise VideoProcessingError(
            f"Разрешение изменилось: {out_info.display_width}x{out_info.display_height} "
            f"вместо {in_info.display_width}x{in_info.display_height}."
        )
    tol = max(1.0, 0.05 * in_info.duration)
    if abs(out_info.duration - in_info.duration) > tol:
        raise VideoProcessingError(
            f"Длительность изменилась: {out_info.duration:.2f} с вместо {in_info.duration:.2f} с."
        )
    if in_info.has_audio and not out_info.has_audio:
        raise VideoProcessingError("В исходном видео была аудиодорожка, а в результате её нет.")
    if in_info.has_audio and out_info.audio_duration and in_info.audio_duration:
        if abs(out_info.audio_duration - in_info.audio_duration) > 1.5:
            raise VideoProcessingError("Длительность аудиодорожки заметно отличается от исходной.")
    if settings.verify_max_duration_sec and in_info.duration <= settings.verify_max_duration_sec:
        ok, err = ff.verify_decodable(output_path)
        if not ok:
            raise VideoProcessingError(f"Итоговый файл не декодируется полностью: {err}")
    return out_info


# ----------------------------------------------------------------------
def process_video(
    input_path,
    output_path,
    settings: Settings,
    detector: Optional[FaceDetector] = None,
    progress: Optional[ProgressCallback] = None,
) -> ProcessResult:
    """Обрабатывает видео: находит лица, размывает, кодирует результат.

    При ошибке частичный итоговый файл удаляется.
    """
    t_start = time.perf_counter()
    input_path = Path(input_path)
    output_path = Path(output_path)
    ff = FFmpegTool(settings.ffmpeg_path, settings.ffprobe_path)

    # --- 1. Анализ входного файла ---
    try:
        info: VideoInfo = ff.probe(input_path)
    except ProbeError as exc:
        raise VideoProcessingError(str(exc)) from exc

    if info.duration <= 0:
        raise VideoProcessingError(
            "Не удалось определить длительность видео — файл может быть повреждён."
        )
    if info.fps <= 0:
        raise VideoProcessingError(
            "Не удалось определить частоту кадров видео — файл может быть повреждён."
        )
    if settings.max_duration_sec and info.duration > settings.max_duration_sec:
        raise VideoProcessingError(
            f"Видео слишком длинное: {info.duration:.0f} с "
            f"(лимит: {settings.max_duration_sec:.0f} с, настраивается в MAX_DURATION_SEC)."
        )

    encoder_name = choose_encoder(settings, ff)
    stats = ProcessStats(
        encoder=encoder_name,
        compute_mode="GPU (NVENC)" if encoder_name == "h264_nvenc" else "CPU (libx264)",
    )

    detector = detector or FaceDetector(
        model_path=settings.model_path,
        score_threshold=settings.detect_score_threshold,
        nms_threshold=settings.detect_nms_threshold,
        max_side=settings.detect_max_side,
        scales=settings.detect_scales,
        nms_iou=settings.detect_nms_iou,
    )

    fps_arg = info.fps_fraction if ("/" in info.fps_fraction and info.fps > 0) else f"{info.fps:.6f}"
    total_estimate = info.nb_frames or max(1, int(round(info.duration * info.fps)))

    logger.info(
        "Обработка %s: %dx%d, %s fps, %.2f с, кодек %s, аудио: %s, энкодер %s",
        input_path.name, info.display_width, info.display_height, fps_arg, info.duration,
        info.video_codec, "да" if info.has_audio else "нет", encoder_name,
    )

    # --- 2. Команды FFmpeg ---
    color_filter = build_decode_color_filter(ff, info, settings)
    dec_cmd = [
        str(ff.ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-map", "0:v:0",
        "-vsync", "0",  # все кадры без дублирования/пропуска
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    if color_filter:
        # HDR/BT.2020 -> SDR BT.709 до передачи кадров в OpenCV.
        # -vf — выходная опция: ставим перед "-f rawvideo".
        dec_cmd[dec_cmd.index("-f"):dec_cmd.index("-f")] = ["-vf", color_filter]
        logger.info("Конвертация цвета на декодировании: %s", color_filter)

    color_out_args = build_color_out_args(info, hdr_converted=bool(color_filter))

    def make_enc_cmd(w: int, h: int, audio_args: Sequence[str]) -> list:
        cmd = [
            str(ff.ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", fps_arg,
            "-i", "pipe:0",
            "-i", str(input_path),
            "-map", "0:v:0", "-map", "1:a?",
            *build_encoder_args(settings, encoder_name),
            "-pix_fmt", "yuv420p",
            *color_out_args,
            *audio_args,
        ]
        if output_path.suffix.lower() in MP4_LIKE:
            cmd += ["-movflags", "+faststart"]
        cmd.append(str(output_path))
        return cmd

    # Аудио: без дорожки — ничего не добавляем; в MP4 копируем только безопасные
    # кодеки (иначе Telegram/плееры могут не воспроизвести), остальные перекодируем
    # в AAC; в прочие контейнеры (mkv и т.п.) — сначала копирование, при отказе — AAC.
    if not info.has_audio:
        audio_variants = [["-an"]]
    elif output_path.suffix.lower() in MP4_LIKE:
        if set(info.audio_codecs) <= MP4_SAFE_AUDIO:
            audio_variants = [["-c:a", "copy"], ["-c:a", "aac", "-b:a", "192k"]]
        else:
            audio_variants = [["-c:a", "aac", "-b:a", "192k"]]
    else:
        audio_variants = [["-c:a", "copy"], ["-c:a", "aac", "-b:a", "192k"]]
    logger.info("варианты аудио-схемы: %s", audio_variants)

    last_encoder_error = ""
    try:
        for attempt, audio_args in enumerate(audio_variants):
            ok, last_encoder_error = _run_stream(
                ff=ff,
                dec_cmd=dec_cmd,
                make_enc_cmd=make_enc_cmd,
                audio_args=audio_args,
                detector=detector,
                settings=settings,
                stats=stats,
                total_estimate=total_estimate,
                progress=progress,
                input_path=input_path,
                output_path=output_path,
                info=info,
                fresh_stats=True,
            )
            if ok:
                break
            logger.warning(
                "Кодирование не удалось (аудио: %s): %s",
                " ".join(audio_args), last_encoder_error,
            )
            output_path.unlink(missing_ok=True)
            if attempt + 1 < len(audio_variants):
                logger.info("Пробуем другую схему аудио-дорожки...")
    except VideoProcessingError:
        output_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise VideoProcessingError(f"Непредвиденная ошибка обработки: {exc}") from exc

    if not ok:
        raise VideoProcessingError(
            "Не удалось закодировать итоговое видео: "
            + (last_encoder_error or "неизвестная ошибка")
        )

    stats.elapsed_sec = time.perf_counter() - t_start

    # --- 3. Проверка результата ---
    try:
        out_info = validate_output(ff, output_path, info, settings)
    except VideoProcessingError:
        output_path.unlink(missing_ok=True)
        raise

    logger.info(
        "Готово: %d кадров, %d лиц (в %d кадрах), %.1f с (%.1f кадров/с), файл: %s",
        stats.frames_processed, stats.faces_detected, stats.frames_with_faces,
        stats.elapsed_sec, stats.frames_processed / max(1e-9, stats.elapsed_sec),
        output_path.name,
    )

    return ProcessResult(
        input_path=input_path,
        output_path=output_path,
        in_width=info.display_width,
        in_height=info.display_height,
        out_width=out_info.display_width,
        out_height=out_info.display_height,
        in_duration=info.duration,
        out_duration=out_info.duration,
        fps=info.fps,
        stats=stats,
    )


# ----------------------------------------------------------------------
def _run_stream(
    ff: FFmpegTool,
    dec_cmd: list,
    make_enc_cmd,
    audio_args: Sequence[str],
    detector: FaceDetector,
    settings: Settings,
    stats: ProcessStats,
    total_estimate: int,
    progress: Optional[ProgressCallback],
    input_path: Path,
    output_path: Path,
    info: VideoInfo,
    fresh_stats: bool,
) -> tuple[bool, str]:
    """Один проход потоковой обработки.

    Возвращает ``(True, "")`` при успехе. Возвращает ``(False, описание)``,
    если упало именно кодирование (можно повторить с другой схемой аудио);
    ошибки декодирования/входного файла бросаются как исключения.
    """
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_popen_kwargs())
    dec_err = _StderrDrain(dec.stderr)
    dec_err.start()

    enc = None
    enc_err = None
    try:
        w, h = info.display_width, info.display_height
        first = _read_exact(dec.stdout, w * h * 3)
        if first is None:
            raise VideoProcessingError(
                "Не удалось прочитать ни одного кадра — видео повреждено или не декодируется."
            )

        enc = subprocess.Popen(
            make_enc_cmd(w, h, audio_args),
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, **_popen_kwargs(),
        )
        enc_err = _StderrDrain(enc.stderr)
        enc_err.start()

        if fresh_stats:
            stats.faces_detected = 0
            stats.raw_detections = 0
            stats.frames_processed = 0
            stats.frames_with_faces = 0
        last_report = 0

        frame = first
        while frame is not None:
            view = np.frombuffer(frame, dtype=np.uint8).reshape(h, w, 3)
            boxes, raw = detector.detect(view)
            if boxes:
                blurred = blur_boxes(view, boxes, settings.blur_mode,
                                     settings.blur_strength, settings.face_padding)
                payload: bytes = blurred.tobytes()
                stats.faces_detected += len(boxes)
                stats.frames_with_faces += 1
            else:
                payload = frame  # размытие не требуется — кадр не копируем
            stats.raw_detections += raw
            stats.frames_processed += 1

            try:
                enc.stdin.write(payload)
            except (BrokenPipeError, OSError):
                return False, _enc_error(enc, enc_err)

            if progress is not None and stats.frames_processed - last_report >= 25:
                progress(stats.frames_processed, total_estimate)
                last_report = stats.frames_processed

            frame = _read_exact(dec.stdout, w * h * 3)

        if progress is not None:
            progress(stats.frames_processed, stats.frames_processed)

        dec.stdout.close()
        if dec.wait(timeout=120) != 0:
            raise VideoProcessingError(
                "Ошибка декодирования входного видео: " + (dec_err.text() or "неизвестная ошибка")
            )

        enc.stdin.close()
        if enc.wait(timeout=600) != 0:
            return False, _enc_error(enc, enc_err)

        if not output_path.exists() or output_path.stat().st_size == 0:
            return False, "кодировщик не создал итоговый файл"
        return True, ""

    except (BrokenPipeError, OSError) as exc:
        raise VideoProcessingError(f"Сбой обмена с FFmpeg: {exc}") from exc
    finally:
        for proc in (dec, enc):
            if proc is not None and proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass


def _enc_error(enc, enc_err) -> str:
    detail = enc_err.text() if enc_err else ""
    return detail or f"ffmpeg завершился с кодом {enc.returncode}"