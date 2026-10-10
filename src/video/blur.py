"""Размытие областей лиц на кадре.

Режимы:
    normal   — умеренно сильное размытие;
    strong   — сильное размытие (по умолчанию);
    pixelate — пикселизация.

Важно: размытие — инструмент визуального сокрытия, а не криптографическая
гарантия анонимности.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from src.config import BLUR_MODES


class BlurModeError(ValueError):
    pass


def _downscale_upscale(roi: np.ndarray, factor: float) -> np.ndarray:
    """Сильное размытие: уменьшение ROI и обратное увеличение."""
    h, w = roi.shape[:2]
    nw = max(2, int(round(w / factor)))
    nh = max(2, int(round(h / factor)))
    small = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _pixelate(roi: np.ndarray, block: int) -> np.ndarray:
    h, w = roi.shape[:2]
    nw = max(2, w // block)
    nh = max(2, h // block)
    small = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def blur_roi(roi: np.ndarray, mode: str, strength: float) -> np.ndarray:
    """Размывает один ROI (копию региона) в выбранном режиме."""
    if mode not in BLUR_MODES:
        raise BlurModeError(f"Неизвестный режим размытия: {mode!r} (доступны: {', '.join(BLUR_MODES)})")
    strength = max(0.1, float(strength))
    if mode == "normal":
        out = _downscale_upscale(roi, 4.0 * strength)
        ksize = 15
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)
        return out
    if mode == "strong":
        out = _downscale_upscale(roi, 8.0 * strength)
        ksize = 21
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)
        return out
    # pixelate
    h, w = roi.shape[:2]
    block = max(4, int(round(max(w, h) / (10.0 * strength))))
    return _pixelate(roi, block)


def expand_box(box: Sequence[float], padding: float, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Расширяет box (x, y, w, h) на долю padding и возвращает x1, y1, x2, y2 (в границах кадра)."""
    x, y, w, h = (float(v) for v in box[:4])
    cx, cy = x + w / 2.0, y + h / 2.0
    ew = w * (1.0 + 2.0 * padding) + 4.0  # + несколько пикселей запаса
    eh = h * (1.0 + 2.0 * padding) + 4.0
    x1 = int(round(cx - ew / 2.0))
    y1 = int(round(cy - eh / 2.0))
    x2 = int(round(cx + ew / 2.0))
    y2 = int(round(cy + eh / 2.0))
    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(x1 + 1, min(frame_w, x2))
    y2 = max(y1 + 1, min(frame_h, y2))
    return x1, y1, x2, y2


def blur_boxes(
    frame: np.ndarray,
    boxes: Sequence[Sequence[float]],
    mode: str = "strong",
    strength: float = 1.0,
    padding: float = 0.30,
) -> np.ndarray:
    """Возвращает копию кадра с размытыми областями ``boxes`` (x, y, w, h)."""
    out = frame.copy()
    h, w = frame.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = expand_box(box, padding, w, h)
        # Пропускаем слишком маленькие области (шум детектора в 1-2 пикселя).
        if x2 - x1 < 6 or y2 - y1 < 6:
            continue
        roi = out[y1:y2, x1:x2]
        out[y1:y2, x1:x2] = blur_roi(roi, mode, strength)
    return out