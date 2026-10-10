"""Обёртка над детектором лиц YuNet (OpenCV FaceDetectorYN).

Особенности:
* детекция на уменьшенном масштабе (длинная сторона кадра ограничена
  ``max_side``) — YuNet обучен на лица ~10..300 px в масштабе входа,
  поэтому подача полного 1080p-кадра и медленнее, и пропускает крупные
  планы;
* дополнительный проход в масштабе 0.5 ловит очень крупные лица
  (крупный план, селфи вплотную);
* объединение результатов разных масштабов через NMS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

BoundingBox = tuple[float, float, float, float]  # (x, y, w, h)


class DetectorError(RuntimeError):
    pass


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def nms(boxes_with_scores: list, iou_threshold: float) -> list:
    """Простой NMS по IoU. ``boxes_with_scores`` — элементы [x, y, w, h, score]."""
    items = sorted(boxes_with_scores, key=lambda b: b[4], reverse=True)
    kept: list = []
    for box in items:
        if all(_iou(box, k) <= iou_threshold for k in kept):
            kept.append(box)
    return kept


class FaceDetector:
    """Потокобезопасность: один экземпляр — одна задача (используется из одного потока)."""

    def __init__(
        self,
        model_path: Path,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        max_side: int = 960,
        scales: Sequence[float] = (1.0, 0.5),
        nms_iou: float = 0.35,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise DetectorError(
                f"Модель детектора не найдена: {model_path}. "
                "Скачайте face_yunet_2023mar.onnx (см. README, раздел «Установка»)"
            )
        self._model_path = model_path
        self._score_threshold = float(score_threshold)
        self._nms_threshold = float(nms_threshold)
        self._max_side = int(max_side)
        self._scales = tuple(float(s) for s in scales if 0 < float(s) <= 1.0) or (1.0,)
        self._nms_iou = float(nms_iou)
        try:
            self._det = cv2.FaceDetectorYN.create(
                str(model_path), "", (320, 320),
                score_threshold=self._score_threshold,
                nms_threshold=self._nms_threshold,
                top_k=int(top_k),
            )
        except cv2.error as exc:
            raise DetectorError(
                "Не удалось загрузить модель YuNet (файл повреждён или версия OpenCV несовместима): "
                + str(exc).splitlines()[0]
            ) from exc
        self._last_input_size: Optional[tuple] = None

    @property
    def model_path(self) -> Path:
        return self._model_path

    def _detect_at(self, image: np.ndarray) -> list:
        h, w = image.shape[:2]
        size = (w, h)
        if size != self._last_input_size:
            self._det.setInputSize(size)
            self._last_input_size = size
        _, faces = self._det.detect(image)
        if faces is None:
            return []
        return [list(map(float, row[:4])) + [float(row[14])] for row in faces]

    def detect(self, frame: np.ndarray) -> tuple[list, int]:
        """Находит лица на кадре BGR.

        Возвращает ``(boxes, raw_count)``:
            boxes     — список (x, y, w, h) в координатах исходного кадра, после NMS;
            raw_count — суммарное число срабатываний до NMS (для статистики).
        """
        h, w = frame.shape[:2]
        long_side = max(h, w)
        base = min(1.0, self._max_side / long_side) if long_side > 0 else 1.0

        candidates: list = []
        for scale in self._scales:
            f = base * scale
            tw = int(round(w * f))
            th = int(round(h * f))
            if tw < 32 or th < 32:
                continue
            if (tw, th) == (w, h):
                scaled = frame
            else:
                interp = cv2.INTER_AREA if f < 1.0 else cv2.INTER_LINEAR
                scaled = cv2.resize(frame, (tw, th), interpolation=interp)
            for x, y, bw, bh, score in self._detect_at(scaled):
                candidates.append([x / f, y / f, bw / f, bh / f, score])

        raw_count = len(candidates)
        kept = nms(candidates, self._nms_iou)
        return [tuple(b[:4]) for b in kept], raw_count