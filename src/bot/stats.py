"""Простейшая статистика использования бота.

Append-only JSONL-файл (одна JSON-строка на событие). Никакой БД —
достаточно, чтобы утром посмотреть, сколько людей и видео прошло через бота.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional


def record(stats_file: Path, event: str, **fields) -> None:
    """Добавляет событие в файл статистики. Ошибки записи игнорируются —
    статистика не должна ломать обработку видео."""
    rec = {"ts": round(time.time(), 3), "event": event, **fields}
    try:
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with stats_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _load(stats_file: Path, since_ts: float) -> list[dict]:
    if not stats_file.exists():
        return []
    out = []
    try:
        with stats_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("ts", 0) >= since_ts:
                    out.append(rec)
    except OSError:
        pass
    return out


def summarize(stats_file: Path, days: float = 1.0) -> str:
    """Текстовая сводка за последние ``days`` дней."""
    since = time.time() - days * 86400
    recs = _load(stats_file, since)
    if not recs:
        return f"За последние {days:g} дн. событий нет."

    users: dict[int, str] = {}
    videos = 0
    ok = 0
    errors: dict[str, int] = defaultdict(int)
    total_proc_sec = 0.0
    total_frames = 0
    total_in_mb = 0.0

    for r in recs:
        ev = r.get("event")
        uid = r.get("user_id") or 0
        if uid:
            users[uid] = r.get("name") or users.get(uid, "")
        if ev == "received":
            videos += 1
            total_in_mb += float(r.get("mb") or 0)
        elif ev == "processed":
            ok += 1
            total_proc_sec += float(r.get("proc_sec") or 0)
            total_frames += int(r.get("frames") or 0)
        elif ev == "error":
            errors[str(r.get("stage") or "?")] += 1

    lines = [
        f"📊 Статистика за {days:g} дн.",
        f"Пользователей: {len(users)}",
        f"Видео получено: {videos}",
        f"Успешно обработано: {ok}",
    ]
    if errors:
        lines.append("Ошибки: " + ", ".join(f"{k}={v}" for k, v in sorted(errors.items())))
    if ok:
        lines.append(
            f"Суммарно обработки: {total_proc_sec / 60:.1f} мин, "
            f"{total_frames} кадров, {total_in_mb:.1f} МБ входных файлов"
        )
    return "\n".join(lines)


def top_users(stats_file: Path, days: float = 1.0, limit: int = 10) -> str:
    """Топ пользователей по числу обработанных видео."""
    since = time.time() - days * 86400
    recs = _load(stats_file, since)
    cnt: dict[int, int] = defaultdict(int)
    names: dict[int, str] = {}
    for r in recs:
        if r.get("event") == "received":
            uid = r.get("user_id") or 0
            cnt[uid] += 1
            if uid and r.get("name"):
                names[uid] = r["name"]
    if not cnt:
        return "Нет данных."
    lines = [f"👥 Топ пользователей за {days:g} дн.:"]
    for uid, n in sorted(cnt.items(), key=lambda kv: -kv[1])[:limit]:
        label = names.get(uid) or f"id {uid}"
        lines.append(f"  {label} — {n} видео")
    return "\n".join(lines)
