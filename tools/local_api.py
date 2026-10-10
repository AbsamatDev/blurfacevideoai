"""Управление локальным Bot API сервером.

    python tools/local_api.py status   # проверить облако и локальный сервер
    python tools/local_api.py logout   # ОДИН РАЗ выйти из облака перед переключением

Порядок первого переключения на локальный сервер:
  1. docker compose up -d
  2. python tools/local_api.py logout
  3. в .env: TELEGRAM_API_BASE=http://localhost:8081 (и поднять лимиты)
  4. перезапустить бота
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from src.config import Settings  # noqa: E402

CLOUD = "https://api.telegram.org"


def _call(base: str, token: str, method: str, timeout: float = 15.0) -> dict:
    r = httpx.post(f"{base}/bot{token}/{method}", timeout=timeout)
    data = r.json()
    return data


def _origin(settings: Settings) -> str:
    return settings.telegram_api_base.rstrip("/") if settings.telegram_api_base else ""


def cmd_status(settings: Settings) -> int:
    ok = True

    r = _call(CLOUD, settings.bot_token, "getMe")
    print("облако:", "OK, бот @" + r["result"]["username"] if r.get("ok") else f"ОШИБКА: {r}")

    origin = _origin(settings)
    if not origin:
        print("локальный сервер: TELEGRAM_API_BASE не задан в .env — режим облака")
        return 0 if ok else 1

    try:
        r = _call(origin, settings.bot_token, "getMe", timeout=5.0)
        print("локальный сервер:", "OK, бот @" + r["result"]["username"] if r.get("ok") else f"ОШИБКА: {r}")
        if not r.get("ok"):
            ok = False
            print("  (если 'Unauthorized' — выполните один раз: python tools/local_api.py logout)")
    except httpx.HTTPError as exc:
        ok = False
        print(f"локальный сервер: НЕДОСТУПЕН ({exc.__class__.__name__}: {exc})")
        print("  запустите: docker compose up -d")
    return 0 if ok else 1


def cmd_logout(settings: Settings) -> int:
    print("Внимание: бот выйдет из облачного сервера (обратный переход займёт ~10 минут).")
    answer = input("Продолжить? [y/N]: ").strip().lower()
    if answer not in ("y", "yes", "да"):
        print("Отменено.")
        return 1
    r = _call(CLOUD, settings.bot_token, "logOut")
    if r.get("ok"):
        print("OK: бот отключён от облака. Теперь запустите локальный сервер")
        print("  docker compose up -d")
        print("и перезапустите бота (он подхватит TELEGRAM_API_BASE из .env).")
        return 0
    print(f"Ошибка: {json.dumps(r, ensure_ascii=False)}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "logout"])
    args = parser.parse_args()
    settings = Settings.load()
    if not settings.bot_token:
        print("TELEGRAM_BOT_TOKEN не задан (.env).", file=sys.stderr)
        return 1
    return cmd_status(settings) if args.command == "status" else cmd_logout(settings)


if __name__ == "__main__":
    raise SystemExit(main())
