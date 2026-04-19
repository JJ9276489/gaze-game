import json
import time
from typing import Any
from uuid import uuid4


def now_ms() -> int:
    return int(time.time() * 1000)


def clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def make_client_id() -> str:
    return uuid4().hex[:12]


def encode(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def decode(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Message must be a JSON object")
    return payload


def make_join(room: str, name: str, color: tuple[int, int, int]) -> dict[str, Any]:
    return {
        "type": "join",
        "room": room,
        "name": name,
        "color": [int(color[0]), int(color[1]), int(color[2])],
    }


def make_cursor(
    room: str,
    x: float | None,
    y: float | None,
    tracking: bool,
    seq: int,
    timestamp_ms: int | None = None,
) -> dict[str, Any]:
    return {
        "type": "cursor",
        "room": room,
        "x": None if x is None else clamp01(x),
        "y": None if y is None else clamp01(y),
        "tracking": bool(tracking),
        "seq": seq,
        "ts": timestamp_ms if timestamp_ms is not None else now_ms(),
    }

