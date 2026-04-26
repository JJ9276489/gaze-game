import argparse
import asyncio
from dataclasses import dataclass
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Any

from websockets.datastructures import Headers
from websockets.http11 import Response

try:
    from websockets.asyncio.server import serve
except ImportError:  # pragma: no cover - compatibility for older websockets
    from websockets import serve

from shared_gaze.protocol import clamp01, decode, encode, make_client_id, now_ms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEB_DIR = PROJECT_ROOT / "web"
MAX_WAVE_TARGETS = 120
DEFAULT_WAVE_DURATION_MS = 30_000
MAX_WAVE_DURATION_MS = 120_000
MAX_CLIENTS_PER_ROOM = 8
MAX_MESSAGE_BYTES = 16 * 1024


@dataclass
class RelayClient:
    id: str
    websocket: Any
    room: str | None = None
    name: str = "anonymous"
    color: tuple[int, int, int] = (0, 255, 255)


class RelayState:
    def __init__(self) -> None:
        self.clients: dict[str, RelayClient] = {}
        self.rooms: dict[str, dict[str, RelayClient]] = {}
        self.waves: dict[str, dict[str, Any]] = {}

    def join_room(self, client: RelayClient, room: str) -> None:
        self.leave_room(client)
        client.room = room
        self.rooms.setdefault(room, {})[client.id] = client

    def can_join_room(self, client: RelayClient, room: str) -> bool:
        room_clients = self.rooms.get(room, {})
        return client.id in room_clients or len(room_clients) < MAX_CLIENTS_PER_ROOM

    def leave_room(self, client: RelayClient) -> None:
        if client.room is None:
            return
        room_clients = self.rooms.get(client.room)
        if room_clients is not None:
            room_clients.pop(client.id, None)
            if not room_clients:
                self.rooms.pop(client.room, None)
                self.waves.pop(client.room, None)
        client.room = None

    def active_wave_for(self, room: str) -> dict[str, Any] | None:
        wave = self.waves.get(room)
        if not wave:
            return None
        if now_ms() >= int(wave.get("ends_at") or 0):
            self.waves.pop(room, None)
            return None
        return public_wave(wave)

    def start_wave(self, room: str, client: RelayClient, message: dict[str, Any]) -> dict[str, Any]:
        active = self.active_wave_for(room)
        if active is not None:
            return active

        started_at = now_ms()
        duration_ms = clean_duration_ms(message.get("duration_ms"))
        wave = {
            "id": make_client_id(),
            "mode": "multiplayer",
            "seed": clean_wave_seed(message.get("seed")),
            "targets": clean_wave_targets(message.get("targets")),
            "started_at": started_at,
            "duration_ms": duration_ms,
            "ends_at": started_at + duration_ms,
            "started_by": client.id,
            "started_by_name": client.name,
            "scores": {},
        }
        self.waves[room] = wave
        return public_wave(wave)

    def update_wave_score(
        self,
        room: str,
        client: RelayClient,
        wave_id: Any,
        score: Any,
        target_id: Any,
    ) -> dict[str, Any] | None:
        wave = self.waves.get(room)
        if not wave or wave.get("id") != str(wave_id or ""):
            return None
        if now_ms() >= int(wave.get("ends_at") or 0):
            self.waves.pop(room, None)
            return None
        try:
            clean_score = int(score or 0)
        except Exception:
            clean_score = 0
        clean_score = max(0, min(9999, clean_score))
        scores = wave.setdefault("scores", {})
        scores[client.id] = {
            "id": client.id,
            "name": client.name,
            "color": list(client.color),
            "score": clean_score,
        }
        return {
            "type": "wave_score",
            "wave_id": wave["id"],
            "id": client.id,
            "name": client.name,
            "color": list(client.color),
            "score": clean_score,
            "target_id": str(target_id or "")[:64],
            "server_ts": now_ms(),
        }

    def peers_for(self, client: RelayClient) -> list[dict[str, Any]]:
        if client.room is None:
            return []
        return [
            {
                "id": peer.id,
                "name": peer.name,
                "color": list(peer.color),
            }
            for peer in self.rooms.get(client.room, {}).values()
            if peer.id != client.id
        ]

    async def broadcast(
        self,
        room: str,
        payload: dict[str, Any],
        exclude_id: str | None = None,
    ) -> None:
        peers = list(self.rooms.get(room, {}).values())
        if not peers:
            return
        message = encode(payload)
        dead: list[RelayClient] = []
        for peer in peers:
            if peer.id == exclude_id:
                continue
            try:
                await peer.websocket.send(message)
            except Exception:
                dead.append(peer)
        for peer in dead:
            self.leave_room(peer)
            self.clients.pop(peer.id, None)


def parse_color(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return (0, 255, 255)
    try:
        return tuple(max(0, min(255, int(channel))) for channel in value)  # type: ignore[return-value]
    except Exception:
        return (0, 255, 255)


def clean_room(value: Any) -> str:
    room = str(value or "lobby").strip()
    return room[:48] or "lobby"


def clean_name(value: Any) -> str:
    name = str(value or "anonymous").strip()
    return name[:32] or "anonymous"


def clean_wave_seed(value: Any) -> str:
    seed = str(value or make_client_id()).strip()
    return seed[:64] or make_client_id()


def clean_duration_ms(value: Any) -> int:
    try:
        duration = int(value)
    except Exception:
        duration = DEFAULT_WAVE_DURATION_MS
    return max(5_000, min(MAX_WAVE_DURATION_MS, duration))


def clean_wave_targets(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    targets: list[dict[str, Any]] = []
    for index, item in enumerate(value[:MAX_WAVE_TARGETS]):
        if not isinstance(item, dict):
            continue
        targets.append(
            {
                "id": str(item.get("id") or f"enemy-{index}")[:64],
                "x": clamp01(item.get("x")),
                "y": clamp01(item.get("y")),
            }
        )
    return targets


def public_wave(wave: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": wave["id"],
        "mode": wave.get("mode", "multiplayer"),
        "seed": wave.get("seed", ""),
        "targets": wave.get("targets", []),
        "started_at": int(wave.get("started_at") or 0),
        "duration_ms": int(wave.get("duration_ms") or DEFAULT_WAVE_DURATION_MS),
        "started_by": wave.get("started_by"),
        "started_by_name": wave.get("started_by_name", "Guest"),
        "scores": list((wave.get("scores") or {}).values()),
    }


async def handle_client(websocket, state: RelayState) -> None:
    client = RelayClient(id=make_client_id(), websocket=websocket)
    state.clients[client.id] = client
    try:
        async for raw in websocket:
            try:
                message = decode(raw)
            except Exception:
                await websocket.send(encode({"type": "error", "message": "invalid_json"}))
                continue

            message_type = message.get("type")
            if message_type == "join":
                room = clean_room(message.get("room"))
                if not state.can_join_room(client, room):
                    await websocket.send(encode({"type": "error", "message": "room_full"}))
                    continue
                client.name = clean_name(message.get("name"))
                client.color = parse_color(message.get("color"))
                state.join_room(client, room)
                await websocket.send(
                    encode(
                        {
                            "type": "welcome",
                            "id": client.id,
                            "room": room,
                            "peers": state.peers_for(client),
                            "wave": state.active_wave_for(room),
                            "ts": now_ms(),
                        }
                    )
                )
                await state.broadcast(
                    room,
                    {
                        "type": "peer_join",
                        "id": client.id,
                        "name": client.name,
                        "color": list(client.color),
                        "ts": now_ms(),
                    },
                    exclude_id=client.id,
                )
                print(f"{client.id} joined room={room!r} name={client.name!r}")
                continue

            if message_type == "cursor":
                if client.room is None:
                    continue
                tracking = bool(message.get("tracking"))
                x = None if message.get("x") is None else clamp01(message.get("x"))
                y = None if message.get("y") is None else clamp01(message.get("y"))
                await state.broadcast(
                    client.room,
                    {
                        "type": "cursor",
                        "id": client.id,
                        "name": client.name,
                        "color": list(client.color),
                        "x": x,
                        "y": y,
                        "tracking": tracking and x is not None and y is not None,
                        "seq": int(message.get("seq") or 0),
                        "client_ts": message.get("ts"),
                        "server_ts": now_ms(),
                    },
                    exclude_id=client.id,
                )
                continue

            if message_type == "wave_start":
                if client.room is None:
                    continue
                wave = state.start_wave(client.room, client, message)
                await state.broadcast(
                    client.room,
                    {
                        "type": "wave_start",
                        **wave,
                        "server_ts": now_ms(),
                    },
                )
                continue

            if message_type == "wave_hit":
                if client.room is None:
                    continue
                payload = state.update_wave_score(
                    client.room,
                    client,
                    message.get("wave_id"),
                    message.get("score"),
                    message.get("target_id"),
                )
                if payload is not None:
                    await state.broadcast(client.room, payload)
                continue

            await websocket.send(encode({"type": "error", "message": "unknown_type"}))
    finally:
        old_room = client.room
        state.leave_room(client)
        state.clients.pop(client.id, None)
        if old_room is not None:
            await state.broadcast(
                old_room,
                {
                    "type": "peer_leave",
                    "id": client.id,
                    "ts": now_ms(),
                },
            )
        print(f"{client.id} disconnected")


def _response(
    status_code: int,
    reason_phrase: str,
    body: bytes,
    content_type: str = "text/plain; charset=utf-8",
) -> Response:
    headers = Headers()
    headers["Content-Type"] = content_type
    headers["Content-Length"] = str(len(body))
    headers["X-Content-Type-Options"] = "nosniff"
    return Response(status_code, reason_phrase, headers, body)


def _static_response(web_dir: Path, request_path: str) -> Response:
    parsed = urlparse(request_path)
    rel_path = unquote(parsed.path).lstrip("/")
    if not rel_path:
        rel_path = "index.html"

    root = web_dir.resolve()
    target = (root / rel_path).resolve()
    if root != target and root not in target.parents:
        return _response(403, "Forbidden", b"Forbidden\n")
    if target.is_dir():
        target = target / "index.html"

    try:
        body = target.read_bytes()
    except FileNotFoundError:
        return _response(404, "Not Found", b"Not found\n")
    except OSError:
        return _response(500, "Internal Server Error", b"Could not read file\n")

    content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    cache_control = (
        "no-store"
        if target.suffix in {".html", ".js", ".css"}
        else "public, max-age=3600"
    )
    headers = Headers()
    headers["Content-Type"] = content_type
    headers["Content-Length"] = str(len(body))
    headers["Cache-Control"] = cache_control
    headers["X-Content-Type-Options"] = "nosniff"
    return Response(200, "OK", headers, body)


def make_static_process_request(web_dir: Path):
    def process_request(connection, request):
        upgrade = request.headers.get("Upgrade", "")
        if upgrade.lower() == "websocket":
            return None
        return _static_response(web_dir, request.path)

    return process_request


async def run_server(host: str, port: int, web_dir: Path | None = DEFAULT_WEB_DIR) -> None:
    state = RelayState()
    process_request = None
    if web_dir is not None:
        web_dir = web_dir.expanduser().resolve()
        if web_dir.exists():
            process_request = make_static_process_request(web_dir)
        else:
            print(f"Web client disabled because {web_dir} does not exist")

    async with serve(
        lambda websocket: handle_client(websocket, state),
        host,
        port,
        process_request=process_request,
        max_size=MAX_MESSAGE_BYTES,
    ):
        print(f"Gaze relay listening on ws://{host}:{port}")
        if process_request is not None:
            print(f"Browser client listening on http://{host}:{port}")
        await asyncio.Future()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shared gaze-cursor relay.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--web-dir",
        type=Path,
        default=DEFAULT_WEB_DIR,
        help="Directory to serve as the browser client. Defaults to ./web.",
    )
    parser.add_argument("--no-web", action="store_true", help="Disable the browser client.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_server(args.host, args.port, None if args.no_web else args.web_dir))
