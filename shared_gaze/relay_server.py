import argparse
import asyncio
from dataclasses import dataclass
from typing import Any

try:
    from websockets.asyncio.server import serve
except ImportError:  # pragma: no cover - compatibility for older websockets
    from websockets import serve

from shared_gaze.protocol import clamp01, decode, encode, make_client_id, now_ms


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

    def join_room(self, client: RelayClient, room: str) -> None:
        self.leave_room(client)
        client.room = room
        self.rooms.setdefault(room, {})[client.id] = client

    def leave_room(self, client: RelayClient) -> None:
        if client.room is None:
            return
        room_clients = self.rooms.get(client.room)
        if room_clients is not None:
            room_clients.pop(client.id, None)
            if not room_clients:
                self.rooms.pop(client.room, None)
        client.room = None

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


async def run_server(host: str, port: int) -> None:
    state = RelayState()
    async with serve(lambda websocket: handle_client(websocket, state), host, port):
        print(f"Gaze relay listening on ws://{host}:{port}")
        await asyncio.Future()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shared gaze-cursor relay.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_server(args.host, args.port))

