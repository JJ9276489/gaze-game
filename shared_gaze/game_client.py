import argparse
import asyncio
from collections import deque
from dataclasses import dataclass, field
import getpass
import hashlib
import os
from pathlib import Path
import time
from typing import Any

import pygame

try:
    from websockets.asyncio.client import connect
except ImportError:  # pragma: no cover - compatibility for older websockets
    from websockets import connect

from shared_gaze.config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FACE_LANDMARKER_PATH,
    DEFAULT_RELAY_URL,
    DEFAULT_RELAY_URLS,
)
from shared_gaze.protocol import decode, encode, make_cursor, make_join
from shared_gaze.rooms import generate_room_code, normalize_room_code


BACKGROUND = (5, 7, 11)
GRID = (18, 24, 33)
TEXT = (224, 229, 236)
MUTED = (116, 128, 146)
LOCAL_COLOR = (0, 232, 255)
PALETTE = [
    (255, 92, 124),
    (73, 221, 136),
    (255, 205, 92),
    (166, 132, 255),
    (255, 139, 68),
    (92, 176, 255),
]


@dataclass
class LocalReading:
    x: float | None
    y: float | None
    tracking: bool
    timestamp_ms: int
    model_label: str | None = None


@dataclass
class LobbyChoice:
    name: str
    room: str
    display_room: str


@dataclass
class CursorState:
    id: str
    name: str
    color: tuple[int, int, int]
    point: tuple[float, float] | None = None
    tracking: bool = False
    last_seen: float = 0.0
    trail: deque[tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=80)
    )

    def update(self, x: float | None, y: float | None, tracking: bool, now: float) -> None:
        self.tracking = tracking and x is not None and y is not None
        self.last_seen = now
        if self.tracking:
            self.point = (float(x), float(y))
            self.trail.append((float(x), float(y), now))


class MouseSource:
    def start(self) -> "MouseSource":
        return self

    def close(self) -> None:
        pass

    def read(self, surface_size: tuple[int, int]) -> LocalReading:
        width, height = surface_size
        mouse_x, mouse_y = pygame.mouse.get_pos()
        return LocalReading(
            x=max(0.0, min(1.0, mouse_x / max(width - 1, 1))),
            y=max(0.0, min(1.0, mouse_y / max(height - 1, 1))),
            tracking=True,
            timestamp_ms=int(time.time() * 1000),
            model_label="mouse",
        )


class GazeSource:
    def __init__(
        self,
        checkpoint_path: Path,
        face_landmarker_path: Path,
        requested_device: str | None,
        camera_index: int,
        smoothing_alpha: float,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.face_landmarker_path = face_landmarker_path
        self.requested_device = requested_device
        self.camera_index = camera_index
        self.smoothing_alpha = smoothing_alpha
        self.runtime = None

    def start(self) -> "GazeSource":
        from shared_gaze.gaze_runtime import GazeRuntime

        self.runtime = GazeRuntime(
            checkpoint_path=self.checkpoint_path,
            face_landmarker_path=self.face_landmarker_path,
            requested_device=self.requested_device,
            camera_index=self.camera_index,
            smoothing_alpha=self.smoothing_alpha,
        ).start()
        return self

    def close(self) -> None:
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None

    def read(self, surface_size: tuple[int, int]) -> LocalReading:
        if self.runtime is None:
            raise RuntimeError("GazeSource.start() must be called before read()")
        reading = self.runtime.read()
        return LocalReading(
            x=reading.x,
            y=reading.y,
            tracking=reading.tracking,
            timestamp_ms=reading.timestamp_ms,
            model_label=reading.model_label,
        )


def color_for_name(name: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return PALETTE[digest[0] % len(PALETTE)]


def parse_hex_color(value: str | None, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if not value:
        return fallback
    color = value.strip().lstrip("#")
    if len(color) != 6:
        raise ValueError("Color must be RRGGBB or #RRGGBB")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def parse_relay_urls(values: list[str] | None) -> list[str]:
    urls: list[str] = []
    for value in values or []:
        for item in value.replace(";", ",").split(","):
            url = item.strip()
            if url and url not in urls:
                urls.append(url)
    return urls or list(DEFAULT_RELAY_URLS)


def color_from_payload(value: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return fallback
    try:
        return tuple(max(0, min(255, int(channel))) for channel in value)  # type: ignore[return-value]
    except Exception:
        return fallback


def denormalize(point: tuple[float, float], size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    return (
        int(max(0.0, min(1.0, point[0])) * (width - 1)),
        int(max(0.0, min(1.0, point[1])) * (height - 1)),
    )


def dim_color(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(int(BACKGROUND[index] * (1.0 - amount) + color[index] * amount) for index in range(3))


def lobby_layout(size: tuple[int, int]) -> dict[str, pygame.Rect]:
    width, height = size
    panel_width = min(620, width - 48)
    panel_x = (width - panel_width) // 2
    top = max(40, height // 2 - 185)
    return {
        "name": pygame.Rect(panel_x, top + 118, panel_width, 44),
        "room": pygame.Rect(panel_x, top + 202, panel_width, 44),
        "create": pygame.Rect(panel_x, top + 286, 190, 46),
        "join": pygame.Rect(panel_x + panel_width - 190, top + 286, 190, 46),
    }


def draw_lobby_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = TEXT,
) -> None:
    surface.blit(font.render(text, True, color), position)


def wrap_text(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.size(candidate)[0] <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        while font.size(word)[0] > max_width and len(word) > 1:
            end = len(word)
            while end > 1 and font.size(word[:end])[0] > max_width:
                end -= 1
            lines.append(word[:end])
            word = word[end:]
        current = word

    if current:
        lines.append(current)
    return lines


def show_error_screen(
    title: str,
    lines: list[str],
    existing_surface: pygame.Surface | None = None,
) -> None:
    print(f"{title}: {' '.join(lines)}")
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        return

    pygame.init()
    pygame.display.set_caption("Gaze Game")
    owns_display = existing_surface is None
    surface = existing_surface or pygame.display.set_mode((880, 520), pygame.RESIZABLE)
    title_font = pygame.font.SysFont("Helvetica", 36)
    font = pygame.font.SysFont("Helvetica", 20)
    small_font = pygame.font.SysFont("Helvetica", 16)
    clock = pygame.time.Clock()

    running = True
    while running:
        width, height = surface.get_size()
        close_rect = pygame.Rect(0, 0, 140, 42)
        close_rect.midbottom = (width // 2, height - 42)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (
                pygame.K_ESCAPE,
                pygame.K_RETURN,
            ):
                running = False
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and close_rect.collidepoint(event.pos)
            ):
                running = False

        surface.fill(BACKGROUND)
        draw_grid(surface)

        panel = pygame.Rect(0, 0, min(760, width - 32), min(420, height - 32))
        panel.center = (width // 2, height // 2)
        pygame.draw.rect(surface, (8, 12, 18), panel, border_radius=8)
        pygame.draw.rect(surface, (86, 59, 70), panel, width=1, border_radius=8)

        draw_lobby_text(surface, title_font, title, (panel.x + 42, panel.y + 38), TEXT)
        y = panel.y + 98
        for line in lines:
            for wrapped in wrap_text(font, line, panel.width - 84):
                draw_lobby_text(surface, font, wrapped, (panel.x + 44, y), TEXT)
                y += 30
            y += 6

        draw_lobby_text(
            surface,
            small_font,
            "Press Esc, Return, or Close to exit.",
            (panel.x + 44, panel.bottom - 54),
            MUTED,
        )
        draw_button(surface, font, close_rect, "Close", False)
        pygame.display.flip()
        clock.tick(30)

    if owns_display:
        pygame.quit()


def startup_error_message(error: Exception) -> tuple[str, list[str]]:
    detail = str(error) or type(error).__name__
    lower_detail = detail.lower()
    if "webcam" in lower_detail or "camera" in lower_detail:
        return (
            "Camera unavailable",
            [
                "Gaze Game could not open the MacBook camera.",
                (
                    "Allow Camera for Gaze Game in System Settings > "
                    "Privacy & Security > Camera, then reopen the app."
                ),
                f"Details: {detail}",
            ],
        )
    return (
        "Gaze tracking failed",
        [
            "The gaze model, face landmarker, or camera could not start.",
            f"Details: {detail}",
        ],
    )


def draw_input(
    surface: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    label: str,
    value: str,
    active: bool,
) -> None:
    border = LOCAL_COLOR if active else (62, 75, 92)
    pygame.draw.rect(surface, (10, 14, 21), rect, border_radius=6)
    pygame.draw.rect(surface, border, rect, width=2, border_radius=6)
    draw_lobby_text(surface, font, label, (rect.x, rect.y - 28), MUTED)
    shown = value or ""
    if active and int(time.monotonic() * 2) % 2 == 0:
        shown += "|"
    draw_lobby_text(surface, font, shown, (rect.x + 14, rect.y + 10), TEXT if value else MUTED)


def draw_button(
    surface: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    label: str,
    primary: bool,
) -> None:
    fill = (0, 84, 98) if primary else (22, 29, 40)
    border = LOCAL_COLOR if primary else (78, 91, 108)
    pygame.draw.rect(surface, fill, rect, border_radius=6)
    pygame.draw.rect(surface, border, rect, width=2, border_radius=6)
    text = font.render(label, True, TEXT)
    surface.blit(
        text,
        (
            rect.centerx - text.get_width() // 2,
            rect.centery - text.get_height() // 2,
        ),
    )


def prompt_for_lobby(args: argparse.Namespace) -> LobbyChoice:
    pygame.init()
    pygame.display.set_caption("Gaze Game")
    surface = pygame.display.set_mode((820, 520), pygame.RESIZABLE)
    title_font = pygame.font.SysFont("Helvetica", 42)
    font = pygame.font.SysFont("Helvetica", 22)
    small_font = pygame.font.SysFont("Helvetica", 16)
    clock = pygame.time.Clock()

    name = (args.name or getpass.getuser() or "Player")[:24]
    room_value = ""
    active_field = "room"
    message = "Create a room, or enter the code someone sent you."

    while True:
        layout = lobby_layout(surface.get_size())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise SystemExit
                if event.key == pygame.K_TAB:
                    active_field = "name" if active_field == "room" else "room"
                    continue
                if event.key == pygame.K_RETURN:
                    if room_value.strip():
                        try:
                            room = normalize_room_code(room_value)
                        except ValueError as error:
                            message = str(error)
                            continue
                        pygame.quit()
                        return LobbyChoice(name=name.strip() or "Player", room=room, display_room=room_value.upper())
                    message = "Enter a room code, or click Create Room."
                    continue
                if event.key == pygame.K_BACKSPACE:
                    if active_field == "name":
                        name = name[:-1]
                    else:
                        room_value = room_value[:-1]
                    continue
                if event.unicode and event.unicode.isprintable():
                    if active_field == "name":
                        if len(name) < 24:
                            name += event.unicode
                    elif len(room_value) < 16 and (event.unicode.isalnum() or event.unicode in "- "):
                        room_value += event.unicode.upper()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if layout["name"].collidepoint(event.pos):
                    active_field = "name"
                elif layout["room"].collidepoint(event.pos):
                    active_field = "room"
                elif layout["create"].collidepoint(event.pos):
                    display_room = generate_room_code()
                    pygame.quit()
                    return LobbyChoice(
                        name=name.strip() or "Player",
                        room=normalize_room_code(display_room),
                        display_room=display_room,
                    )
                elif layout["join"].collidepoint(event.pos):
                    if not room_value.strip():
                        message = "Enter the room code first."
                        continue
                    try:
                        room = normalize_room_code(room_value)
                    except ValueError as error:
                        message = str(error)
                        continue
                    pygame.quit()
                    return LobbyChoice(
                        name=name.strip() or "Player",
                        room=room,
                        display_room=room_value.upper(),
                    )

        surface.fill(BACKGROUND)
        draw_grid(surface)
        width, height = surface.get_size()
        panel = pygame.Rect(0, 0, min(720, width - 32), min(440, height - 32))
        panel.center = (width // 2, height // 2)
        pygame.draw.rect(surface, (8, 12, 18), panel, border_radius=8)
        pygame.draw.rect(surface, (40, 52, 68), panel, width=1, border_radius=8)

        draw_lobby_text(surface, title_font, "Gaze Game", (panel.x + 48, panel.y + 36))
        draw_lobby_text(surface, small_font, message, (panel.x + 50, panel.y + 88), MUTED)
        draw_input(surface, font, layout["name"], "Name", name, active_field == "name")
        draw_input(surface, font, layout["room"], "Room Code", room_value, active_field == "room")
        draw_button(surface, font, layout["create"], "Create Room", True)
        draw_button(surface, font, layout["join"], "Join Room", False)
        draw_lobby_text(
            surface,
            small_font,
            "Remote testers need access to the same relay URL.",
            (panel.x + 50, panel.bottom - 48),
            MUTED,
        )
        pygame.display.flip()
        clock.tick(60)


def draw_grid(surface: pygame.Surface) -> None:
    width, height = surface.get_size()
    spacing = 120
    for x in range(0, width, spacing):
        pygame.draw.line(surface, GRID, (x, 0), (x, height), 1)
    for y in range(0, height, spacing):
        pygame.draw.line(surface, GRID, (0, y), (width, y), 1)


def draw_trail(
    surface: pygame.Surface,
    state: CursorState,
    now: float,
) -> None:
    if len(state.trail) < 2:
        return

    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    points = list(state.trail)
    for previous, current in zip(points, points[1:]):
        age = now - current[2]
        if age > 2.5:
            continue
        alpha = int(max(0.0, 1.0 - age / 2.5) * 130)
        color = (*state.color, alpha)
        pygame.draw.line(
            overlay,
            color,
            denormalize((previous[0], previous[1]), surface.get_size()),
            denormalize((current[0], current[1]), surface.get_size()),
            3,
        )
    surface.blit(overlay, (0, 0))


def draw_cursor(
    surface: pygame.Surface,
    font: pygame.font.Font,
    state: CursorState,
    now: float,
    local: bool = False,
) -> None:
    if state.point is None:
        return

    if local:
        amount = 1.0 if state.tracking else 0.35
    else:
        age = now - state.last_seen
        amount = max(0.0, min(1.0, 1.0 - age / 3.0))
        if not state.tracking:
            amount *= 0.35
    if amount <= 0.02:
        return

    x, y = denormalize(state.point, surface.get_size())
    color = dim_color(state.color, amount)
    radius = 25 if local else 20
    pygame.draw.circle(surface, color, (x, y), radius, width=2)
    pygame.draw.circle(surface, color, (x, y), 6 if local else 5)
    pygame.draw.line(surface, color, (x - radius - 12, y), (x + radius + 12, y), 1)
    pygame.draw.line(surface, color, (x, y - radius - 12), (x, y + radius + 12), 1)

    label = font.render(state.name, True, color)
    label_x = min(x + 18, surface.get_width() - label.get_width() - 12)
    label_y = max(12, y - radius - label.get_height() - 8)
    surface.blit(label, (label_x, label_y))


def draw_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    room: str,
    server: str,
    local_state: CursorState,
    remote_count: int,
    model_label: str | None,
    connected: bool,
) -> None:
    title = font.render(f"room {room}", True, TEXT)
    surface.blit(title, (28, 24))

    status_parts = [
        "connected" if connected else "offline",
        f"{remote_count} peer{'s' if remote_count != 1 else ''}",
        f"local {'tracking' if local_state.tracking else 'lost'}",
    ]
    status = small_font.render(" | ".join(status_parts), True, MUTED)
    surface.blit(status, (28, 55))

    model = small_font.render(model_label or "model unavailable", True, MUTED)
    surface.blit(model, (28, surface.get_height() - 36))

    endpoint = small_font.render(server, True, MUTED)
    surface.blit(endpoint, (surface.get_width() - endpoint.get_width() - 28, surface.get_height() - 36))


async def receive_loop(
    websocket,
    remotes: dict[str, CursorState],
    local_id_holder: dict[str, str | None],
) -> None:
    async for raw in websocket:
        message = decode(raw)
        message_type = message.get("type")
        now = time.monotonic()

        if message_type == "welcome":
            local_id_holder["id"] = str(message.get("id"))
            for peer in message.get("peers") or []:
                peer_id = str(peer.get("id"))
                remotes[peer_id] = CursorState(
                    id=peer_id,
                    name=str(peer.get("name") or "peer"),
                    color=color_from_payload(peer.get("color"), color_for_name(peer_id)),
                    last_seen=now,
                )
            continue

        if message_type == "peer_join":
            peer_id = str(message.get("id"))
            if peer_id == local_id_holder.get("id"):
                continue
            remotes[peer_id] = CursorState(
                id=peer_id,
                name=str(message.get("name") or "peer"),
                color=color_from_payload(message.get("color"), color_for_name(peer_id)),
                last_seen=now,
            )
            continue

        if message_type == "peer_leave":
            remotes.pop(str(message.get("id")), None)
            continue

        if message_type == "cursor":
            peer_id = str(message.get("id"))
            if peer_id == local_id_holder.get("id"):
                continue
            state = remotes.get(peer_id)
            if state is None:
                state = CursorState(
                    id=peer_id,
                    name=str(message.get("name") or "peer"),
                    color=color_from_payload(message.get("color"), color_for_name(peer_id)),
                    last_seen=now,
                )
                remotes[peer_id] = state
            state.update(
                message.get("x"),
                message.get("y"),
                bool(message.get("tracking")),
                now,
            )


async def game_loop(
    args: argparse.Namespace,
    websocket,
    source: MouseSource | GazeSource,
    local_color: tuple[int, int, int],
    relay_label: str,
) -> None:
    pygame.init()
    pygame.display.set_caption("Gaze Game")
    if args.windowed:
        surface = pygame.display.set_mode((args.width, args.height), pygame.RESIZABLE)
    else:
        surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    font = pygame.font.SysFont("Helvetica", 24)
    small_font = pygame.font.SysFont("Helvetica", 16)
    clock = pygame.time.Clock()

    remotes: dict[str, CursorState] = {}
    local_id_holder: dict[str, str | None] = {"id": None}
    receive_task = None
    connected = websocket is not None

    local_state = CursorState(
        id="local",
        name=args.name,
        color=local_color,
        last_seen=time.monotonic(),
    )
    next_send_at = 0.0
    seq = 0
    running = True
    model_label = "starting"

    try:
        try:
            source.start()
        except Exception as error:
            title, lines = startup_error_message(error)
            show_error_screen(title, lines, existing_surface=surface)
            return

        if websocket is not None:
            try:
                await websocket.send(encode(make_join(args.room, args.name, local_color)))
                receive_task = asyncio.create_task(
                    receive_loop(websocket, remotes, local_id_holder)
                )
            except Exception:
                connected = False
                websocket = None

        while running:
            now = time.monotonic()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            reading = source.read(surface.get_size())
            model_label = reading.model_label or model_label
            local_state.update(reading.x, reading.y, reading.tracking, now)

            if websocket is not None and now >= next_send_at:
                seq += 1
                next_send_at = now + 1.0 / max(args.send_hz, 1)
                try:
                    await websocket.send(
                        encode(
                            make_cursor(
                                args.room,
                                reading.x,
                                reading.y,
                                reading.tracking,
                                seq,
                                reading.timestamp_ms,
                            )
                        )
                    )
                    connected = True
                except Exception:
                    connected = False
                    websocket = None

            surface.fill(BACKGROUND)
            draw_grid(surface)
            draw_trail(surface, local_state, now)
            for remote in remotes.values():
                draw_trail(surface, remote, now)
            for remote in remotes.values():
                draw_cursor(surface, small_font, remote, now)
            draw_cursor(surface, small_font, local_state, now, local=True)
            draw_hud(
                surface,
                font,
                small_font,
                getattr(args, "display_room", args.room),
                relay_label,
                local_state,
                len(remotes),
                model_label,
                connected,
            )
            pygame.display.flip()
            clock.tick(args.fps)
            await asyncio.sleep(0)
    finally:
        source.close()
        if receive_task is not None:
            receive_task.cancel()
        pygame.quit()


async def run_client(args: argparse.Namespace) -> None:
    if args.room is None:
        choice = prompt_for_lobby(args)
        args.name = choice.name
        args.room = choice.room
        args.display_room = choice.display_room

    local_color = parse_hex_color(args.color, color_for_name(args.name) if args.color else LOCAL_COLOR)
    if args.mouse:
        source: MouseSource | GazeSource = MouseSource()
    else:
        source = GazeSource(
            checkpoint_path=args.checkpoint,
            face_landmarker_path=args.face_model,
            requested_device=args.device,
            camera_index=args.camera_index,
            smoothing_alpha=args.smoothing_alpha,
        )

    if args.offline:
        await game_loop(args, None, source, local_color, "offline")
        return

    errors: list[str] = []
    for relay_url in args.servers:
        try:
            print(f"Connecting to relay: {relay_url}")
            async with connect(
                relay_url,
                open_timeout=args.connect_timeout,
                ping_interval=20,
                ping_timeout=20,
            ) as websocket:
                print(f"Connected to relay: {relay_url}")
                await game_loop(args, websocket, source, local_color, relay_url)
                return
        except Exception as error:
            errors.append(f"{relay_url}: {error}")
            print(f"Relay unavailable: {relay_url} ({error})")

    show_error_screen(
        "Relay unavailable",
        [
            "Gaze Game could not connect to any relay.",
            "Check the relay URL, network access, and relay process, then try again.",
            *errors[:3],
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shared gaze-cursor game client.")
    parser.add_argument(
        "--server",
        action="append",
        default=None,
        help=(
            "Relay URL. Can be repeated or comma-separated. "
            f"Defaults to {DEFAULT_RELAY_URL} plus fallbacks."
        ),
    )
    parser.add_argument("--room", default=None)
    parser.add_argument(
        "--create-room",
        action="store_true",
        help="Generate a shareable room code instead of using --room.",
    )
    parser.add_argument("--name", default=getpass.getuser())
    parser.add_argument("--color", default=None, help="Local cursor color as #RRGGBB.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--face-model", type=Path, default=DEFAULT_FACE_LANDMARKER_PATH)
    parser.add_argument("--device", default=None, help="Torch device override, e.g. mps or cpu.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--send-hz", type=float, default=30.0)
    parser.add_argument("--connect-timeout", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--mouse", action="store_true", help="Use the mouse as the local cursor.")
    parser.add_argument("--offline", action="store_true", help="Run without connecting to a relay.")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if args.create_room:
        args.display_room = generate_room_code()
        args.room = normalize_room_code(args.display_room)
        print(f"Created room code: {args.display_room}")
    elif args.room:
        args.display_room = args.room.upper()
        args.room = normalize_room_code(args.room)
    else:
        args.display_room = None
        args.room = None
    args.servers = parse_relay_urls(args.server)
    args.server = args.servers[0]
    return args


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_client(args))
    except KeyboardInterrupt:
        pass
    except SystemExit:
        raise
    except Exception as error:
        show_error_screen(
            "Gaze Game stopped",
            [
                "An unexpected error closed the session.",
                f"Details: {error}",
            ],
        )
