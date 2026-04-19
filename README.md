# Gaze Game

Shared gaze-cursor rooms for MacBooks.

Each laptop runs gaze inference locally and sends only normalized cursor coordinates to a
small WebSocket relay. The relay broadcasts cursor positions to everyone in the same
room. Webcam frames, eye crops, face landmarks, and model tensors stay on the local
machine.

The product target is a packaged macOS app. Users should install `Gaze Game.app`, allow
Camera when macOS prompts, enter a name, then click Create Room or Join Room. Users
should not need Terminal, Python, or a self-hosted server once a production relay exists.

See [docs/product-target.md](docs/product-target.md) for the intended install/connect
flow.

## Relay Configuration

By default, the development client tries a local relay:

```text
ws://127.0.0.1:8765
```

Override relay URLs at runtime with either command-line flags:

```bash
python gaze_game.py --server ws://127.0.0.1:8765 --room K7M-4QX
```

Or environment variables:

```bash
GAZE_GAME_RELAY_URLS="wss://relay.example.com,ws://127.0.0.1:8765" python gaze_game.py
```

Do not expose the included unauthenticated relay directly to the public internet. For
public testing, put it behind TLS and add authentication, rate limits, room limits, and
basic abuse controls first.

Packaged alpha builds can also be configured without Terminal. The build script can put a
`relay_urls.txt` file in the same unzipped folder as `Gaze Game.app`:

```text
Gaze-Game-alpha-macos-arm64/
  Gaze Game.app
  relay_urls.txt
  README-FIRST.txt
```

Use one relay URL per line, or comma-separated URLs. Lines can include comments after
`#`. See [relay_urls.example.txt](relay_urls.example.txt).

## Model Assets

The app expects:

```text
models/vision_gaze_spatial_geom.pt
models/face_landmarker.task
```

Those files are intentionally ignored by git. The build script can stage them from a
sibling `eye-cursor` checkout if available:

```text
../eye-cursor/models/
```

That keeps `eye-cursor` as the research/training project while this project stays focused
on multiplayer rendering and networking.

## Developer Setup

```bash
git clone git@github.com:JJ9276489/gaze-game.git
cd gaze-game
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use Python 3.11. Newer Python versions may not work with the pinned native dependencies.

## Developer Run

Start the relay:

```bash
python relay_server.py --host 127.0.0.1 --port 8765
```

Launch the game client:

```bash
python gaze_game.py
```

Start one client with the gaze model and a generated room:

```bash
python gaze_game.py --create-room --name Player
```

Join a known room code:

```bash
python gaze_game.py --room K7M-4QX --name Player
```

Start a second local test client controlled by the mouse:

```bash
python gaze_game.py --room K7M-4QX --name Mouse --mouse --windowed
```

Use `Esc` to quit.

## Build A Mac App

Install build dependencies and create an app bundle:

```bash
./scripts/build_macos_app.sh
```

The PyInstaller spec includes `NSCameraUsageDescription`, which lets macOS show a normal
Camera permission prompt for `Gaze Game.app`. The built app is written to:

```text
dist/Gaze Game.app
```

The build script stages local model assets from `../eye-cursor/models/` into the ignored
`models/` directory before packaging. It also locally signs the app with the camera
entitlement and writes an ignored alpha zip:

```text
dist/Gaze-Game-alpha-macos-arm64.zip
```

If an ignored `relay_urls.local.txt` file exists, the build script copies it into the zip
as `relay_urls.txt` in the same folder as `Gaze Game.app`. This is the intended way to
send a preconfigured private alpha build to non-technical testers. The user-facing
instructions should simply say to unzip the folder and open the app from there.

This local build is for Apple Silicon Macs and is not Developer ID notarized. Testers may
need to right-click `Gaze Game.app` and choose Open the first time.

## Remote Alpha Tester Flow

Use this flow for private remote testing before there is a production relay.

Send the tester [docs/alpha-tester-guide.md](docs/alpha-tester-guide.md), or send the
`README-FIRST.txt` that is included in the alpha zip.

Relay setup:

1. Host `relay_server.py` on a machine reachable by every tester.
2. Prefer a private network for early alpha tests.
3. Use a public `wss://` endpoint only after adding authentication and abuse controls.
4. Configure clients with `--server`, `GAZE_GAME_RELAY_URL`, or `GAZE_GAME_RELAY_URLS`.
5. Keep relay logs open during the first test so you can see joins and disconnects.

Tester setup:

1. Tester uses an Apple Silicon MacBook.
2. Tester receives the latest `dist/Gaze-Game-alpha-macos-arm64.zip`.
3. Tester unzips it.
4. Tester right-clicks `Gaze Game.app` and chooses Open the first time.
5. Tester allows Camera when macOS asks.

Room test:

1. One person opens `Gaze Game.app`, enters a name, and clicks Create Room.
2. They send the room code, for example `ABC-123`.
3. The other person opens `Gaze Game.app`, enters a name and that room code, then clicks Join Room.
4. Both people should see shared cursors on the dark grid.

Troubleshooting:

- `Relay unavailable`: the relay URL is wrong, unreachable, or the relay process is down.
- `Camera unavailable`: camera permission was denied, or another app is using the camera.
  Re-enable it in System Settings > Privacy & Security > Camera.
- App will not open: this alpha is locally signed but not Developer ID notarized, so use
  right-click > Open the first time.
- Bad gaze quality: expected for users who have not trained their own model yet. The
  networking test is still valid if their cursor appears.

## Relay Operations

See [docs/relay-operations.md](docs/relay-operations.md) for generic relay deployment
notes.

## Model Notes

The architecture is shared, but the checkpoint is personal. Each participant should
eventually collect and train their own `spatial_geom` checkpoint on their own MacBook,
then run this app with:

```bash
python gaze_game.py --checkpoint /path/to/vision_gaze_spatial_geom.pt
```

The app sends only `x`, `y`, `tracking`, timestamp, name, and color.
