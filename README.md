# Gaze Ninja

Browser-based shared gaze rooms, built as a ninja target game.

Each client estimates gaze locally, renders a cursor on a shared grid, and sends only
normalized cursor state to a small WebSocket relay. Webcam frames, eye crops, landmarks,
and model tensors stay on the user's device.

The current product path is a hosted HTTPS browser app. The room is a hangout by default,
and players opt into game runs when they are ready:

1. Open the shared URL.
2. Enter a name.
3. Enter the Dojo to calibrate and train a local personal NN adapter, or create/join a
   room.
4. Allow camera access.
5. Calibrate in fullscreen.
6. Run a Trial, play Solo, or start a Multiplayer wave from a room.

## Project Shape

- `web/` is the browser client and Gaze Ninja game surface.
- `shared_gaze/relay_server.py` serves `web/` and relays cursor messages.
- `scripts/export_browser_onnx.py` exports the local PyTorch gaze checkpoint for browser
  inference.
- `scripts/verify_browser_onnx.py` checks ONNX output against the PyTorch model.
- The legacy Python/macOS client remains in the repo as a benchmark path, not the primary
  user experience.

## Privacy Boundary

The relay receives:

- room code
- display name
- cursor color
- normalized `x` / `y`
- tracking state
- timestamps and sequence counters

The relay does not receive:

- webcam frames
- eye crops
- face landmarks
- MediaPipe outputs
- model tensors
- checkpoints
- personal training samples

Current personal NN training is browser-local and stored in browser storage.

## Setup

Use Python 3.11 for the Python tooling:

```bash
git clone git@github.com:JJ9276489/gaze-game.git
cd gaze-game
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For relay-only deployment:

```bash
python -m pip install -r requirements-relay.txt
```

For ONNX export and verification:

```bash
python -m pip install -r requirements-dev.txt
```

## Model Assets

The browser client looks for these optional model files:

```text
web/models/vision_gaze_spatial_geom.onnx
web/models/vision_gaze_latest.onnx
```

Model files are intentionally ignored by git. Generate or copy them onto the relay
machine before testing gaze inference. If a selected ONNX model is missing, the browser
falls back to a heuristic predictor so room networking can still be tested.

Export the spatial-geom checkpoint:

```bash
source .venv/bin/activate
python scripts/export_browser_onnx.py
python scripts/verify_browser_onnx.py
```

The default output path is:

```text
web/models/vision_gaze_spatial_geom.onnx
```

The PyTorch legacy path expects:

```text
models/vision_gaze_spatial_geom.pt
models/face_landmarker.task
```

Those files are also ignored.

## Local Run

Start the relay and browser client:

```bash
source .venv/bin/activate
python relay_server.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

For a two-client local test, open a second tab, join the same room, and enable `Mouse
mode` under `Connection`.

Useful in-session controls:

- `Full screen` enters or exits fullscreen.
- `Hide buttons` keeps the room code visible and removes distracting controls.
- `F` toggles fullscreen.
- `H` toggles hidden controls.

Game modes:

- `Dojo` spawns training dummies and uses them as local NN training labels.
- `Trial` measures held-out accuracy against fixed marks.
- `Solo` spawns enemy waves locally.
- `Multiplayer` starts a synchronized room wave with shared targets and relay-broadcast
  scores while still keeping webcam data local.

## Private Remote Run

For early remote tests, keep the relay bound to localhost and expose it through private
HTTPS, for example with Tailscale Serve:

```bash
cd ~/gaze-game-relay
source .venv/bin/activate
python -m pip install -r requirements-relay.txt
python relay_server.py --host 127.0.0.1 --port 8765
tailscale serve --bg 8765
```

Send testers the generated `https://...ts.net` URL and a room code. Browser camera access
requires HTTPS for remote pages.

Do not expose the included unauthenticated relay to the public internet. Before public
testing, add authentication, room limits, rate limits, message-size limits, idle cleanup,
and operator controls.

## Docs

- [Product target](docs/product-target.md)
- [Browser alpha operator notes](docs/browser-alpha.md)
- [Alpha tester guide](docs/alpha-tester-guide.md)
- [Relay operations](docs/relay-operations.md)

## Legacy Mac App

The packaged macOS app is retained for local comparison with the original PyTorch
runtime:

```bash
./scripts/build_macos_app.sh
```

The build output is ignored:

```text
dist/Gaze Game.app
dist/Gaze-Game-alpha-macos-arm64.zip
```

`relay_urls.example.txt` documents the old packaged-alpha relay URL mechanism.
