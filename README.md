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
6. Train in the Dojo, then play Solo or start a Multiplayer wave from a room.

## Project Shape

- `web/` is the browser client and Gaze Ninja game surface.
- `shared_gaze/relay_server.py` serves `web/` and relays cursor messages.
- `scripts/export_browser_onnx.py` exports the local PyTorch gaze checkpoint for browser
  inference.
- `scripts/verify_browser_onnx.py` checks ONNX output against the PyTorch model.
- The legacy Python/macOS client remains in the repo as a benchmark path, not the primary
  user experience.

## Privacy Boundary

The relay receives only cursor-level state (room code, name, normalized coordinates,
tracking, timestamps, wave seeds/scores). Webcam frames, eye crops, face landmarks,
MediaPipe outputs, model tensors, checkpoints, and personal training samples stay on the
user's device. See
[docs/product-target.md#data-sent-over-the-network](docs/product-target.md#data-sent-over-the-network)
for the authoritative list.

## Setup

Python 3.11 for the local tooling. Pick the requirements file that matches the role:

```bash
git clone git@github.com:JJ9276489/gaze-game.git
cd gaze-game
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt        # legacy Python client + dev
python -m pip install -r requirements-relay.txt  # relay-only host
python -m pip install -r requirements-dev.txt    # also gets ONNX export/verify
```

## Model Assets

Model files are ignored by git; generate or copy them onto the relay machine before
testing gaze inference. If no ONNX model loads, the browser falls back to a heuristic
predictor so room networking still works.

Browser (primary):

```text
web/models/vision_gaze_spatial_geom.onnx
web/models/vision_gaze_latest.onnx
```

Export from the local PyTorch checkpoint:

```bash
python scripts/export_browser_onnx.py
python scripts/verify_browser_onnx.py
```

Legacy Python/macOS path:

```text
models/vision_gaze_spatial_geom.pt
models/face_landmarker.task
```

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

See [docs/alpha-tester-guide.md](docs/alpha-tester-guide.md) for in-session screen
controls and the Dojo / Solo / Multiplayer game modes.

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
