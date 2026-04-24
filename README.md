# Gaze Game

Shared gaze-cursor rooms in the browser.

Each client runs gaze inference locally and sends only normalized cursor coordinates to a
small WebSocket relay. The relay broadcasts cursor positions to everyone in the same
room. Webcam frames, eye crops, face landmarks, and model tensors stay on the local
machine.

The product direction is one client path: a hosted browser app. Each participant opens an
HTTPS URL, allows camera access, joins a room, runs a short fullscreen calibration, and
starts sharing a cursor. The relay stays a small websocket/static host and never receives
camera frames or landmarks.

See [docs/product-target.md](docs/product-target.md) for the intended install/connect
flow.

## Relay Configuration

The browser client discovers the relay through the same origin it was served from, so no
per-user configuration is needed once testers have the HTTPS URL.

Do not expose the included unauthenticated relay directly to the public internet. For
public testing, put it behind TLS and add authentication, rate limits, room limits, and
basic abuse controls first.

The legacy Python client is configurable at runtime. By default it tries a local relay at
`ws://127.0.0.1:8765`. Override with flags or env:

```bash
python gaze_game.py --server ws://127.0.0.1:8765 --room K7M-4QX
GAZE_GAME_RELAY_URLS="wss://relay.example.com,ws://127.0.0.1:8765" python gaze_game.py
```

Packaged legacy alpha builds can also be configured without Terminal. The build script
can put a `relay_urls.txt` file in the same unzipped folder as `Gaze Game.app`:

```text
Gaze-Game-alpha-macos-arm64/
  Gaze Game.app
  relay_urls.txt
  README-FIRST.txt
```

Use one relay URL per line, or comma-separated URLs. Lines can include comments after
`#`. See [relay_urls.example.txt](relay_urls.example.txt).

## Model Assets

The browser client loads `web/models/vision_gaze_spatial_geom.onnx`. See the
[Browser App](#browser-app) section below for how to export it.

The legacy Python/macOS path expects local PyTorch assets:

```text
models/vision_gaze_spatial_geom.pt
models/face_landmarker.task
```

All model files are intentionally ignored by git. The macOS build script can stage the
PyTorch assets from a sibling `eye-cursor` checkout if available:

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

## Browser App

The browser client lives in `web/` and is served by `relay_server.py` by default. It is
the primary product path and supports:

- room creation and joining
- same-origin websocket relay discovery
- browser camera permission prompts
- in-browser MediaPipe Face Landmarker preprocessing
- in-browser ONNX Runtime Web inference with the exported gaze checkpoint
- five-point fullscreen calibration stored in the browser
- mouse mode fallback for devices without usable gaze

Export the local PyTorch checkpoint to the browser model path:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/export_browser_onnx.py
python scripts/verify_browser_onnx.py
```

The exported model is written to `web/models/vision_gaze_spatial_geom.onnx` and is
intentionally ignored by git, like the PyTorch checkpoint. Keep it on the private relay
machine or inside private alpha deploy artifacts, not in the public repo.

### Local Run

Start the relay and browser client:

```bash
python relay_server.py --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`, then:

1. Enter a name.
2. Click `Create room`.
3. Allow camera access.
4. Click `Calibrate`.
5. Keep the page fullscreen and stare at each target until it moves.

For a two-client test on one computer, open the page in a second tab, join the same
room, and enable `Mouse mode` under `Connection`.

### Private Remote Run

Serve the same port through a private HTTPS URL, such as Tailscale Serve:

```bash
python -m pip install -r requirements-relay.txt
python relay_server.py --host 127.0.0.1 --port 8765
tailscale serve --bg 8765
```

Send the generated `https://...ts.net` URL and a room code. Testers should join, allow
camera, click `Calibrate`, and keep the page fullscreen during calibration. See
[docs/browser-alpha.md](docs/browser-alpha.md) for more detail.

## Legacy Mac App

The packaged macOS app remains useful as a local benchmark against the original PyTorch
runtime, but it is no longer the primary user path. Build the app bundle:

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

Use this flow for private browser testing before there is a public relay.

Relay setup:

1. Host `relay_server.py` on a machine reachable by every tester.
2. Expose it over private HTTPS such as Tailscale Serve.
3. Keep the relay bound to `127.0.0.1` on the host and let the HTTPS layer front it.
4. Keep relay logs open during first tests so joins and disconnects are visible.

Tester setup:

1. Tester opens the shared HTTPS URL.
2. Tester enters a name.
3. Tester creates or joins a room.
4. Tester allows camera access.
5. Tester clicks `Calibrate`.

Room test:

1. One person creates a room and sends the short code.
2. The other person joins the same room.
3. Both people finish calibration.
4. Both people should see shared cursors on the dark grid.

Troubleshooting:

- `Could not connect to relay`: the host relay is down or the HTTPS front door is misconfigured.
- `Camera access needs HTTPS or localhost`: use the shared HTTPS URL, not plain HTTP.
- Bad gaze quality: recalibrate in fullscreen and keep your head still during the five targets.

## Relay Operations

See [docs/relay-operations.md](docs/relay-operations.md) for generic relay deployment
notes.

## Model Notes

The architecture is shared, but the checkpoint is personal. Each participant should
eventually collect and train their own `spatial_geom` checkpoint on their own MacBook,
then re-export it to ONNX for the browser with `scripts/export_browser_onnx.py`. The
legacy Python client can run a personal checkpoint directly:

```bash
python gaze_game.py --checkpoint /path/to/vision_gaze_spatial_geom.pt
```

Regardless of the path, the client sends only `x`, `y`, `tracking`, timestamp, name, and
color over the wire.
