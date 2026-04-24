# Browser Alpha

The browser app is the primary product path. Testers join from Chrome, Edge, Safari, or a
Chromebook without installing a native app. It uses the webcam through the browser,
estimates gaze locally, calibrates against the current viewport, and sends only cursor
coordinates to the relay.

## What Works

- laptop and desktop browsers with webcam access
- Chromebook browsers
- local two-client tests from one computer
- remote private tests through Tailscale or another private HTTPS front door
- ONNX-exported gaze checkpoint inference
- five-point fullscreen calibration with saved local browser state
- browser-local personal NN adapter training
- held-out testing and timed target competition
- heuristic fallback if the ONNX model asset is missing
- mouse mode when camera gaze is unavailable

## Model Asset

The browser client loads:

```text
web/models/vision_gaze_spatial_geom.onnx
web/models/vision_gaze_latest.onnx
```

Generate it from the local PyTorch checkpoint before deploying the browser alpha:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/export_browser_onnx.py
python scripts/verify_browser_onnx.py
```

ONNX files are ignored by git because they are derived from local checkpoints. Put the
needed model files on the private relay machine with the web client, but do not commit
them to the public repo.

## Local Test

Start the relay and browser app:

```bash
cd gaze-game
source .venv/bin/activate
python relay_server.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

For a two-client test on one computer:

1. Open the page in one browser tab.
2. Enter your name.
3. Click Create room.
4. Click `Calibrate` and keep staring at each target until it moves.
5. Click `Train NN` and complete a target run.
6. Click `Test` to measure the held-out error.
7. Open the page in a second tab or another browser.
8. Enter a different name.
9. Enter the same room code.
10. Open Connection and enable Mouse mode.
11. Click Join room.

## Private Remote Test With Tailscale

Run the relay/browser app on the Linux relay machine:

```bash
cd ~/gaze-game-relay
source .venv/bin/activate
python -m pip install -r requirements-relay.txt
python relay_server.py --host 127.0.0.1 --port 8765
```

In another terminal on that relay machine, expose it privately through Tailscale Serve:

```bash
tailscale serve --bg 8765
```

Tailscale prints a private HTTPS URL like:

```text
https://relay-machine.tailnet-name.ts.net
```

Send that URL to testers who are signed in to the same Tailscale network or have accepted
a Tailscale share for the relay machine.

## Tester Steps

1. Install Tailscale if the organizer says the test uses Tailscale.
2. Sign in or accept the relay-machine share.
3. Open the HTTPS URL from the organizer.
4. Enter a name.
5. Click Create room, or enter a room code and click Join room.
6. Allow camera access when the browser asks.
7. Click `Calibrate`.
8. Click `Train NN`.
9. Click `Test` or `Compete`.

## Multi-User Behavior

Rooms share cursor state only. Calibration, training, and testing are per-user browser
runs. One user starting `Train NN` does not start training for anyone else, and the app
hides that user's cursor from peers during training and testing so stale target movement
does not distract the room.

`Compete` is the current shared gameplay mode. It still runs each user's personal model
locally, then broadcasts only the resulting cursor state.

## Screen Controls

- `Full screen` enters or exits fullscreen.
- `Hide buttons` keeps the room code visible and hides the larger HUD controls.
- `F` toggles fullscreen.
- `H` toggles hidden controls.

## Chromebook Notes

Chromebooks should use the browser alpha, not the macOS app. Use Chrome, keep Tailscale
connected, and open the HTTPS URL. If camera gaze is unreliable, open Connection, enable
Mouse mode, and join the same room to validate networking.

## Troubleshooting

`Camera access needs HTTPS or localhost`

Open the `https://...ts.net` Tailscale Serve URL for remote tests. Plain HTTP works for
local testing only on `localhost` or `127.0.0.1`.

`Could not connect to relay`

Make sure the relay process is running, then check `tailscale serve status` on the relay
machine.

The cursor is inaccurate

If the status says `Model`, the browser is running the exported checkpoint. Accuracy can
still vary on non-MacBook cameras because the original model was trained against a
specific MacBook camera/screen geometry. Run `Calibrate` in fullscreen any time the page
feels off. If the status says `Fallback`, the ONNX model was not available and the
browser fell back to the heuristic predictor.
