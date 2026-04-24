# Browser Alpha

The browser app is the primary product path. Testers join from Chrome, Edge, Safari, or a
Chromebook without installing a native app. It uses the webcam through the browser,
estimates gaze locally, calibrates against the current viewport, and sends only cursor
coordinates to the relay.

## What Works

- MacBook browsers
- Chromebook browsers
- local two-client tests from one computer
- remote private tests through Tailscale
- ONNX-exported gaze checkpoint inference
- five-point fullscreen calibration with saved local browser state
- heuristic fallback if the ONNX model asset is missing
- mouse mode when camera gaze is unavailable

## Model Asset

The browser client loads:

```text
web/models/vision_gaze_spatial_geom.onnx
```

Generate it from the local PyTorch checkpoint before deploying the browser alpha:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/export_browser_onnx.py
python scripts/verify_browser_onnx.py
```

The ONNX file is ignored by git because it is derived from the local checkpoint. Put it on
the private relay machine with the web client, but do not commit it to the public repo.

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
5. Open the page in a second tab or another browser.
6. Enter a different name.
7. Enter the same room code.
8. Open Connection and enable Mouse mode.
9. Click Join room.

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
