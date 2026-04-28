# Browser Alpha

The browser app is the primary product path. The current game skin is Gaze Ninja: the
room is a hangout by default, the Dojo trains the personal model with dummies, and room
play uses Solo or Multiplayer enemy waves as gaze targets.

## What Works

- laptop and desktop browsers with webcam access
- Chromebook browsers
- local two-client tests from one computer
- remote private tests through Tailscale or another private HTTPS front door
- ONNX-exported gaze checkpoint inference
- five-point fullscreen calibration with saved local browser state
- browser-local personal NN adapter training
- live debug metrics and JSON log export for remote tester reports
- timed Solo and Multiplayer target competition
- hangout rooms with opt-in game waves
- synchronized multiplayer enemy waves with room-visible scores
- server-generated multiplayer targets with server-side score increments
- pinned browser runtime assets served from the relay origin
- heuristic fallback if the ONNX model asset is missing
- mouse mode when camera gaze is unavailable

## Runtime And Model Assets

Vendor the browser runtime before local or remote testing:

```bash
source .venv/bin/activate
python scripts/vendor_browser_runtime.py
```

This places pinned MediaPipe, face landmarker, and ONNX Runtime Web assets under
`web/vendor/`. Testers' browsers load those runtime files from the relay origin, not from
public CDNs. `web/vendor/` is generated and ignored by git.

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

The model list and checkpoint paths live in `config/browser_models.json`. ONNX files are
ignored by git because they are derived from local checkpoints. Put the needed model files
on the private relay machine with the web client, but do not commit them to the public
repo.

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
3. Click `Dojo`.
4. Click `Calibrate` and keep staring at each target until it moves.
5. Complete a Dojo dummy run.
6. Click `Leave`, then click `Create room`.
7. Open the page in a second tab or another browser.
8. Enter a different name.
9. Enter the same room code.
10. Open Connection and enable Mouse mode if you only need to validate networking.
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

See [alpha-tester-guide.md](alpha-tester-guide.md) for the tester-facing join and play
steps.

When a remote tester reports bad gaze quality, ask them to click `Debug`, then
`Export log`, and send the downloaded JSON. The log contains runtime/model/cursor timing
state and recent normalized gaze samples, not webcam frames or the actual room code.

## Multi-User Behavior

Rooms share cursor state only. Calibration and Dojo are per-user browser runs. One user
starting `Dojo` does not start training for anyone else, and the app hides that user's
cursor from peers during local training so stale target movement does not distract the
room.

`Solo` runs an enemy wave locally and hides it from peers. `Multiplayer` shares a wave
across the room; see [relay-operations.md](relay-operations.md#local-development) for the
relay-side `wave_start` / `wave_hit` / `wave_score` behavior.

The current relay generates multiplayer targets, rate-limits noisy messages, and ignores
client score totals. It validates each hit against the expected target order and the
client's recent cursor position, then increments the score server-side. This is enough for
private alpha testing, but it is still not a public anti-cheat or matchmaking server.

## Screen Controls

See [alpha-tester-guide.md](alpha-tester-guide.md#screen-controls).

## Chromebook Notes

Chromebooks should use Chrome, keep Tailscale connected, and open the HTTPS URL. If
camera gaze is unreliable, open Connection, enable Mouse mode, and join the same room to
validate networking.

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
