# Relay Operations

The relay is a small WebSocket process that receives cursor updates and broadcasts them
to peers in the same room. It also serves the browser client from `web/` by default.

## Local Development

Run a local relay and static browser host:

```bash
python relay_server.py --host 127.0.0.1 --port 8765
```

Open the browser client:

```text
http://127.0.0.1:8765
```

Join a room in the browser, then click `Calibrate` and keep the page fullscreen during the
five targets.

For a local multi-user smoke test, open a second tab, join the same room, and enable
`Mouse mode` under `Connection`.

The room itself is a hangout. Dojo, Trial, and Solo are local runs. Multiplayer starts a
room-visible wave. The relay stores one active wave per room, broadcasts `wave_start` to
everyone in that room, includes the active wave in `welcome` for late joiners, and
broadcasts sanitized `wave_score` updates after clients send `wave_hit`.

## Private Alpha Deployment

For early remote tests, host the relay on a machine reachable by all testers through a
private network or VPN.

For browser tests, keep the Python relay bound to localhost and expose it with a private
HTTPS proxy such as Tailscale Serve:

```bash
python -m pip install -r requirements-relay.txt
python relay_server.py --host 127.0.0.1 --port 8765
tailscale serve --bg 8765
```

Testers open the generated `https://...ts.net` URL. Browser camera access requires HTTPS
for remote pages. The relay machine must have the ignored
`web/models/*.onnx` assets needed for browser model inference. Generate and verify those
assets on a development machine with `requirements-dev.txt`, then sync them to the relay.

For the current Prometheus LAN relay:

```bash
ssh prometheus@prometheus
cd ~/gaze-game-relay
systemctl --user status gaze-game-relay.service
systemctl --user restart gaze-game-relay.service
tailscale serve status
```

The relay should remain fronted by HTTPS:

```text
https://prometheus.tailcf7f8f.ts.net/
```

## Production Requirements

The included relay is intentionally minimal. Before exposing it on the public internet,
add:

- TLS, usually by terminating `wss://` at a reverse proxy or hosting platform
- room-code authorization
- message-size limits
- per-client rate limits
- max clients per room
- idle timeout cleanup
- structured logs and basic metrics
- authoritative wave scoring or explicit anti-cheat boundaries
- abuse reporting or operator controls
- explicit data-collection consent before accepting webcam-derived training data

Users should not expose a relay port from their own Mac for normal use.
