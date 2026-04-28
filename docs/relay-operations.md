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

The room itself is a hangout. Dojo is a separate local training context, Solo is a local
room run, and Multiplayer starts a room-visible wave. The relay stores one active wave
per room, generates the multiplayer target list, broadcasts `wave_start` to everyone in
that room, includes the active wave in `welcome` for late joiners, and broadcasts
server-incremented `wave_score` updates after validating `wave_hit` against target order
and the client's recent cursor position.

The alpha relay also caps websocket message size, room occupancy, and per-client message
rates. Generated room codes are high-entropy bearer tokens. It is still not a public
matchmaking or anti-cheat server.

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
for remote pages. The relay machine must have both `web/vendor/` runtime assets and the
ignored `web/models/*.onnx` assets needed for browser model inference. Generate, vendor,
and verify those assets on a development machine with `requirements-dev.txt`, then sync
them to the relay.

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
- account or invite authorization beyond room-code bearer tokens
- idle timeout cleanup
- structured logs and basic metrics
- stronger authoritative scoring or explicit anti-cheat boundaries
- abuse reporting or operator controls
- explicit data-collection consent before accepting webcam-derived training data

Users should not expose a relay port from their own Mac for normal use.
