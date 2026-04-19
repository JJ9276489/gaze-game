# Relay Operations

The relay is a small WebSocket process that receives cursor updates and broadcasts them
to peers in the same room.

## Local Development

Run a local relay:

```bash
python relay_server.py --host 127.0.0.1 --port 8765
```

Run a client against it:

```bash
python gaze_game.py --server ws://127.0.0.1:8765 --room TEST-01 --name Player
```

Run a second local client with the mouse:

```bash
python gaze_game.py --server ws://127.0.0.1:8765 --room TEST-01 --name Mouse --mouse --windowed
```

## Private Alpha Deployment

For early remote tests, host the relay on a machine reachable by all testers through a
private network or VPN. Configure clients with:

```bash
GAZE_GAME_RELAY_URL=ws://relay-host.example:8765 python gaze_game.py
```

Or pass a relay explicitly:

```bash
python gaze_game.py --server ws://relay-host.example:8765
```

For packaged alpha builds, put relay URLs in `relay_urls.local.txt` before running the
macOS build script. The script copies that ignored local file into the alpha zip as
`relay_urls.txt`, next to `Gaze Game.app`, so testers do not need Terminal.

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
- abuse reporting or operator controls

Users should not expose a relay port from their own Mac for normal use.
