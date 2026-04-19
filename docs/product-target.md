# Product Target

The end-user version should not require Terminal, Python, or self-hosting.

## User Flow

1. User installs `Gaze Game.app`.
2. On first launch, macOS asks for Camera permission.
3. User enters a display name.
4. User chooses `Create Room` or `Join Room`.
5. `Create Room` shows a short code like `K7M-4QX`.
6. Another user enters that room code.
7. Both Macs connect to the hosted relay over `wss://`.
8. Each Mac runs gaze inference locally and sends only cursor coordinates.

For local development, the default relay is:

```text
ws://127.0.0.1:8765
```

Private alpha builds should configure their own reachable relay URL with
`GAZE_GAME_RELAY_URL`, `GAZE_GAME_RELAY_URLS`, or `--server`.

## Camera Permission

Camera permission must belong to the shipped app bundle. A packaged macOS app needs an
`NSCameraUsageDescription` entry in `Info.plist`. Running from Codex or Terminal is a
developer-only path and may attribute camera permission to the parent app instead of the
final product.

The app copy in this repo uses:

```text
Gaze Game uses your camera locally to estimate gaze. Video, eye crops, and face
landmarks never leave your Mac.
```

## Relay

End users should connect to a hosted relay URL such as:

```text
wss://relay.example.com
```

Private-network relays are acceptable for alpha testing, not consumer distribution.

The relay should enforce:

- room-code authorization
- message-size limits
- per-client rate limits
- max clients per room
- idle timeout cleanup
- TLS termination through the hosting platform or reverse proxy

Users should not expose a relay port from their own Mac for normal use.

## Data Sent Over The Network

Allowed:

- room code
- display name
- cursor color
- normalized cursor coordinates
- tracking status
- timestamps and sequence counters

Not allowed:

- webcam frames
- eye crops
- face landmarks
- MediaPipe output
- model tensors
- checkpoints
