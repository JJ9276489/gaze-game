# Product Target

The end-user version should not require Terminal, Python, native app installation, or
self-hosting.

## User Flow

1. User opens a hosted HTTPS URL.
2. The browser asks for Camera permission.
3. User enters a display name.
4. User chooses `Create Room` or `Join Room`.
5. `Create Room` shows a short code like `K7M-4QX`.
6. Another user enters that room code.
7. User clicks `Calibrate` and looks at five fullscreen targets.
8. Both clients connect to the hosted relay over `wss://`.
9. Each browser runs gaze inference locally with the ONNX-exported model and sends only
   cursor coordinates.

For local development, the default relay is:

```text
ws://127.0.0.1:8765
```

The packaged macOS app is still useful as a benchmark against the original PyTorch
runtime, but it is not the primary user path.

## Camera Permission

For the browser version, Camera permission belongs to the HTTPS origin. Remote browser
tests must use HTTPS; local tests can use `localhost`.

For the packaged macOS app, Camera permission must belong to the shipped app bundle. A
packaged macOS app needs an `NSCameraUsageDescription` entry in `Info.plist`. Running
from Codex or Terminal is a developer-only path and may attribute camera permission to
the parent app instead of the final product.

The macOS app copy in this repo uses:

```text
Gaze Game uses your camera locally to estimate gaze. Video, eye crops, and face
landmarks never leave your Mac.
```

The browser client should eventually show an equivalent line that swaps "your Mac" for
"your device" near the camera prompt.

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
