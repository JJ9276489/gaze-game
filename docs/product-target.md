# Product Target

The end-user version should not require Terminal, Python, native app installation, or
self-hosting.

The current product skin is Gaze Ninja: train alone in the Dojo first, then bring that
personal model into a room hangout for Solo or Multiplayer enemy waves.

## User Flow

1. User opens a hosted HTTPS URL.
2. The browser asks for Camera permission.
3. User enters a display name.
4. User chooses `Dojo`, `Join room`, or `Create room`.
5. `Dojo` starts a local camera session for calibration, sample collection, and fitting a
   personal adapter without joining the relay.
6. `Create room` shows a short code like `K7M-4QX`.
7. Another user enters that room code and clicks `Join room`.
8. Users can click `Solo` for local waves or `Multiplayer` for synchronized room-visible
   waves.
9. Room clients connect to the hosted relay over `wss://`.
10. Each browser runs gaze inference locally with the ONNX-exported model and sends only
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

## Training Model

The current browser product trains a small personal NN adapter in the Dojo on top of the
selected base ONNX model:

```text
base gaze model -> raw gaze/features -> personal adapter -> corrected cursor
```

This is intentionally local-first. Training samples and fitted weights stay in browser
storage unless the user explicitly exports or uploads them in a future data-collection
flow.

Future full-model training should be staged behind explicit consent and separate data
collection controls because it requires storing eye crops or derived tensors, not just
cursor coordinates.

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
- multiplayer wave seed, target coordinates, and score updates
- timestamps and sequence counters

Not allowed:

- webcam frames
- eye crops
- face landmarks
- MediaPipe output
- model tensors
- checkpoints
