# Gaze Game Alpha Tester Guide

This guide is for someone receiving a prebuilt Gaze Game zip. It does not require
Terminal or Python.

## What You Need

- An Apple Silicon MacBook
- A webcam that works in other apps
- Tailscale, if the test organizer says the relay is on a private network
- The latest `Gaze-Game-alpha-macos-arm64.zip`

## Install Tailscale

Skip this section if the test organizer says Tailscale is not needed.

1. Open the official Mac download page:
   [tailscale.com/download/mac](https://tailscale.com/download/mac)
2. Install Tailscale.
3. Sign in, or accept the invite sent by the test organizer.
4. Wait until Tailscale says it is connected.
5. Leave Tailscale running while using Gaze Game.

Tailscale also has official Mac install docs:
[tailscale.com/kb/1016/install-mac](https://tailscale.com/kb/1016/install-mac).

## Open The App

1. Unzip `Gaze-Game-alpha-macos-arm64.zip`.
2. Open the unzipped folder.
3. Keep `relay_urls.txt` next to `Gaze Game.app` if that file is included.
4. Right-click `Gaze Game.app`.
5. Click Open.
6. If macOS asks whether you are sure, click Open again.
7. When macOS asks for Camera access, click Allow.

## Create A Room

Use this if you are starting the session.

1. Type your name.
2. Click Create Room.
3. Send the room code to the other person.

## Join A Room

Use this if someone sent you a room code.

1. Type your name.
2. Type the room code.
3. Click Join Room.

## What You Should See

You should see a dark grid. Your gaze cursor and the other person's cursor should both
appear in the room.

## Troubleshooting

`Relay unavailable`

The app cannot reach the relay. Make sure Tailscale is connected, and make sure
`relay_urls.txt` is still next to `Gaze Game.app` if the organizer included one.

`Camera unavailable`

Open System Settings > Privacy & Security > Camera, enable Gaze Game, then reopen the
app.

The app will not open

Right-click `Gaze Game.app` and choose Open. Do not double-click the first time.

The cursor is inaccurate

This can happen in early testing if the gaze model was not trained on your MacBook and
your eyes. The networking test is still valid if your cursor appears and moves.
