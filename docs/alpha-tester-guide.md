# Gaze Ninja Alpha Tester Guide

This guide is for browser testers. You should not need Terminal, Python, or a native app.

## What You Need

- A laptop or desktop with a working webcam.
- Chrome, Edge, Safari, or another modern browser.
- Tailscale only if the organizer says the test URL is private to Tailscale.
- The HTTPS test URL from the organizer.

## Join The Test

1. Open the HTTPS URL from the organizer.
2. Enter your name.
3. Click `Dojo` if you want to train before joining friends, or click `Create room` /
   enter a room code and click `Join room`.
4. Allow camera access when the browser asks.
5. Click `Calibrate`.
6. Use the fullscreen prompt or click `Full screen`.
7. Look at each target until it moves.

After calibration, you should be in the room hangout. If another tester joins the same
room, you should see their cursor too.

## Train And Play

1. Click `Dojo`.
2. Defeat the training dummies by looking at them until the run finishes.
3. Join or create a room.
4. Click `Solo` to play your own enemy wave.
5. Click `Multiplayer` to start a synchronized room wave.

Dojo training is local to your browser. Other people in the room do not receive your
webcam data, training samples, or personal model.

## Screen Controls

- `Full screen` enters or exits fullscreen.
- `Hide buttons` removes most controls while keeping the room code visible.
- `Debug` shows model, cursor, camera, and latency metrics.
- `Export log` downloads a JSON report you can send to the organizer.
- Press `F` to toggle fullscreen.
- Press `H` to hide or show controls.
- Press `D` to show or hide debug metrics.

During calibration, training, testing, and competition, the app keeps the screen minimal
so the UI does not block targets.

The debug export does not include webcam frames, eye crops, face landmarks, or the actual
room code. It includes browser/device status, model status, cursor coordinates, timing,
and recent gaze samples.

## Troubleshooting

`Camera access needs HTTPS or localhost`

Use the organizer's `https://...` URL. Plain HTTP camera access only works for local
developer tests on `localhost` or `127.0.0.1`.

`Could not connect to relay`

The relay host may be down, or Tailscale may not be connected. Tell the organizer and
include the room code you tried.

The cursor is inaccurate

Re-enter fullscreen and run `Calibrate` again. Then collect a fresh `Dojo` run before
playing another room wave.

The model says `Fallback`

The browser could not load the ONNX gaze model from the relay. The room can still be
tested with mouse mode, but gaze quality will not represent the real model.
