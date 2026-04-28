import assert from "node:assert/strict";
import test from "node:test";

import {
  buildCursorMessage,
  buildWaveHitMessage,
  buildWaveStartMessage,
  connectRelaySocket,
  defaultRelayUrl,
  normalizeRelayUrl,
  parseRelayMessage,
  relayIsOpen,
  sendRelayMessage,
} from "./relay_client.js";

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 3;
  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = FakeWebSocket.CONNECTING;
    this.sent = [];
    this.listeners = new Map();
    FakeWebSocket.instances.push(this);
  }

  addEventListener(type, callback) {
    const listeners = this.listeners.get(type) || [];
    listeners.push(callback);
    this.listeners.set(type, listeners);
  }

  send(data) {
    this.sent.push(data);
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED;
    this.emit("close", {});
  }

  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.emit("open", {});
  }

  message(payload) {
    this.emit("message", {
      data: typeof payload === "string" ? payload : JSON.stringify(payload),
    });
  }

  emit(type, event) {
    for (const listener of this.listeners.get(type) || []) {
      listener(event);
    }
  }
}

function noTimer() {
  return 1;
}

test("relay URL helpers normalize common browser and manual inputs", () => {
  assert.equal(defaultRelayUrl({ protocol: "https:", host: "game.example" }), "wss://game.example/ws");
  assert.equal(defaultRelayUrl({ protocol: "http:", host: "127.0.0.1:8765" }), "ws://127.0.0.1:8765/ws");
  assert.equal(defaultRelayUrl({ protocol: "file:", host: "" }), "ws://127.0.0.1:8765/ws");

  assert.equal(normalizeRelayUrl("https://game.example/ws"), "wss://game.example/ws");
  assert.equal(normalizeRelayUrl("http://127.0.0.1:8765/ws"), "ws://127.0.0.1:8765/ws");
  assert.equal(normalizeRelayUrl("wss://relay.example/ws"), "wss://relay.example/ws");
  assert.equal(normalizeRelayUrl("relay.example/ws"), "ws://relay.example/ws");
});

test("relay payload builders use the server wire format", () => {
  assert.deepEqual(
    buildCursorMessage({
      room: "ABC-123",
      x: 0.1,
      y: 0.2,
      tracking: true,
      seq: 7,
      timestamp: 1000,
    }),
    {
      type: "cursor",
      room: "ABC-123",
      x: 0.1,
      y: 0.2,
      tracking: true,
      seq: 7,
      ts: 1000,
    },
  );
  assert.deepEqual(
    buildWaveStartMessage({
      room: "ABC-123",
      seed: "seed",
      durationMs: 30000,
      targets: [{ id: "t1", x: 0.1, y: 0.2 }],
      timestamp: 2000,
    }),
    {
      type: "wave_start",
      room: "ABC-123",
      seed: "seed",
      duration_ms: 30000,
      targets: [{ id: "t1", x: 0.1, y: 0.2 }],
      ts: 2000,
    },
  );
  assert.deepEqual(
    buildWaveHitMessage({
      room: "ABC-123",
      waveId: "wave",
      targetId: "t1",
      score: 5,
      timestamp: 3000,
    }),
    {
      type: "wave_hit",
      room: "ABC-123",
      wave_id: "wave",
      target_id: "t1",
      score: 5,
      ts: 3000,
    },
  );
});

test("relay message parsing and send guard tolerate bad websocket state", () => {
  assert.deepEqual(parseRelayMessage('{"type":"welcome"}'), { type: "welcome" });
  assert.equal(parseRelayMessage("not-json"), null);
  assert.equal(parseRelayMessage("[1,2]"), null);

  const ws = new FakeWebSocket("ws://relay");
  assert.equal(relayIsOpen(ws, FakeWebSocket.OPEN), false);
  assert.equal(sendRelayMessage(ws, { type: "cursor" }), false);
  ws.open();
  assert.equal(relayIsOpen(ws, FakeWebSocket.OPEN), true);
  assert.equal(sendRelayMessage(ws, { type: "cursor" }), true);
  assert.equal(ws.sent.at(-1), '{"type":"cursor"}');
});

test("relay connection sends join and resolves on welcome", async () => {
  FakeWebSocket.instances = [];
  let welcome = null;
  const promise = connectRelaySocket({
    url: "ws://relay/ws",
    room: "ABC-123",
    name: "Ada",
    color: [1, 2, 3],
    WebSocketCtor: FakeWebSocket,
    setTimeoutFn: noTimer,
    clearTimeoutFn: () => {},
    onWelcome(message) {
      welcome = message;
    },
  });
  const ws = FakeWebSocket.instances.at(-1);

  ws.open();
  assert.deepEqual(JSON.parse(ws.sent[0]), {
    type: "join",
    room: "ABC-123",
    name: "Ada",
    color: [1, 2, 3],
  });
  ws.message({ type: "welcome", id: "client-a", peers: [] });

  assert.equal(await promise, ws);
  assert.equal(welcome.id, "client-a");
});

test("relay connection rejects explicit join errors and reports later messages", async () => {
  FakeWebSocket.instances = [];
  await assert.rejects(
    async () => {
      const promise = connectRelaySocket({
        url: "ws://relay/ws",
        room: "ABC-123",
        name: "Ada",
        color: [1, 2, 3],
        WebSocketCtor: FakeWebSocket,
        setTimeoutFn: noTimer,
        clearTimeoutFn: () => {},
      });
      const ws = FakeWebSocket.instances.at(-1);
      ws.open();
      ws.message({ type: "error", message: "room_full" });
      await promise;
    },
    /room_full/,
  );

  let laterMessage = null;
  const promise = connectRelaySocket({
    url: "ws://relay/ws",
    room: "ABC-123",
    name: "Ada",
    color: [1, 2, 3],
    WebSocketCtor: FakeWebSocket,
    setTimeoutFn: noTimer,
    clearTimeoutFn: () => {},
    onMessage(message) {
      laterMessage = message;
    },
  });
  const ws = FakeWebSocket.instances.at(-1);
  ws.open();
  ws.message({ type: "welcome", id: "client-a", peers: [] });
  await promise;
  ws.message({ type: "cursor", id: "peer-a" });

  assert.deepEqual(laterMessage, { type: "cursor", id: "peer-a" });
});
