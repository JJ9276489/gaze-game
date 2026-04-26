export const RELAY_JOIN_TIMEOUT_MS = 8000;
export const RELAY_SEND_INTERVAL_MS = 33;

export function defaultRelayUrl(locationLike = globalThis.location) {
  if (locationLike?.protocol === "https:") {
    return `wss://${locationLike.host}/ws`;
  }
  if (locationLike?.protocol === "http:") {
    return `ws://${locationLike.host}/ws`;
  }
  return "ws://127.0.0.1:8765/ws";
}

export function normalizeRelayUrl(value, locationLike = globalThis.location) {
  const raw = (value || defaultRelayUrl(locationLike)).trim();
  if (raw.startsWith("https://")) {
    return `wss://${raw.slice("https://".length)}`;
  }
  if (raw.startsWith("http://")) {
    return `ws://${raw.slice("http://".length)}`;
  }
  if (raw.startsWith("ws://") || raw.startsWith("wss://")) {
    return raw;
  }
  return `ws://${raw}`;
}

export function relayIsOpen(ws, openState = globalThis.WebSocket?.OPEN ?? 1) {
  return Boolean(ws && ws.readyState === openState);
}

export function parseRelayMessage(data) {
  try {
    const message = JSON.parse(data);
    return message && typeof message === "object" && !Array.isArray(message) ? message : null;
  } catch {
    return null;
  }
}

export function buildJoinMessage({ room, name, color }) {
  return {
    type: "join",
    room,
    name,
    color,
  };
}

export function buildCursorMessage({ room, x, y, tracking, seq, timestamp = Date.now() }) {
  return {
    type: "cursor",
    room,
    x,
    y,
    tracking,
    seq,
    ts: timestamp,
  };
}

export function buildWaveStartMessage({
  room,
  seed,
  durationMs,
  targets,
  timestamp = Date.now(),
}) {
  return {
    type: "wave_start",
    room,
    seed,
    duration_ms: durationMs,
    targets,
    ts: timestamp,
  };
}

export function buildWaveHitMessage({ room, waveId, targetId, score, timestamp = Date.now() }) {
  return {
    type: "wave_hit",
    room,
    wave_id: waveId,
    target_id: targetId || "",
    score,
    ts: timestamp,
  };
}

export function sendRelayMessage(ws, message) {
  if (!relayIsOpen(ws)) {
    return false;
  }
  ws.send(JSON.stringify(message));
  return true;
}

export function connectRelaySocket({
  url,
  room,
  name,
  color,
  timeoutMs = RELAY_JOIN_TIMEOUT_MS,
  WebSocketCtor = globalThis.WebSocket,
  setTimeoutFn = globalThis.setTimeout,
  clearTimeoutFn = globalThis.clearTimeout,
  onWelcome = () => {},
  onMessage = () => {},
  onDisconnect = () => {},
}) {
  if (!WebSocketCtor) {
    return Promise.reject(new Error("WebSocket is not available."));
  }

  return new Promise((resolve, reject) => {
    const ws = new WebSocketCtor(url);
    let settled = false;
    let suppressDisconnect = false;
    const timeout = setTimeoutFn(() => {
      if (!settled) {
        settled = true;
        suppressDisconnect = true;
        ws.close();
        reject(new Error("Relay connection timed out."));
      }
    }, timeoutMs);

    ws.addEventListener("open", () => {
      ws.send(JSON.stringify(buildJoinMessage({ room, name, color })));
    });

    ws.addEventListener("message", (event) => {
      const message = parseRelayMessage(event.data);
      if (!message) {
        return;
      }

      if (message.type === "error" && !settled) {
        settled = true;
        suppressDisconnect = true;
        clearTimeoutFn(timeout);
        ws.close();
        reject(new Error(`Relay rejected join: ${message.message || "unknown_error"}`));
        return;
      }

      if (message.type === "welcome") {
        if (!settled) {
          settled = true;
          clearTimeoutFn(timeout);
          onWelcome(message);
          resolve(ws);
        }
        return;
      }

      onMessage(message);
    });

    ws.addEventListener("error", () => {
      if (!settled) {
        settled = true;
        clearTimeoutFn(timeout);
        reject(new Error(`Could not connect to ${url}`));
      }
    });

    ws.addEventListener("close", () => {
      if (!settled) {
        settled = true;
        clearTimeoutFn(timeout);
        reject(new Error("Relay closed before joining."));
        return;
      }
      if (!suppressDisconnect) {
        onDisconnect();
      }
    });
  });
}
