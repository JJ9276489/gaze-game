export const CHALLENGE_DURATION_MS = 30000;
export const WAVE_TARGET_COUNT = 96;

const WAVE_MODES = new Set(["solo", "multiplayer"]);

export function isWaveMode(mode) {
  return WAVE_MODES.has(mode);
}

export function isMultiplayerWaveMode(mode) {
  return mode === "multiplayer";
}

export function createWaveSeed() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export function makeEnemyTargets(seed, count = WAVE_TARGET_COUNT) {
  const random = seededRandom(seed || createWaveSeed());
  const targets = [];
  let previous = null;
  for (let index = 0; index < count; index += 1) {
    let target = null;
    for (let attempt = 0; attempt < 10; attempt += 1) {
      target = {
        id: `enemy-${index}`,
        x: 0.12 + random() * 0.76,
        y: 0.14 + random() * 0.62,
      };
      if (!previous || Math.hypot(target.x - previous.x, target.y - previous.y) > 0.25) {
        break;
      }
    }
    targets.push(target);
    previous = target;
  }
  return targets;
}

export function normalizeRelayWave(message) {
  if (!message || typeof message !== "object") {
    return null;
  }
  const seed = String(message.seed || createWaveSeed());
  const targets = normalizedWaveTargets(message.targets);
  return {
    id: String(message.id || message.wave_id || `wave-${seed}`),
    seed,
    targets: targets.length ? targets : makeEnemyTargets(seed, WAVE_TARGET_COUNT),
    startedAt: Number(message.started_at) || Number(message.server_ts) || Date.now(),
    durationMs: Math.max(1000, Number(message.duration_ms) || CHALLENGE_DURATION_MS),
    startedBy: message.started_by || "",
    startedByName: message.started_by_name || "A player",
    scores: normalizeWaveScores(message.scores),
  };
}

export function normalizedWaveTargets(targets) {
  if (!Array.isArray(targets)) {
    return [];
  }
  return targets
    .map((target, index) => ({
      id: String(target?.id || `enemy-${index}`),
      x: clamp01(target?.x),
      y: clamp01(target?.y),
    }))
    .filter((target) => Number.isFinite(target.x) && Number.isFinite(target.y));
}

export function normalizeWaveScores(scores) {
  if (!Array.isArray(scores)) {
    return [];
  }
  return scores
    .map((score) => ({
      id: String(score?.id || ""),
      name: score?.name || "Guest",
      color: Array.isArray(score?.color) ? score.color : [255, 255, 255],
      score: Math.max(0, Number(score?.score) || 0),
    }))
    .filter((score) => score.id);
}

function seededRandom(seed) {
  let value = hashSeed(seed);
  return () => {
    value += 0x6d2b79f5;
    let next = value;
    next = Math.imul(next ^ (next >>> 15), next | 1);
    next ^= next + Math.imul(next ^ (next >>> 7), next | 61);
    return ((next ^ (next >>> 14)) >>> 0) / 4294967296;
  };
}

function hashSeed(seed) {
  let hash = 2166136261;
  const text = String(seed);
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function clamp01(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return 0;
  }
  return Math.max(0, Math.min(1, number));
}
