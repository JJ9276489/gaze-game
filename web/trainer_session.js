import {
  CHALLENGE_DURATION_MS,
  WAVE_TARGET_COUNT,
  createWaveSeed,
  isWaveMode,
  makeEnemyTargets,
  normalizedWaveTargets,
} from "./game_logic.js";
import { createTrainingSample, distancePx } from "./personal_model.js";

export const TRAIN_TARGETS = Object.freeze([
  { id: "train-01", x: 0.15, y: 0.16 },
  { id: "train-02", x: 0.36, y: 0.16 },
  { id: "train-03", x: 0.64, y: 0.16 },
  { id: "train-04", x: 0.85, y: 0.16 },
  { id: "train-05", x: 0.22, y: 0.34 },
  { id: "train-06", x: 0.48, y: 0.34 },
  { id: "train-07", x: 0.73, y: 0.34 },
  { id: "train-08", x: 0.15, y: 0.52 },
  { id: "train-09", x: 0.36, y: 0.52 },
  { id: "train-10", x: 0.64, y: 0.52 },
  { id: "train-11", x: 0.85, y: 0.52 },
  { id: "train-12", x: 0.27, y: 0.68 },
  { id: "train-13", x: 0.52, y: 0.68 },
  { id: "train-14", x: 0.78, y: 0.68 },
  { id: "train-15", x: 0.15, y: 0.78 },
  { id: "train-16", x: 0.36, y: 0.78 },
  { id: "train-17", x: 0.64, y: 0.78 },
  { id: "train-18", x: 0.85, y: 0.78 },
]);

export const TRAIN_SETTLE_MS = 360;
export const TRAIN_CAPTURE_MS = 640;
export const TRAIN_MIN_SAMPLES_PER_TARGET = 6;
export const CHALLENGE_TARGET_RADIUS_PX = 34;
export const CHALLENGE_DWELL_MS = 240;

const UI_REFRESH_MS = 150;

export function createTrainerSession({
  mode,
  kind,
  wave = null,
  now = nowMs(),
  wallNow = Date.now(),
  random = Math.random,
  waveSeedFactory = createWaveSeed,
} = {}) {
  const waveMode = isWaveMode(mode);
  const seed = wave?.seed || (waveMode ? waveSeedFactory() : "");
  const durationMs = Number(wave?.durationMs) || CHALLENGE_DURATION_MS;
  const remainingMs =
    waveMode && Number.isFinite(wave?.startedAt)
      ? Math.max(1000, wave.startedAt + durationMs - wallNow)
      : durationMs;
  const waveTargets = waveMode ? normalizedWaveTargets(wave?.targets) : [];
  const targets = waveMode
    ? waveTargets.length
      ? waveTargets
      : makeEnemyTargets(seed, WAVE_TARGET_COUNT)
    : shuffledTargets(TRAIN_TARGETS, random);

  return {
    active: true,
    mode,
    kind,
    waveId: wave?.id || `${mode || "run"}-${Math.floor(wallNow)}`,
    seed,
    targets,
    stepIndex: 0,
    phase: waveMode ? "active" : "settle",
    phaseStartedAt: now,
    startedAt: now,
    endsAt: waveMode ? now + remainingMs : 0,
    durationMs,
    samples: [],
    capturedSamples: [],
    score: 0,
    dwellStartedAt: 0,
    lastErrorPx: null,
    lastUiAt: 0,
  };
}

export function trainerWaveScoreMap(wave) {
  const scores = new Map();
  for (const score of wave?.scores || []) {
    if (score?.id) {
      scores.set(score.id, score);
    }
  }
  return scores;
}

export function advanceTrainerSession(
  session,
  {
    reading,
    activePoint,
    now = nowMs(),
    viewportWidth = 1,
    viewportHeight = 1,
    targetMapper = identityTarget,
  } = {},
) {
  if (!session?.active) {
    return { type: "inactive", refresh: false, complete: false };
  }
  if (reading?.kind !== session.kind) {
    return {
      type: "kind_mismatch",
      refresh: false,
      complete: false,
      message: "Model changed. Start a new run.",
    };
  }
  if (isWaveMode(session.mode)) {
    return advanceWaveSession(session, {
      activePoint,
      now,
      viewportWidth,
      viewportHeight,
      targetMapper,
    });
  }
  return advanceDojoSession(session, {
    reading,
    now,
    viewportWidth,
    viewportHeight,
    targetMapper,
  });
}

export function currentTrainerTarget(session, targetMapper = identityTarget) {
  if (!session?.targets?.length) {
    return null;
  }
  const target = isWaveMode(session.mode)
    ? session.targets[session.stepIndex % session.targets.length]
    : session.targets[session.stepIndex];
  return target ? targetMapper(target) : null;
}

export function trainerOverlayView(session, { now = nowMs() } = {}) {
  if (!session?.active) {
    return {
      hidden: true,
      progressPercent: 0,
      step: "",
      title: "",
      body: "",
    };
  }

  if (isWaveMode(session.mode)) {
    const secondsLeft = Math.max(0, Math.ceil((session.endsAt - now) / 1000));
    const progress = clamp01((now - session.startedAt) / session.durationMs);
    return {
      hidden: false,
      progressPercent: Math.round(progress * 100),
      step: `${trainerModeLabel(session.mode)} · ${secondsLeft}s`,
      title: `${session.score} takedowns`,
      body: session.lastErrorPx === null ? "Acquire enemies." : `${Math.round(session.lastErrorPx)} px`,
    };
  }

  const total = session.targets.length;
  const targetProgress = session.stepIndex / total;
  const phaseProgress =
    session.phase === "capture"
      ? Math.min(1, (now - session.phaseStartedAt) / TRAIN_CAPTURE_MS)
      : 0;
  const progress = targetProgress + phaseProgress / total;
  const targetNoun = trainerTargetNoun(session.mode);
  return {
    hidden: false,
    progressPercent: Math.round(progress * 100),
    step: `${trainerModeLabel(session.mode)} ${session.stepIndex + 1} of ${total}`,
    title: session.phase === "capture" ? `Hold ${targetNoun}` : `Acquire ${targetNoun}`,
    body: `${session.capturedSamples.length + session.samples.length} samples`,
  };
}

export function trainerModeLabel(mode) {
  if (mode === "dojo") return "Dojo";
  if (mode === "solo") return "Solo";
  if (mode === "multiplayer") return "Multiplayer wave";
  return "Run";
}

export function trainerTargetNoun(mode) {
  if (mode === "dojo") return "dummy";
  return "enemy";
}

function advanceDojoSession(
  session,
  { reading, now, viewportWidth, viewportHeight, targetMapper = identityTarget },
) {
  const target = currentTrainerTarget(session, targetMapper);
  if (!target) {
    return { type: "complete", refresh: true, complete: true };
  }

  if (session.phase === "settle") {
    if (now - session.phaseStartedAt >= TRAIN_SETTLE_MS) {
      session.phase = "capture";
      session.phaseStartedAt = now;
      session.samples = [];
      return { type: "phase_change", refresh: true, complete: false };
    }
    return { type: "settling", refresh: false, complete: false };
  }

  const sample = createTrainingSample(reading, target, {
    width: Math.max(1, viewportWidth),
    height: Math.max(1, viewportHeight),
  });
  if (sample) {
    session.samples.push(sample);
  }

  let refresh = markUiRefresh(session, now);
  if (
    now - session.phaseStartedAt < TRAIN_CAPTURE_MS ||
    session.samples.length < TRAIN_MIN_SAMPLES_PER_TARGET
  ) {
    return { type: "capturing", refresh, complete: false };
  }

  session.capturedSamples.push(...session.samples);
  session.stepIndex += 1;
  if (session.stepIndex >= session.targets.length) {
    return { type: "complete", refresh: true, complete: true };
  }

  session.phase = "settle";
  session.phaseStartedAt = now;
  session.samples = [];
  refresh = true;
  return { type: "target_advanced", refresh, complete: false };
}

function advanceWaveSession(
  session,
  { activePoint, now, viewportWidth, viewportHeight, targetMapper = identityTarget },
) {
  if (now >= session.endsAt) {
    return { type: "complete", refresh: true, complete: true };
  }

  const target = currentTrainerTarget(session, targetMapper);
  if (!target) {
    return { type: "complete", refresh: true, complete: true };
  }

  const point = activePoint || { x: 0, y: 0 };
  const errorPx = distancePx(
    point.x,
    point.y,
    target.x,
    target.y,
    Math.max(1, viewportWidth),
    Math.max(1, viewportHeight),
  );
  session.lastErrorPx = errorPx;

  if (errorPx <= CHALLENGE_TARGET_RADIUS_PX) {
    if (!session.dwellStartedAt) {
      session.dwellStartedAt = now;
    }
    if (now - session.dwellStartedAt >= CHALLENGE_DWELL_MS) {
      session.score += 1;
      session.stepIndex += 1;
      session.dwellStartedAt = 0;
      session.phaseStartedAt = now;
      markUiRefresh(session, now);
      return {
        type: "wave_hit",
        refresh: true,
        complete: false,
        hitTarget: target,
      };
    }
  } else {
    session.dwellStartedAt = 0;
  }

  return { type: "wave_active", refresh: markUiRefresh(session, now), complete: false };
}

function shuffledTargets(targets, random) {
  const copy = targets.map((target) => ({ ...target }));
  shuffleInPlace(copy, random);
  return copy;
}

function shuffleInPlace(items, random) {
  for (let index = items.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [items[index], items[swapIndex]] = [items[swapIndex], items[index]];
  }
  return items;
}

function markUiRefresh(session, now) {
  if (now - session.lastUiAt <= UI_REFRESH_MS) {
    return false;
  }
  session.lastUiAt = now;
  return true;
}

function identityTarget(target) {
  return target;
}

function nowMs() {
  return globalThis.performance?.now?.() ?? Date.now();
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}
