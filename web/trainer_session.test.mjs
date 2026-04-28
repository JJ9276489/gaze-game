import assert from "node:assert/strict";
import test from "node:test";

import { PERSONAL_FEATURE_KEYS } from "./personal_model.js";
import {
  CHALLENGE_DWELL_MS,
  TRAIN_CAPTURE_MS,
  TRAIN_MIN_SAMPLES_PER_TARGET,
  TRAIN_SETTLE_MS,
  TRAIN_TARGETS,
  advanceTrainerSession,
  createTrainerSession,
  currentTrainerTarget,
  trainerOverlayView,
  trainerWaveScoreMap,
} from "./trainer_session.js";

function reading(kind = "spatial_geom") {
  return {
    kind,
    rawX: 0.42,
    rawY: 0.38,
    features: new Array(PERSONAL_FEATURE_KEYS.length).fill(0.5),
  };
}

test("createTrainerSession initializes dojo and wave runs", () => {
  const dojo = createTrainerSession({
    mode: "dojo",
    kind: "spatial_geom",
    now: 100,
    wallNow: 2000,
    random: () => 0,
  });

  assert.equal(dojo.active, true);
  assert.equal(dojo.phase, "settle");
  assert.equal(dojo.waveId, "dojo-2000");
  assert.equal(dojo.targets.length, TRAIN_TARGETS.length);
  assert.notEqual(dojo.targets[0], TRAIN_TARGETS[0]);

  const wave = createTrainerSession({
    mode: "solo",
    kind: "spatial_geom",
    wave: {
      id: "wave-1",
      seed: "seed-1",
      targets: [{ id: "enemy-a", x: 0.2, y: 0.3 }],
      durationMs: 3000,
      startedAt: 10000,
    },
    now: 50,
    wallNow: 11000,
  });

  assert.equal(wave.phase, "active");
  assert.equal(wave.waveId, "wave-1");
  assert.equal(wave.seed, "seed-1");
  assert.equal(wave.endsAt, 2050);
  assert.deepEqual(wave.targets, [{ id: "enemy-a", x: 0.2, y: 0.3 }]);
});

test("advanceTrainerSession captures dojo samples through settle and capture phases", () => {
  const session = createTrainerSession({
    mode: "dojo",
    kind: "spatial_geom",
    now: 0,
    wallNow: 0,
    random: () => 0.5,
  });
  session.targets = [{ id: "only", x: 0.25, y: 0.4 }];

  assert.deepEqual(
    advanceTrainerSession(session, {
      reading: reading(),
      now: TRAIN_SETTLE_MS - 1,
      viewportWidth: 1200,
      viewportHeight: 800,
    }),
    { type: "settling", refresh: false, complete: false },
  );

  assert.deepEqual(
    advanceTrainerSession(session, {
      reading: reading(),
      now: TRAIN_SETTLE_MS,
      viewportWidth: 1200,
      viewportHeight: 800,
    }),
    { type: "phase_change", refresh: true, complete: false },
  );
  assert.equal(session.phase, "capture");

  let result = null;
  for (let index = 0; index < TRAIN_MIN_SAMPLES_PER_TARGET; index += 1) {
    result = advanceTrainerSession(session, {
      reading: reading(),
      now: TRAIN_SETTLE_MS + TRAIN_CAPTURE_MS + index,
      viewportWidth: 1200,
      viewportHeight: 800,
    });
  }

  assert.equal(result.complete, true);
  assert.equal(session.capturedSamples.length, TRAIN_MIN_SAMPLES_PER_TARGET);
  assert.equal(session.capturedSamples[0].targetX, 0.25);
});

test("advanceTrainerSession scores wave hits after dwell time", () => {
  const session = createTrainerSession({
    mode: "multiplayer",
    kind: "spatial_geom",
    wave: {
      id: "wave-2",
      seed: "seed-2",
      targets: [{ id: "enemy-a", x: 0.5, y: 0.5 }],
      durationMs: 5000,
      startedAt: 10000,
    },
    now: 100,
    wallNow: 10000,
  });

  const first = advanceTrainerSession(session, {
    reading: reading(),
    activePoint: { x: 0.5, y: 0.5 },
    now: 100,
    viewportWidth: 1000,
    viewportHeight: 800,
  });
  assert.equal(first.type, "wave_active");
  assert.equal(session.score, 0);
  assert.equal(session.dwellStartedAt, 100);

  const hit = advanceTrainerSession(session, {
    reading: reading(),
    activePoint: { x: 0.5, y: 0.5 },
    now: 100 + CHALLENGE_DWELL_MS,
    viewportWidth: 1000,
    viewportHeight: 800,
  });

  assert.equal(hit.type, "wave_hit");
  assert.equal(hit.hitTarget.id, "enemy-a");
  assert.equal(session.score, 1);
  assert.equal(session.stepIndex, 1);
  assert.equal(currentTrainerTarget(session).id, "enemy-a");
});

test("trainer overlay view and wave score map are presentation-only", () => {
  const scores = trainerWaveScoreMap({
    scores: [
      { id: "p1", name: "Ada", color: [1, 2, 3], score: 2 },
      { name: "No id", score: 99 },
    ],
  });

  assert.deepEqual([...scores.keys()], ["p1"]);

  const session = createTrainerSession({
    mode: "solo",
    kind: "spatial_geom",
    wave: {
      id: "wave-3",
      seed: "seed-3",
      targets: [{ id: "enemy-a", x: 0.5, y: 0.5 }],
      durationMs: 10000,
      startedAt: 10000,
    },
    now: 200,
    wallNow: 10000,
  });
  session.score = 4;
  session.lastErrorPx = 12.4;

  assert.deepEqual(trainerOverlayView(session, { now: 1200 }), {
    hidden: false,
    progressPercent: 10,
    step: "Solo · 9s",
    title: "4 takedowns",
    body: "12 px",
  });
});
