import assert from "node:assert/strict";
import test from "node:test";

import {
  CHALLENGE_DURATION_MS,
  WAVE_TARGET_COUNT,
  isMultiplayerWaveMode,
  isWaveMode,
  makeEnemyTargets,
  normalizeRelayWave,
  normalizeWaveScores,
  normalizedWaveTargets,
} from "./game_logic.js";

test("wave mode helpers distinguish local and room waves", () => {
  assert.equal(isWaveMode("solo"), true);
  assert.equal(isWaveMode("multiplayer"), true);
  assert.equal(isWaveMode("dojo"), false);
  assert.equal(isMultiplayerWaveMode("multiplayer"), true);
  assert.equal(isMultiplayerWaveMode("solo"), false);
});

test("enemy target generation is deterministic and bounded", () => {
  const first = makeEnemyTargets("same-seed", 12);
  const second = makeEnemyTargets("same-seed", 12);

  assert.deepEqual(first, second);
  assert.equal(first.length, 12);
  for (const [index, target] of first.entries()) {
    assert.equal(target.id, `enemy-${index}`);
    assert.ok(target.x >= 0.12 && target.x <= 0.88);
    assert.ok(target.y >= 0.14 && target.y <= 0.76);
  }
});

test("relay wave normalization sanitizes partial server messages", () => {
  const wave = normalizeRelayWave({
    wave_id: "server-wave",
    seed: "normalized",
    targets: [
      { id: "a", x: -2, y: 1.5 },
      { x: 0.2, y: 0.4 },
    ],
    duration_ms: 0,
    server_ts: 12345,
    scores: [
      { id: "p1", name: "Ada", color: [1, 2, 3], score: 4 },
      { score: 99 },
    ],
  });

  assert.equal(wave.id, "server-wave");
  assert.equal(wave.seed, "normalized");
  assert.equal(wave.startedAt, 12345);
  assert.equal(wave.durationMs, CHALLENGE_DURATION_MS);
  assert.deepEqual(wave.targets, [
    { id: "a", x: 0, y: 1 },
    { id: "enemy-1", x: 0.2, y: 0.4 },
  ]);
  assert.deepEqual(wave.scores, [{ id: "p1", name: "Ada", color: [1, 2, 3], score: 4 }]);
});

test("relay wave normalization falls back to generated targets", () => {
  const wave = normalizeRelayWave({ id: "empty-wave", seed: "fallback", targets: [] });

  assert.equal(wave.id, "empty-wave");
  assert.equal(wave.targets.length, WAVE_TARGET_COUNT);
});

test("target and score normalizers tolerate malformed inputs", () => {
  assert.deepEqual(normalizedWaveTargets(null), []);
  assert.deepEqual(normalizeWaveScores(null), []);
  assert.deepEqual(normalizedWaveTargets([{ id: "x", x: "0.3", y: "bad" }]), [
    { id: "x", x: 0.3, y: 0 },
  ]);
  assert.deepEqual(normalizeWaveScores([{ id: "p2", score: -8 }]), [
    { id: "p2", name: "Guest", color: [255, 255, 255], score: 0 },
  ]);
});
