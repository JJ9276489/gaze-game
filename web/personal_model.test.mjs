import assert from "node:assert/strict";
import test from "node:test";

import {
  PERSONAL_FEATURE_KEYS,
  appendTrainingSamples,
  clearPersonalStatsForKind,
  loadPersonalStats,
  loadTrainingSamples,
  personalStatsForKind,
  recordTrainingSamples,
  samplesForKind,
  saveTrainingSamples,
} from "./personal_model.js";

function installLocalStorage() {
  const store = new Map();
  globalThis.localStorage = {
    getItem: (key) => (store.has(key) ? store.get(key) : null),
    setItem: (key, value) => store.set(key, String(value)),
    removeItem: (key) => store.delete(key),
    clear: () => store.clear(),
  };
  return store;
}

function sample(kind, ts) {
  return {
    version: 1,
    kind,
    targetX: 0.5,
    targetY: 0.55,
    rawX: 0.45,
    rawY: 0.5,
    features: new Array(PERSONAL_FEATURE_KEYS.length).fill(0.1),
    viewportW: 1200,
    viewportH: 800,
    ts,
  };
}

test("training samples save, load, and filter by model kind", () => {
  installLocalStorage();
  const saved = saveTrainingSamples([
    sample("spatial_geom", 1),
    sample("latest", 2),
    { ...sample("bad", 3), features: [1] },
  ]);

  assert.equal(saved.length, 2);
  assert.deepEqual(loadTrainingSamples(), saved);
  assert.equal(samplesForKind(saved, "spatial_geom").length, 1);
  assert.equal(samplesForKind(saved, "latest").length, 1);
});

test("sample stats remain cumulative while retained samples are capped", () => {
  installLocalStorage();
  const existing = Array.from({ length: 6000 }, (_, index) => sample("spatial_geom", index + 1));
  let stats = loadPersonalStats(existing);
  const update = appendTrainingSamples(
    existing,
    Array.from({ length: 20 }, (_, index) => sample("spatial_geom", 7000 + index)),
  );
  const retained = samplesForKind(update.samples, "spatial_geom").length;

  stats = recordTrainingSamples(stats, "spatial_geom", update.addedCount, retained);
  const finalStats = personalStatsForKind(stats, "spatial_geom", update.samples);

  assert.equal(retained, 6000);
  assert.equal(finalStats.totalSamples, 6020);
  assert.equal(finalStats.retainedSamples, 6000);
});

test("clearing stats removes the selected model kind", () => {
  installLocalStorage();
  let stats = loadPersonalStats([sample("spatial_geom", 1), sample("latest", 2)]);
  stats = recordTrainingSamples(stats, "spatial_geom", 10, 10);
  stats = clearPersonalStatsForKind(stats, "spatial_geom", 0);

  assert.equal(personalStatsForKind(stats, "spatial_geom", []).totalSamples, 0);
  assert.equal(personalStatsForKind(stats, "latest", [sample("latest", 2)]).totalSamples, 1);
});
