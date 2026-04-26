import assert from "node:assert/strict";
import test from "node:test";

import {
  CALIBRATION_TARGETS,
  buildCalibrationMapping,
  calibrationTargetLabel,
  defaultCalibration,
  hasAnyCalibrationMapping,
  loadCalibration,
  mapPointWithCalibration,
  saveCalibration,
} from "./calibration_logic.js";

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

function offsetCalibrationPoints(dx = 0.04, dy = -0.03) {
  return CALIBRATION_TARGETS.map((target) => ({
    id: target.id,
    rawX: target.x + dx,
    rawY: target.y + dy,
    targetX: target.x,
    targetY: target.y,
  }));
}

function assertClose(actual, expected, epsilon = 1e-8) {
  assert.ok(
    Math.abs(actual - expected) <= epsilon,
    `expected ${actual} to be within ${epsilon} of ${expected}`,
  );
}

test("calibration mapping maps offset raw points back onto target points", () => {
  const mapping = buildCalibrationMapping("spatial_geom", offsetCalibrationPoints());

  assert.ok(mapping);
  assert.equal(mapping.kind, "spatial_geom");
  assert.equal(mapping.points.length, CALIBRATION_TARGETS.length);
  assert.deepEqual(mapping.center, {
    id: "center",
    rawX: 0.54,
    rawY: 0.47,
    targetX: 0.5,
    targetY: 0.5,
  });

  for (const point of mapping.points) {
    const mapped = mapPointWithCalibration([point.rawX, point.rawY], mapping);
    assertClose(mapped[0], point.targetX);
    assertClose(mapped[1], point.targetY);
  }
});

test("calibration mapping rejects incomplete or degenerate input", () => {
  assert.equal(buildCalibrationMapping("spatial_geom", offsetCalibrationPoints().slice(1)), null);
  assert.equal(
    buildCalibrationMapping(
      "spatial_geom",
      CALIBRATION_TARGETS.map((target) => ({
        id: target.id,
        rawX: 0.5,
        rawY: 0.5,
        targetX: target.x,
        targetY: target.y,
      })),
    ),
    null,
  );
});

test("calibration storage loads modern and legacy mappings", () => {
  installLocalStorage();
  const mapping = buildCalibrationMapping("spatial_geom", offsetCalibrationPoints());
  const calibration = defaultCalibration();
  calibration.mappings.spatial_geom = mapping;
  saveCalibration(calibration);

  const loaded = loadCalibration();
  assert.equal(hasAnyCalibrationMapping(loaded), true);
  assert.equal(loaded.mappings.spatial_geom.kind, "spatial_geom");

  localStorage.setItem("gazeGame.calibration", JSON.stringify(mapping));
  const legacy = loadCalibration();
  assert.equal(legacy.mappings.spatial_geom.kind, "spatial_geom");
});

test("calibration storage falls back on malformed data", () => {
  installLocalStorage().set("gazeGame.calibration", "{nope");

  assert.deepEqual(loadCalibration(), defaultCalibration());
});

test("calibration target labels are human readable", () => {
  assert.equal(calibrationTargetLabel("topLeft"), "top-left");
  assert.equal(calibrationTargetLabel("bottomRight"), "bottom-right");
  assert.equal(calibrationTargetLabel("unknown"), "target");
});
