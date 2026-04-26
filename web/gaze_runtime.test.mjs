import assert from "node:assert/strict";
import test from "node:test";

import { PERSONAL_FEATURE_KEYS } from "./personal_model.js";
import {
  buildFeatureFrame,
  computeHeadPose,
  estimateGaze,
  estimateHeuristicGaze,
  gazeModelLabel,
  isModelReading,
  modelConfig,
  normalizeModelKey,
} from "./gaze_runtime.js";

function syntheticLandmarks() {
  const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5 }));
  landmarks[0] = { x: 0.1, y: 0.2 };
  landmarks[1] = { x: 0.9, y: 0.8 };

  setEye(landmarks, {
    corners: [33, 133],
    upper: [159, 158, 160, 161],
    lower: [145, 153, 144, 163],
    iris: [469, 470, 471, 472],
    x: 0.38,
    y: 0.45,
  });
  setEye(landmarks, {
    corners: [263, 362],
    upper: [386, 385, 387, 388],
    lower: [374, 380, 373, 390],
    iris: [474, 475, 476, 477],
    x: 0.62,
    y: 0.45,
  });
  return landmarks;
}

function setEye(landmarks, { corners, upper, lower, iris, x, y }) {
  landmarks[corners[0]] = { x: x - 0.05, y };
  landmarks[corners[1]] = { x: x + 0.05, y };
  for (const index of upper) {
    landmarks[index] = { x, y: y - 0.02 };
  }
  for (const index of lower) {
    landmarks[index] = { x, y: y + 0.02 };
  }
  iris.forEach((index, offset) => {
    landmarks[index] = { x: x + 0.005 * (offset - 1.5), y: y + 0.004 };
  });
}

function identityPoseMatrix({ tx = 1, ty = 2, tz = 3 } = {}) {
  return [1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz, 0, 0, 0, 1];
}

function frame(width = 8, height = 8) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < data.length; index += 4) {
    data[index] = 128;
    data[index + 1] = 128;
    data[index + 2] = 128;
    data[index + 3] = 255;
  }
  return { width, height, data };
}

test("model helpers normalize configured gaze models", () => {
  assert.equal(normalizeModelKey("latest"), "latest");
  assert.equal(normalizeModelKey("missing"), "spatial_geom");
  assert.equal(modelConfig("missing").label, "Spatial geom");
  assert.equal(isModelReading("spatial_geom"), true);
  assert.equal(isModelReading("heuristic"), false);
  assert.equal(gazeModelLabel("latest"), "Concat latest");
});

test("feature frames include signed browser head pose features", () => {
  const pose = computeHeadPose(identityPoseMatrix({ tx: 1, ty: 2, tz: 3 }));
  assert.equal(Math.abs(Math.round(pose.yawDeg)), 0);
  assert.equal(pose.tx, 1);

  const featureFrame = buildFeatureFrame(syntheticLandmarks(), {
    facialTransformationMatrixes: [{ data: identityPoseMatrix({ tx: 1, ty: 2, tz: 3 }) }],
  });

  assert.equal(featureFrame.pose.tx, 1);
  assert.equal(featureFrame.payload.head_tx, -1);
  assert.equal(featureFrame.payload.head_ty, -2);
  assert.equal(featureFrame.payload.head_tz, 3);
  assert.ok(featureFrame.payload.left_x > 0.4 && featureFrame.payload.left_x < 0.6);
  assert.ok(featureFrame.payload.right_openness > 0);
});

test("estimateGaze returns no tracking when no face is visible", async () => {
  assert.deepEqual(await estimateGaze({ faceLandmarks: [] }), { tracking: false });
});

test("estimateGaze uses model output and falls back to heuristic on model errors", async () => {
  const landmarks = syntheticLandmarks();
  const goodModel = {
    key: "spatial_geom",
    extraFeatureKeys: [],
    inputNames: new Set(),
    session: {
      outputNames: ["gaze"],
      async run() {
        return { gaze: { data: [0.25, 0.75] } };
      },
    },
    ort: {},
  };

  const modelReading = await estimateGaze(
    { faceLandmarks: [landmarks], facialTransformationMatrixes: [{ data: identityPoseMatrix() }] },
    { gazeModel: goodModel, frame: frame() },
  );

  assert.equal(modelReading.tracking, true);
  assert.equal(modelReading.kind, "spatial_geom");
  assert.equal(modelReading.rawX, 0.25);
  assert.equal(modelReading.rawY, 0.75);
  assert.equal(modelReading.features.length, PERSONAL_FEATURE_KEYS.length);

  let errorSeen = false;
  const failingModel = {
    ...goodModel,
    session: {
      outputNames: ["gaze"],
      async run() {
        throw new Error("model failed");
      },
    },
  };
  const fallbackReading = await estimateGaze(
    { faceLandmarks: [landmarks], facialTransformationMatrixes: [{ data: identityPoseMatrix() }] },
    {
      gazeModel: failingModel,
      frame: frame(),
      onModelError() {
        errorSeen = true;
      },
    },
  );

  assert.equal(errorSeen, true);
  assert.equal(fallbackReading.kind, "heuristic");
  assert.equal(fallbackReading.features.length, PERSONAL_FEATURE_KEYS.length);
});

test("heuristic gaze stays finite for a synthetic feature frame", () => {
  const reading = estimateHeuristicGaze(buildFeatureFrame(syntheticLandmarks()));

  assert.equal(reading.tracking, true);
  assert.equal(reading.kind, "heuristic");
  assert.equal(Number.isFinite(reading.rawX), true);
  assert.equal(Number.isFinite(reading.rawY), true);
  assert.equal(reading.features.length, PERSONAL_FEATURE_KEYS.length);
});
