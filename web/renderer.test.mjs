import assert from "node:assert/strict";
import test from "node:test";

import {
  buildLeaderboardRows,
  renderStage,
  resizeCanvasForDpr,
  shouldShowLocalCursor,
} from "./renderer.js";

function fakeContext() {
  const calls = [];
  const target = {
    calls,
    measureText(text) {
      calls.push(["measureText", text]);
      return { width: String(text).length * 8 };
    },
  };
  return new Proxy(target, {
    get(object, property) {
      if (property in object) {
        return object[property];
      }
      return (...args) => {
        calls.push([String(property), ...args]);
      };
    },
    set(object, property, value) {
      object[property] = value;
      calls.push(["set", String(property), value]);
      return true;
    },
  });
}

test("resizeCanvasForDpr keeps display and backing dimensions separate", () => {
  const canvas = { clientWidth: 320, clientHeight: 180, width: 0, height: 0 };
  const context = fakeContext();

  const size = resizeCanvasForDpr(canvas, context, 2);

  assert.deepEqual(size, {
    width: 320,
    height: 180,
    backingWidth: 640,
    backingHeight: 360,
    dpr: 2,
  });
  assert.equal(canvas.width, 640);
  assert.equal(canvas.height, 360);
  assert.deepEqual(context.calls.find((call) => call[0] === "setTransform"), [
    "setTransform",
    2,
    0,
    0,
    2,
    0,
    0,
  ]);
});

test("local cursor visibility follows calibration and trainer mode", () => {
  const local = { tracking: true };

  assert.equal(shouldShowLocalCursor({ local, calibrationActive: false, trainerSession: null }), true);
  assert.equal(shouldShowLocalCursor({ local, calibrationActive: true, trainerSession: null }), false);
  assert.equal(
    shouldShowLocalCursor({
      local,
      calibrationActive: false,
      trainerSession: { active: true, mode: "dojo" },
    }),
    false,
  );
  assert.equal(
    shouldShowLocalCursor({
      local,
      calibrationActive: false,
      trainerSession: { active: true, mode: "multiplayer" },
    }),
    true,
  );
  assert.equal(shouldShowLocalCursor({ local: { tracking: false }, calibrationActive: false }), false);
});

test("leaderboard rows include the local score and sort by score then name", () => {
  const rows = buildLeaderboardRows({
    waveScores: new Map([
      ["b", { id: "b", name: "Ben", color: [2, 2, 2], score: 4 }],
      ["a", { id: "a", name: "Ada", color: [1, 1, 1], score: 4 }],
    ]),
    local: { id: "me", name: "Me", color: [3, 3, 3] },
    session: { active: true, mode: "multiplayer", score: 5 },
  });

  assert.deepEqual(
    rows.map((row) => [row.id, row.label || row.name, row.score]),
    [
      ["me", "You", 5],
      ["a", "Ada", 4],
      ["b", "Ben", 4],
    ],
  );
});

test("renderStage draws a complete snapshot without owning app state", () => {
  const canvas = { clientWidth: 640, clientHeight: 360, width: 0, height: 0 };
  const context = fakeContext();

  renderStage({
    canvas,
    context,
    dpr: 1,
    now: 1000,
    peers: [{ x: 0.25, y: 0.5, color: [255, 255, 255], name: "Peer", tracking: true, lastSeen: 900 }],
    local: { id: "local", name: "Local", color: [117, 216, 255], x: 0.5, y: 0.5, tracking: true },
    calibrationActive: false,
    calibrationTarget: { x: 0.2, y: 0.2 },
    trainerSession: {
      active: true,
      mode: "multiplayer",
      phase: "active",
      phaseStartedAt: 900,
      dwellStartedAt: 950,
      score: 2,
    },
    trainerTarget: { x: 0.6, y: 0.4 },
    waveScores: new Map(),
    trainCaptureMs: 640,
    challengeDwellMs: 240,
    challengeTargetRadiusPx: 34,
  });

  assert.equal(canvas.width, 640);
  assert.equal(canvas.height, 360);
  assert.ok(context.calls.some((call) => call[0] === "fillText" && call[1] === "WAVE SCORE"));
  assert.ok(context.calls.some((call) => call[0] === "fillText" && call[1] === "You"));
});
