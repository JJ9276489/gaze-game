import assert from "node:assert/strict";
import test from "node:test";

import { PERSONAL_FEATURE_KEYS } from "./personal_model.js";
import {
  colorForName,
  controlsHiddenView,
  generateRoomCode,
  hudContextView,
  normalizePlayerName,
  normalizeRoom,
  personalModelView,
} from "./ui_state.js";

test("normalizers keep player names and room codes browser friendly", () => {
  assert.equal(normalizePlayerName("  Ada Lovelace  "), "Ada Lovelace");
  assert.equal(normalizePlayerName(""), "Guest");
  assert.equal(normalizePlayerName("x".repeat(40)), "x".repeat(32));

  assert.equal(normalizeRoom("abc123"), "ABC-123");
  assert.equal(normalizeRoom("abc-123!!!"), "ABC-123");
  assert.equal(normalizeRoom("abcd-1234-wxyz"), "ABCD-1234-WXYZ");
  assert.equal(normalizeRoom("room with spaces"), "ROOM-WITH-SPAC");
  assert.equal(normalizeRoom(""), "");
});

test("room code and player color generation are deterministic when seeded", () => {
  const code = generateRoomCode(() => 0);

  assert.equal(code, "AAAA-AAAA-AAAA");
  assert.deepEqual(colorForName("Ada"), colorForName("Ada"));
  assert.notDeepEqual(colorForName("Ada"), colorForName("Ben"));
});

test("hud context view separates local dojo controls from room controls", () => {
  assert.deepEqual(hudContextView("dojo"), {
    isDojo: true,
    roomScopeLabel: "Mode",
    copyRoomHidden: true,
    trainHidden: false,
    trainText: "Train NN",
    challengeHidden: true,
    multiplayerHidden: true,
    resetPersonalHidden: false,
  });
  assert.deepEqual(hudContextView("room"), {
    isDojo: false,
    roomScopeLabel: "Room",
    copyRoomHidden: false,
    trainHidden: true,
    trainText: "Dojo",
    challengeHidden: false,
    multiplayerHidden: false,
    resetPersonalHidden: true,
  });
});

test("controls hidden view owns labels and persistence value", () => {
  assert.deepEqual(controlsHiddenView(true), {
    hidden: true,
    buttonText: "Show buttons",
    ariaExpanded: "false",
    ariaPressed: "true",
    ariaLabel: "Show buttons",
    storageValue: "1",
  });
  assert.equal(controlsHiddenView(false).buttonText, "Hide buttons");
  assert.equal(controlsHiddenView(false).ariaExpanded, "true");
});

test("personal model view reports empty, partial, and trained states", () => {
  assert.deepEqual(
    personalModelView({
      kind: "spatial_geom",
      trainingSamples: [],
      personalStats: {},
      personalModels: {},
      isDojo: true,
    }),
    {
      challengeDisabled: true,
      multiplayerDisabled: true,
      resetDisabled: true,
      progressPercent: 0,
      label: "No data",
      meta: "Local only",
    },
  );

  const partialSamples = Array.from({ length: 30 }, (_, index) =>
    sample("spatial_geom", index),
  );
  const partial = personalModelView({
    kind: "spatial_geom",
    trainingSamples: partialSamples,
    personalStats: {},
    personalModels: {},
    isDojo: true,
  });
  assert.equal(partial.challengeDisabled, true);
  assert.equal(partial.resetDisabled, false);
  assert.equal(partial.progressPercent, 50);
  assert.equal(partial.label, "30/60 samples");

  const trained = personalModelView({
    kind: "spatial_geom",
    trainingSamples: partialSamples,
    personalStats: { byKind: { spatial_geom: { totalSamples: 120, retainedSamples: 30 } } },
    personalModels: { spatial_geom: { fitMeanPx: 42.4, totalSampleCount: 120 } },
    isDojo: false,
  });
  assert.equal(trained.challengeDisabled, false);
  assert.equal(trained.multiplayerDisabled, false);
  assert.equal(trained.progressPercent, 100);
  assert.equal(trained.label, "120 samples · fit 42 px");
  assert.equal(trained.meta, "30 retained for training");
});

function sample(kind, index) {
  return {
    version: 1,
    kind,
    targetX: 0.5,
    targetY: 0.5,
    rawX: 0.4,
    rawY: 0.4,
    features: new Array(PERSONAL_FEATURE_KEYS.length).fill(index),
  };
}
