import { PERSONAL_MIN_SAMPLES, personalStatsForKind, samplesForKind } from "./personal_model.js";

export function normalizePlayerName(value) {
  return (value || "Guest").trim().slice(0, 32) || "Guest";
}

export function normalizeRoom(value) {
  const compact = (value || "").toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 12);
  if (!compact) {
    return "";
  }
  if (compact.length === 6) {
    return `${compact.slice(0, 3)}-${compact.slice(3)}`;
  }
  if (compact.length === 12) {
    return `${compact.slice(0, 4)}-${compact.slice(4, 8)}-${compact.slice(8)}`;
  }
  return compact;
}

export function generateRoomCode(random = Math.random) {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let code = "";
  for (let index = 0; index < 12; index += 1) {
    code += alphabet[Math.floor(random() * alphabet.length)];
  }
  return normalizeRoom(code);
}

export function colorForName(name) {
  let hash = 0;
  for (let index = 0; index < name.length; index += 1) {
    hash = (hash * 31 + name.charCodeAt(index)) >>> 0;
  }
  const hue = (hash % 360) / 360;
  return hslToRgb(hue, 0.74, 0.62);
}

export function hudContextView(sessionMode) {
  const isDojo = sessionMode === "dojo";
  return {
    isDojo,
    roomScopeLabel: isDojo ? "Mode" : "Room",
    copyRoomHidden: isDojo,
    trainHidden: !isDojo,
    trainText: isDojo ? "Train NN" : "Dojo",
    challengeHidden: isDojo,
    multiplayerHidden: isDojo,
    resetPersonalHidden: !isDojo,
  };
}

export function controlsHiddenView(hidden) {
  const controlsHidden = Boolean(hidden);
  return {
    hidden: controlsHidden,
    buttonText: controlsHidden ? "Show buttons" : "Hide buttons",
    ariaExpanded: String(!controlsHidden),
    ariaPressed: String(controlsHidden),
    ariaLabel: controlsHidden ? "Show buttons" : "Hide buttons",
    storageValue: controlsHidden ? "1" : "0",
  };
}

export function personalModelView({
  kind,
  trainingSamples,
  personalStats,
  personalModels,
  isDojo,
}) {
  const sampleCount = samplesForKind(trainingSamples, kind).length;
  const stats = personalStatsForKind(personalStats, kind, trainingSamples);
  const model = personalModels[kind];
  const totalSamples = Math.max(
    stats.totalSamples,
    Number(model?.totalSampleCount) || 0,
    Number(model?.sampleCount) || 0,
    sampleCount,
  );
  const hasPersonalData = Boolean(model) || sampleCount > 0;

  if (model) {
    const fit = Number.isFinite(model.fitMeanPx) ? `fit ${Math.round(model.fitMeanPx)} px` : "trained";
    return {
      challengeDisabled: false,
      multiplayerDisabled: false,
      resetDisabled: false,
      progressPercent: 100,
      label: `${totalSamples} samples · ${fit}`,
      meta: totalSamples > sampleCount
        ? `${sampleCount} retained for training`
        : isDojo
          ? "Train again to refine fit"
          : "Ready for room play",
    };
  }

  if (sampleCount > 0) {
    return {
      challengeDisabled: true,
      multiplayerDisabled: true,
      resetDisabled: false,
      progressPercent: Math.round(clamp01(sampleCount / PERSONAL_MIN_SAMPLES) * 100),
      label: `${totalSamples}/${PERSONAL_MIN_SAMPLES} samples`,
      meta: "Keep training",
    };
  }

  return {
    challengeDisabled: true,
    multiplayerDisabled: true,
    resetDisabled: !hasPersonalData,
    progressPercent: 0,
    label: "No data",
    meta: "Local only",
  };
}

function hslToRgb(h, s, l) {
  const hueToRgb = (p, q, t) => {
    let value = t;
    if (value < 0) value += 1;
    if (value > 1) value -= 1;
    if (value < 1 / 6) return p + (q - p) * 6 * value;
    if (value < 1 / 2) return q;
    if (value < 2 / 3) return p + (q - p) * (2 / 3 - value) * 6;
    return p;
  };
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [
    Math.round(hueToRgb(p, q, h + 1 / 3) * 255),
    Math.round(hueToRgb(p, q, h) * 255),
    Math.round(hueToRgb(p, q, h - 1 / 3) * 255),
  ];
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}
