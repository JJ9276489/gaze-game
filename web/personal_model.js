export const PERSONAL_MIN_SAMPLES = 60;

const PERSONAL_DATA_VERSION = 1;
const PERSONAL_MODEL_VERSION = 1;
const PERSONAL_STATS_VERSION = 1;
const PERSONAL_HIDDEN_UNITS = 18;
const PERSONAL_TRAINING_EPOCHS = 180;
const PERSONAL_TRAINING_RATE = 0.012;
const PERSONAL_MAX_SAMPLES_PER_MODEL = 6000;
const PERSONAL_MAX_TRAINING_SAMPLES = 900;
const PERSONAL_STORAGE_DATASET_KEY = "gazeGame.trainingSamples";
const PERSONAL_STORAGE_MODELS_KEY = "gazeGame.personalModels";
const PERSONAL_STORAGE_STATS_KEY = "gazeGame.personalStats";

export const PERSONAL_FEATURE_KEYS = [
  "raw_x",
  "raw_y",
  "face_center_x",
  "face_center_y",
  "face_scale",
  "head_yaw_deg",
  "head_pitch_deg",
  "head_roll_deg",
  "head_tx",
  "head_ty",
  "head_tz",
  "left_x",
  "left_y",
  "left_orth_y",
  "left_openness",
  "left_upper_gap",
  "left_lower_gap",
  "right_x",
  "right_y",
  "right_orth_y",
  "right_openness",
  "right_upper_gap",
  "right_lower_gap",
  "avg_x",
  "avg_y",
  "face_width",
  "face_height",
];

const PERSONAL_FEATURE_DEFAULTS = Object.freeze({
  raw_x: 0.5,
  raw_y: 0.5,
  face_center_x: 0.4944416582584381,
  face_center_y: 0.5652406215667725,
  face_scale: 0.35118332505226135,
  head_yaw_deg: 1.3236565589904785,
  head_pitch_deg: 7.618877410888672,
  head_roll_deg: -3.0949666500091553,
  head_tx: -0.2545686364173889,
  head_ty: -1.5664522647857666,
  head_tz: -32.763702392578125,
  left_x: 0.5043262839317322,
  left_y: 0.27408820390701294,
  left_orth_y: -0.06770181655883789,
  left_openness: 0.3100126087665558,
  left_upper_gap: 0.18136809766292572,
  left_lower_gap: 0.2398015856742859,
  right_x: 0.4890110492706299,
  right_y: 0.29838672280311584,
  right_orth_y: -0.05517496168613434,
  right_openness: 0.2884048819541931,
  right_upper_gap: 0.17870494723320007,
  right_lower_gap: 0.2097252458333969,
  avg_x: 0.4966680109500885,
  avg_y: 0.28623655438423157,
  face_width: 0.32,
  face_height: 0.42,
});

export function buildPersonalFeatureVector(payload, rawX, rawY) {
  const featurePayload = {
    ...payload,
    raw_x: rawX,
    raw_y: rawY,
  };
  return PERSONAL_FEATURE_KEYS.map((key) =>
    finiteOrDefault(featurePayload[key], PERSONAL_FEATURE_DEFAULTS[key] ?? 0),
  );
}

export function createTrainingSample(reading, target, viewport) {
  if (!Array.isArray(reading.features) || reading.features.length !== PERSONAL_FEATURE_KEYS.length) {
    return null;
  }
  return {
    version: PERSONAL_DATA_VERSION,
    kind: reading.kind,
    targetX: roundNumber(target.x),
    targetY: roundNumber(target.y),
    rawX: roundNumber(reading.rawX),
    rawY: roundNumber(reading.rawY),
    features: reading.features.map(roundNumber),
    viewportW: Math.max(1, Math.round(viewport.width)),
    viewportH: Math.max(1, Math.round(viewport.height)),
    ts: Date.now(),
  };
}

export function predictWithPersonalModel(model, reading) {
  if (!isUsablePersonalModel(model) || !Array.isArray(reading.features)) {
    return null;
  }
  const residual = runPersonalModel(model, reading.features);
  if (!residual) {
    return null;
  }
  return {
    x: clamp01(reading.rawX + residual[0]),
    y: clamp01(reading.rawY + residual[1]),
    method: "personal",
  };
}

export function runPersonalModel(model, features) {
  if (!isUsablePersonalModel(model) || features.length !== PERSONAL_FEATURE_KEYS.length) {
    return null;
  }
  const hidden = model.hidden;
  const inputSize = PERSONAL_FEATURE_KEYS.length;
  const activations = new Array(hidden).fill(0);
  for (let h = 0; h < hidden; h += 1) {
    let sum = model.b1[h] || 0;
    for (let i = 0; i < inputSize; i += 1) {
      const normalized = (features[i] - model.mean[i]) / model.std[i];
      sum += model.w1[h * inputSize + i] * normalized;
    }
    activations[h] = Math.tanh(sum);
  }

  const output = [model.b2[0] || 0, model.b2[1] || 0];
  for (let h = 0; h < hidden; h += 1) {
    output[0] += model.w2[h] * activations[h];
    output[1] += model.w2[hidden + h] * activations[h];
  }
  if (!Number.isFinite(output[0]) || !Number.isFinite(output[1])) {
    return null;
  }
  return output;
}

export function trainPersonalModel(kind, samples) {
  const allUsable = samples.filter(
    (sample) =>
      sample.kind === kind &&
      Array.isArray(sample.features) &&
      sample.features.length === PERSONAL_FEATURE_KEYS.length &&
      Number.isFinite(sample.rawX) &&
      Number.isFinite(sample.rawY) &&
      Number.isFinite(sample.targetX) &&
      Number.isFinite(sample.targetY),
  );
  if (allUsable.length < PERSONAL_MIN_SAMPLES) {
    throw new Error("Not enough samples for personal training.");
  }
  const usable = selectTrainingSubset(allUsable, PERSONAL_MAX_TRAINING_SAMPLES);

  const inputSize = PERSONAL_FEATURE_KEYS.length;
  const { mean: featureMean, std } = computeFeatureNormalization(usable);
  const residualMean = [
    mean(usable.map((sample) => sample.targetX - sample.rawX)),
    mean(usable.map((sample) => sample.targetY - sample.rawY)),
  ];
  const rng = mulberry32(hashString(`${kind}:${usable.length}:${usable[usable.length - 1]?.ts || 0}`));
  const hidden = PERSONAL_HIDDEN_UNITS;
  const w1 = new Array(hidden * inputSize);
  const b1 = new Array(hidden).fill(0);
  const w2 = new Array(hidden * 2);
  const b2 = [residualMean[0], residualMean[1]];
  const w1Scale = Math.sqrt(2 / inputSize) * 0.22;
  const w2Scale = Math.sqrt(2 / hidden) * 0.08;
  for (let index = 0; index < w1.length; index += 1) {
    w1[index] = randomNormal(rng) * w1Scale;
  }
  for (let index = 0; index < w2.length; index += 1) {
    w2[index] = randomNormal(rng) * w2Scale;
  }

  const model = {
    version: PERSONAL_MODEL_VERSION,
    kind,
    inputKeys: PERSONAL_FEATURE_KEYS,
    hidden,
    mean: featureMean,
    std,
    w1,
    b1,
    w2,
    b2,
    sampleCount: allUsable.length,
    trainedSampleCount: usable.length,
    trainedAt: Date.now(),
    fitMeanPx: 0,
  };

  const moments = {
    w1m: new Array(w1.length).fill(0),
    w1v: new Array(w1.length).fill(0),
    b1m: new Array(b1.length).fill(0),
    b1v: new Array(b1.length).fill(0),
    w2m: new Array(w2.length).fill(0),
    w2v: new Array(w2.length).fill(0),
    b2m: new Array(b2.length).fill(0),
    b2v: new Array(b2.length).fill(0),
  };
  const order = usable.map((_, index) => index);
  let step = 0;
  for (let epoch = 0; epoch < PERSONAL_TRAINING_EPOCHS; epoch += 1) {
    shuffleInPlace(order, rng);
    const lr = PERSONAL_TRAINING_RATE * (1 - epoch / (PERSONAL_TRAINING_EPOCHS * 1.35));
    for (const sampleIndex of order) {
      step += 1;
      trainPersonalSample(model, usable[sampleIndex], moments, step, lr);
    }
  }

  model.fitMeanPx = mean(
    usable.map((sample) => {
      const residual = runPersonalModel(model, sample.features) || [0, 0];
      return distancePx(
        clamp01(sample.rawX + residual[0]),
        clamp01(sample.rawY + residual[1]),
        sample.targetX,
        sample.targetY,
        sample.viewportW || window.innerWidth,
        sample.viewportH || window.innerHeight,
      );
    }),
  );
  return model;
}

export function loadTrainingSamples() {
  try {
    const saved = JSON.parse(localStorage.getItem(PERSONAL_STORAGE_DATASET_KEY) || "{}");
    const samples = Array.isArray(saved) ? saved : saved.samples;
    if (!Array.isArray(samples)) {
      return [];
    }
    return capTrainingSamples(samples.filter(isUsableTrainingSample));
  } catch {
    return [];
  }
}

export function saveTrainingSamples(samples) {
  const capped = capTrainingSamples(samples);
  try {
    localStorage.setItem(
      PERSONAL_STORAGE_DATASET_KEY,
      JSON.stringify({ version: PERSONAL_DATA_VERSION, samples: capped }),
    );
    return capped;
  } catch (error) {
    console.warn("Could not save full training dataset; trimming samples.", error);
    const reduced = capTrainingSamples(capped.slice(-Math.floor(PERSONAL_MAX_SAMPLES_PER_MODEL / 2)));
    localStorage.setItem(
      PERSONAL_STORAGE_DATASET_KEY,
      JSON.stringify({ version: PERSONAL_DATA_VERSION, samples: reduced }),
    );
    return reduced;
  }
}

export function loadPersonalModels() {
  try {
    const saved = JSON.parse(localStorage.getItem(PERSONAL_STORAGE_MODELS_KEY) || "{}");
    const models = saved.models && typeof saved.models === "object" ? saved.models : saved;
    const normalized = {};
    for (const [kind, model] of Object.entries(models || {})) {
      if (isUsablePersonalModel(model) && model.kind === kind) {
        normalized[kind] = model;
      }
    }
    return normalized;
  } catch {
    return {};
  }
}

export function savePersonalModels(models) {
  localStorage.setItem(
    PERSONAL_STORAGE_MODELS_KEY,
    JSON.stringify({ version: PERSONAL_MODEL_VERSION, models }),
  );
}

export function loadPersonalStats(samples = []) {
  const stats = { version: PERSONAL_STATS_VERSION, byKind: {} };
  try {
    const saved = JSON.parse(localStorage.getItem(PERSONAL_STORAGE_STATS_KEY) || "{}");
    const byKind = saved.byKind && typeof saved.byKind === "object" ? saved.byKind : {};
    for (const [kind, entry] of Object.entries(byKind)) {
      if (typeof kind !== "string" || !entry || typeof entry !== "object") {
        continue;
      }
      stats.byKind[kind] = {
        totalSamples: Math.max(0, Math.round(Number(entry.totalSamples) || 0)),
        retainedSamples: Math.max(0, Math.round(Number(entry.retainedSamples) || 0)),
        updatedAt: Math.max(0, Math.round(Number(entry.updatedAt) || 0)),
      };
    }
  } catch {
    // Ignore old or malformed stats; retained samples below provide a migration floor.
  }

  const retainedByKind = countSamplesByKind(samples);
  for (const [kind, retainedSamples] of Object.entries(retainedByKind)) {
    const entry = stats.byKind[kind] || { totalSamples: 0, retainedSamples: 0, updatedAt: 0 };
    stats.byKind[kind] = {
      totalSamples: Math.max(entry.totalSamples, retainedSamples),
      retainedSamples,
      updatedAt: entry.updatedAt,
    };
  }
  return stats;
}

export function savePersonalStats(stats) {
  const normalized = loadPersonalStatsFromObject(stats);
  localStorage.setItem(PERSONAL_STORAGE_STATS_KEY, JSON.stringify(normalized));
  return normalized;
}

export function recordTrainingSamples(stats, kind, addedCount, retainedSamples) {
  const next = loadPersonalStatsFromObject(stats);
  const entry = next.byKind[kind] || { totalSamples: 0, retainedSamples: 0, updatedAt: 0 };
  const cleanAdded = Math.max(0, Math.round(Number(addedCount) || 0));
  const cleanRetained = Math.max(0, Math.round(Number(retainedSamples) || 0));
  next.byKind[kind] = {
    totalSamples: Math.max(entry.totalSamples + cleanAdded, cleanRetained),
    retainedSamples: cleanRetained,
    updatedAt: Date.now(),
  };
  return savePersonalStats(next);
}

export function clearPersonalStatsForKind(stats, kind, retainedSamples = 0) {
  const next = loadPersonalStatsFromObject(stats);
  const cleanRetained = Math.max(0, Math.round(Number(retainedSamples) || 0));
  if (cleanRetained > 0) {
    const entry = next.byKind[kind] || { totalSamples: 0, retainedSamples: 0, updatedAt: 0 };
    next.byKind[kind] = {
      totalSamples: Math.max(entry.totalSamples, cleanRetained),
      retainedSamples: cleanRetained,
      updatedAt: Date.now(),
    };
  } else {
    delete next.byKind[kind];
  }
  return savePersonalStats(next);
}

export function personalStatsForKind(stats, kind, samples = []) {
  const normalized = loadPersonalStatsFromObject(stats);
  const retainedSamples = samplesForKind(samples, kind).length;
  const entry = normalized.byKind[kind] || {};
  return {
    totalSamples: Math.max(Math.round(Number(entry.totalSamples) || 0), retainedSamples),
    retainedSamples: Math.max(Math.round(Number(entry.retainedSamples) || 0), retainedSamples),
    updatedAt: Math.max(0, Math.round(Number(entry.updatedAt) || 0)),
  };
}

export function appendTrainingSamples(existingSamples, newSamples) {
  const usable = newSamples.filter(isUsableTrainingSample);
  return {
    addedCount: usable.length,
    samples: capTrainingSamples([...existingSamples, ...usable]),
  };
}

export function samplesForKind(samples, kind) {
  return samples.filter((sample) => sample.kind === kind && isUsableTrainingSample(sample));
}

export function isUsablePersonalModel(model) {
  const hidden = model?.hidden;
  const inputSize = PERSONAL_FEATURE_KEYS.length;
  return (
    model?.version === PERSONAL_MODEL_VERSION &&
    Array.isArray(model.inputKeys) &&
    model.inputKeys.join("|") === PERSONAL_FEATURE_KEYS.join("|") &&
    Number.isInteger(hidden) &&
    hidden > 0 &&
    Array.isArray(model.mean) &&
    model.mean.length === inputSize &&
    Array.isArray(model.std) &&
    model.std.length === inputSize &&
    Array.isArray(model.w1) &&
    model.w1.length === hidden * inputSize &&
    Array.isArray(model.b1) &&
    model.b1.length === hidden &&
    Array.isArray(model.w2) &&
    model.w2.length === hidden * 2 &&
    Array.isArray(model.b2) &&
    model.b2.length === 2
  );
}

export function mean(values) {
  if (!values.length) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function distancePx(x1, y1, x2, y2, width, height) {
  return Math.hypot((x1 - x2) * width, (y1 - y2) * height);
}

function trainPersonalSample(model, sample, moments, step, learningRate) {
  const inputSize = PERSONAL_FEATURE_KEYS.length;
  const hidden = model.hidden;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const epsilon = 1e-8;
  const l2 = 0.00008;
  const x = new Array(inputSize);
  for (let i = 0; i < inputSize; i += 1) {
    x[i] = (sample.features[i] - model.mean[i]) / model.std[i];
  }

  const hiddenValues = new Array(hidden);
  for (let h = 0; h < hidden; h += 1) {
    let sum = model.b1[h];
    for (let i = 0; i < inputSize; i += 1) {
      sum += model.w1[h * inputSize + i] * x[i];
    }
    hiddenValues[h] = Math.tanh(sum);
  }

  const outputs = [model.b2[0], model.b2[1]];
  for (let h = 0; h < hidden; h += 1) {
    outputs[0] += model.w2[h] * hiddenValues[h];
    outputs[1] += model.w2[hidden + h] * hiddenValues[h];
  }
  const errors = [
    outputs[0] - (sample.targetX - sample.rawX),
    outputs[1] - (sample.targetY - sample.rawY),
  ];

  const gradW2 = new Array(hidden * 2);
  const gradB2 = [errors[0], errors[1]];
  const gradHidden = new Array(hidden);
  for (let h = 0; h < hidden; h += 1) {
    gradW2[h] = errors[0] * hiddenValues[h] + l2 * model.w2[h];
    gradW2[hidden + h] = errors[1] * hiddenValues[h] + l2 * model.w2[hidden + h];
    gradHidden[h] =
      (errors[0] * model.w2[h] + errors[1] * model.w2[hidden + h]) *
      (1 - hiddenValues[h] * hiddenValues[h]);
  }

  const gradW1 = new Array(hidden * inputSize);
  const gradB1 = new Array(hidden);
  for (let h = 0; h < hidden; h += 1) {
    gradB1[h] = gradHidden[h];
    for (let i = 0; i < inputSize; i += 1) {
      const index = h * inputSize + i;
      gradW1[index] = gradHidden[h] * x[i] + l2 * model.w1[index];
    }
  }

  adamUpdate(model.w1, gradW1, moments.w1m, moments.w1v, step, learningRate, beta1, beta2, epsilon);
  adamUpdate(model.b1, gradB1, moments.b1m, moments.b1v, step, learningRate, beta1, beta2, epsilon);
  adamUpdate(model.w2, gradW2, moments.w2m, moments.w2v, step, learningRate, beta1, beta2, epsilon);
  adamUpdate(model.b2, gradB2, moments.b2m, moments.b2v, step, learningRate, beta1, beta2, epsilon);
}

function adamUpdate(values, gradients, firstMoment, secondMoment, step, lr, beta1, beta2, epsilon) {
  const beta1Correction = 1 - Math.pow(beta1, step);
  const beta2Correction = 1 - Math.pow(beta2, step);
  for (let index = 0; index < values.length; index += 1) {
    const gradient = gradients[index];
    firstMoment[index] = beta1 * firstMoment[index] + (1 - beta1) * gradient;
    secondMoment[index] = beta2 * secondMoment[index] + (1 - beta2) * gradient * gradient;
    const mHat = firstMoment[index] / beta1Correction;
    const vHat = secondMoment[index] / beta2Correction;
    values[index] -= (lr * mHat) / (Math.sqrt(vHat) + epsilon);
  }
}

function computeFeatureNormalization(samples) {
  const inputSize = PERSONAL_FEATURE_KEYS.length;
  const featureMean = new Array(inputSize).fill(0);
  for (const sample of samples) {
    for (let i = 0; i < inputSize; i += 1) {
      featureMean[i] += sample.features[i];
    }
  }
  for (let i = 0; i < inputSize; i += 1) {
    featureMean[i] /= samples.length;
  }

  const variance = new Array(inputSize).fill(0);
  for (const sample of samples) {
    for (let i = 0; i < inputSize; i += 1) {
      const delta = sample.features[i] - featureMean[i];
      variance[i] += delta * delta;
    }
  }
  const std = variance.map((value) => Math.max(Math.sqrt(value / samples.length), 0.001));
  return { mean: featureMean, std };
}

function selectTrainingSubset(samples, maxSamples) {
  if (samples.length <= maxSamples) {
    return samples;
  }
  const sorted = [...samples].sort((a, b) => (a.ts || 0) - (b.ts || 0));
  const selected = [];
  for (let index = 0; index < maxSamples; index += 1) {
    const sourceIndex = Math.round((index * (sorted.length - 1)) / (maxSamples - 1));
    selected.push(sorted[sourceIndex]);
  }
  return selected;
}

function capTrainingSamples(samples) {
  const byKind = new Map();
  for (const sample of samples) {
    if (!byKind.has(sample.kind)) {
      byKind.set(sample.kind, []);
    }
    byKind.get(sample.kind).push(sample);
  }
  const capped = [];
  for (const group of byKind.values()) {
    group.sort((a, b) => (a.ts || 0) - (b.ts || 0));
    capped.push(...group.slice(-PERSONAL_MAX_SAMPLES_PER_MODEL));
  }
  capped.sort((a, b) => (a.ts || 0) - (b.ts || 0));
  return capped;
}

function countSamplesByKind(samples) {
  const counts = {};
  for (const sample of samples) {
    if (!isUsableTrainingSample(sample)) {
      continue;
    }
    counts[sample.kind] = (counts[sample.kind] || 0) + 1;
  }
  return counts;
}

function loadPersonalStatsFromObject(value) {
  const stats = { version: PERSONAL_STATS_VERSION, byKind: {} };
  const byKind = value?.byKind && typeof value.byKind === "object" ? value.byKind : {};
  for (const [kind, entry] of Object.entries(byKind)) {
    if (typeof kind !== "string" || !entry || typeof entry !== "object") {
      continue;
    }
    stats.byKind[kind] = {
      totalSamples: Math.max(0, Math.round(Number(entry.totalSamples) || 0)),
      retainedSamples: Math.max(0, Math.round(Number(entry.retainedSamples) || 0)),
      updatedAt: Math.max(0, Math.round(Number(entry.updatedAt) || 0)),
    };
  }
  return stats;
}

function isUsableTrainingSample(sample) {
  return (
    sample?.version === PERSONAL_DATA_VERSION &&
    typeof sample.kind === "string" &&
    Array.isArray(sample.features) &&
    sample.features.length === PERSONAL_FEATURE_KEYS.length &&
    sample.features.every(Number.isFinite) &&
    Number.isFinite(sample.rawX) &&
    Number.isFinite(sample.rawY) &&
    Number.isFinite(sample.targetX) &&
    Number.isFinite(sample.targetY)
  );
}

function shuffleInPlace(items, random) {
  for (let index = items.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [items[index], items[swapIndex]] = [items[swapIndex], items[index]];
  }
  return items;
}

function roundNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return 0;
  }
  return Math.round(number * 1e6) / 1e6;
}

function hashString(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function mulberry32(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomNormal(random) {
  const u1 = Math.max(random(), 1e-12);
  const u2 = Math.max(random(), 1e-12);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function finiteOrDefault(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}
