import { buildPersonalFeatureVector } from "./personal_model.js";

const TASKS_URL = "./vendor/mediapipe/tasks-vision/vision_bundle.mjs";
const WASM_URL = "./vendor/mediapipe/tasks-vision/wasm";
const FACE_MODEL_URL = "./vendor/mediapipe/models/face_landmarker.task";
const ORT_URL = "./vendor/onnxruntime/ort.wasm.min.mjs";
const ORT_WASM_PATH = "./vendor/onnxruntime/";
const USE_BROWSER_HEAD_POSE_FEATURES = true;

const EYE_CROP_WIDTH = 96;
const EYE_CROP_HEIGHT = 64;

const RIGHT_IRIS_POINTS = [469, 470, 471, 472];
const LEFT_IRIS_POINTS = [474, 475, 476, 477];
const RIGHT_EYE_CORNER_POINTS = [33, 133];
const RIGHT_EYE_UPPER_LID_POINTS = [159, 158, 160, 161];
const RIGHT_EYE_LOWER_LID_POINTS = [145, 153, 144, 163];
const LEFT_EYE_CORNER_POINTS = [263, 362];
const LEFT_EYE_UPPER_LID_POINTS = [386, 385, 387, 388];
const LEFT_EYE_LOWER_LID_POINTS = [374, 380, 373, 390];

const HEAD_FEATURE_KEYS = [
  "face_center_x",
  "face_center_y",
  "face_scale",
  "head_yaw_deg",
  "head_pitch_deg",
  "head_roll_deg",
  "head_tx",
  "head_ty",
  "head_tz",
];

const HEAD_FEATURE_DEFAULTS = Object.freeze({
  face_center_x: 0.4944416582584381,
  face_center_y: 0.5652406215667725,
  face_scale: 0.35118332505226135,
  head_yaw_deg: 1.3236565589904785,
  head_pitch_deg: 7.618877410888672,
  head_roll_deg: -3.0949666500091553,
  head_tx: -0.2545686364173889,
  head_ty: -1.5664522647857666,
  head_tz: -32.763702392578125,
});

const BROWSER_HEAD_POSE_SIGNS = Object.freeze({
  yaw: -1,
  pitch: -1,
  roll: -1,
  tx: -1,
  ty: -1,
  tz: 1,
});

const EXTRA_FEATURE_KEYS = [
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
];

const EXTRA_FEATURE_DEFAULTS = Object.freeze({
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
});

export const GAZE_MODELS = Object.freeze({
  spatial_geom: {
    key: "spatial_geom",
    label: "Spatial geom",
    url: "./models/vision_gaze_spatial_geom.onnx",
    extraFeatureKeys: EXTRA_FEATURE_KEYS,
  },
  latest: {
    key: "latest",
    label: "Concat latest",
    url: "./models/vision_gaze_latest.onnx",
    extraFeatureKeys: [],
  },
});
export const DEFAULT_GAZE_MODEL_KEY = "spatial_geom";

let mediaPipePromise = null;
const gazeModelPromises = new Map();

export async function loadFaceLandmarker() {
  if (!mediaPipePromise) {
    mediaPipePromise = import(TASKS_URL).then(async ({ FaceLandmarker, FilesetResolver }) => {
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      return FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: FACE_MODEL_URL,
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputFacialTransformationMatrixes: true,
      });
    });
  }
  return mediaPipePromise;
}

export async function loadGazeModel(modelKey) {
  const config = modelConfig(modelKey);
  if (!gazeModelPromises.has(config.key)) {
    gazeModelPromises.set(
      config.key,
      import(ORT_URL).then(async (ort) => {
        ort.env.wasm.wasmPaths = ORT_WASM_PATH;
        ort.env.wasm.numThreads = 1;
        const session = await ort.InferenceSession.create(config.url, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
        return {
          ort,
          session,
          key: config.key,
          label: config.label,
          extraFeatureKeys: config.extraFeatureKeys,
          inputNames: new Set(session.inputNames),
        };
      }),
    );
  }
  return gazeModelPromises.get(config.key);
}

export function drawMirroredVideoFrame(video, processingCanvas) {
  const width = video.videoWidth || 1;
  const height = video.videoHeight || 1;
  if (processingCanvas.width !== width || processingCanvas.height !== height) {
    processingCanvas.width = width;
    processingCanvas.height = height;
  }
  const context = processingCanvas.getContext("2d", { willReadFrequently: true });
  context.save();
  context.clearRect(0, 0, width, height);
  context.translate(width, 0);
  context.scale(-1, 1);
  context.drawImage(video, 0, 0, width, height);
  context.restore();
  return context.getImageData(0, 0, width, height);
}

export async function estimateGaze(result, { gazeModel = null, frame = null, onModelError = null } = {}) {
  const landmarks = result?.faceLandmarks?.[0];
  if (!landmarks) {
    return { tracking: false };
  }

  const featureFrame = buildFeatureFrame(landmarks, result);
  if (gazeModel && frame) {
    try {
      return await predictWithGazeModel(featureFrame, gazeModel, frame);
    } catch (error) {
      onModelError?.(error);
    }
  }
  return estimateHeuristicGaze(featureFrame);
}

export function buildFeatureFrame(landmarks, result = {}) {
  const leftEye = computeEyeFeature(
    landmarks,
    LEFT_EYE_CORNER_POINTS,
    LEFT_EYE_UPPER_LID_POINTS,
    LEFT_EYE_LOWER_LID_POINTS,
    LEFT_IRIS_POINTS,
  );
  const rightEye = computeEyeFeature(
    landmarks,
    RIGHT_EYE_CORNER_POINTS,
    RIGHT_EYE_UPPER_LID_POINTS,
    RIGHT_EYE_LOWER_LID_POINTS,
    RIGHT_IRIS_POINTS,
  );
  const face = computeFaceFeature(landmarks);
  const pose = computeHeadPose(firstMatrix(result));
  const payload = buildFeaturePayload(leftEye, rightEye, face, pose);

  return {
    landmarks,
    leftEye,
    rightEye,
    face,
    pose,
    payload,
  };
}

export function estimateHeuristicGaze(featureFrame) {
  const { leftEye, rightEye, face, pose } = featureFrame;
  const avgX = (leftEye.xRatio + rightEye.xRatio) / 2;
  const avgY = (leftEye.yRatio + rightEye.yRatio) / 2;

  let rawX = 0.5 + (avgX - 0.5) * 1.65 + (face.centerX - 0.5) * 0.52;
  let rawY = 0.5 + (avgY - 0.5) * 1.35 + (face.centerY - 0.5) * 0.44;
  if (pose) {
    rawX += pose.yawDeg / 95;
    rawY += pose.pitchDeg / 110;
  }

  return {
    tracking: true,
    kind: "heuristic",
    rawX,
    rawY,
    features: buildPersonalFeatureVector(featureFrame.payload, rawX, rawY),
  };
}

export function computeHeadPose(matrix) {
  if (!matrix || matrix.length < 12) {
    return null;
  }
  const values = Array.from(matrix);
  const r00 = values[0];
  const r10 = values[4];
  const r20 = values[8];
  const r11 = values[5];
  const r12 = values[6];
  const r21 = values[9];
  const r22 = values[10];
  const sy = Math.sqrt(r00 * r00 + r10 * r10);
  const singular = sy < 1e-6;
  const pitch = singular ? Math.atan2(-r12, r11) : Math.atan2(r21, r22);
  const yaw = Math.atan2(-r20, sy);
  const roll = singular ? 0 : Math.atan2(r10, r00);
  return {
    pitchDeg: (pitch * 180) / Math.PI,
    yawDeg: (yaw * 180) / Math.PI,
    rollDeg: (roll * 180) / Math.PI,
    tx: finiteOrDefault(values[3], HEAD_FEATURE_DEFAULTS.head_tx),
    ty: finiteOrDefault(values[7], HEAD_FEATURE_DEFAULTS.head_ty),
    tz: finiteOrDefault(values[11], HEAD_FEATURE_DEFAULTS.head_tz),
  };
}

export function normalizeModelKey(value) {
  return GAZE_MODELS[value] ? value : DEFAULT_GAZE_MODEL_KEY;
}

export function modelConfig(modelKey) {
  return GAZE_MODELS[normalizeModelKey(modelKey)];
}

export function isModelReading(kind) {
  return Boolean(GAZE_MODELS[kind]);
}

export function gazeModelLabel(kind) {
  return GAZE_MODELS[kind]?.label || "";
}

async function predictWithGazeModel(featureFrame, gazeModel, frame) {
  const leftEyeTensor = extractEyeCropTensor(
    frame,
    featureFrame.landmarks,
    LEFT_EYE_CORNER_POINTS,
    LEFT_EYE_UPPER_LID_POINTS,
    LEFT_EYE_LOWER_LID_POINTS,
    false,
  );
  const rightEyeTensor = extractEyeCropTensor(
    frame,
    featureFrame.landmarks,
    RIGHT_EYE_CORNER_POINTS,
    RIGHT_EYE_UPPER_LID_POINTS,
    RIGHT_EYE_LOWER_LID_POINTS,
    true,
  );

  const headFeatures = new Float32Array(
    HEAD_FEATURE_KEYS.map((key) =>
      featureValue(featureFrame.payload, key, HEAD_FEATURE_DEFAULTS[key] ?? 0),
    ),
  );
  const extraFeatureKeys = gazeModel.extraFeatureKeys || [];
  const extraFeatures = new Float32Array(
    extraFeatureKeys.map((key) =>
      featureValue(featureFrame.payload, key, EXTRA_FEATURE_DEFAULTS[key] ?? 0),
    ),
  );
  const feeds = {};
  if (gazeModel.inputNames.has("left_eye")) {
    feeds.left_eye = new gazeModel.ort.Tensor("float32", leftEyeTensor, [
      1,
      1,
      EYE_CROP_HEIGHT,
      EYE_CROP_WIDTH,
    ]);
  }
  if (gazeModel.inputNames.has("right_eye")) {
    feeds.right_eye = new gazeModel.ort.Tensor("float32", rightEyeTensor, [
      1,
      1,
      EYE_CROP_HEIGHT,
      EYE_CROP_WIDTH,
    ]);
  }
  if (gazeModel.inputNames.has("head_features")) {
    feeds.head_features = new gazeModel.ort.Tensor("float32", headFeatures, [
      1,
      HEAD_FEATURE_KEYS.length,
    ]);
  }
  if (gazeModel.inputNames.has("extra_features")) {
    feeds.extra_features = new gazeModel.ort.Tensor("float32", extraFeatures, [
      1,
      extraFeatureKeys.length,
    ]);
  }
  const outputs = await gazeModel.session.run(feeds);
  const gaze = outputs.gaze || outputs[gazeModel.session.outputNames[0]];
  const rawX = Number(gaze.data[0]);
  const rawY = Number(gaze.data[1]);
  return {
    tracking: true,
    kind: gazeModel.key,
    rawX,
    rawY,
    features: buildPersonalFeatureVector(featureFrame.payload, rawX, rawY),
  };
}

function computeEyeFeature(landmarks, cornerIndices, upperLidIndices, lowerLidIndices, irisPoints) {
  const [firstCorner, secondCorner] = cornerIndices.map((index) => point(landmarks, index));
  const [leftCorner, rightCorner] =
    firstCorner[0] <= secondCorner[0] ? [firstCorner, secondCorner] : [secondCorner, firstCorner];
  const irisCenter = meanPoint(landmarks, irisPoints);
  const upperLid = meanPoint(landmarks, upperLidIndices);
  const lowerLid = meanPoint(landmarks, lowerLidIndices);

  const horizontalAxis = sub(rightCorner, leftCorner);
  const eyeWidth = Math.max(norm(horizontalAxis), 1e-6);
  const horizontalUnit = scale(horizontalAxis, 1 / eyeWidth);
  let verticalUnit = [-horizontalUnit[1], horizontalUnit[0]];
  if (dot(sub(lowerLid, upperLid), verticalUnit) < 0) {
    verticalUnit = scale(verticalUnit, -1);
  }

  const verticalExtent = Math.max(dot(sub(lowerLid, upperLid), verticalUnit), 1e-6);
  const eyeCenter = scale(add(add(leftCorner, rightCorner), add(upperLid, lowerLid)), 0.25);
  const xProjection = dot(sub(irisCenter, leftCorner), horizontalUnit) / eyeWidth;
  const yProjection = dot(sub(irisCenter, upperLid), verticalUnit) / verticalExtent;
  const orthogonalOffset = dot(sub(irisCenter, eyeCenter), verticalUnit) / eyeWidth;
  const upperGap = norm(sub(irisCenter, upperLid)) / eyeWidth;
  const lowerGap = norm(sub(lowerLid, irisCenter)) / eyeWidth;

  return {
    xRatio: clamp01(xProjection),
    yRatio: clamp01(yProjection),
    orthY: orthogonalOffset,
    upperGap,
    lowerGap,
    eyeOpenness: verticalExtent / eyeWidth,
  };
}

function computeFaceFeature(landmarks) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const landmark of landmarks) {
    minX = Math.min(minX, landmark.x);
    maxX = Math.max(maxX, landmark.x);
    minY = Math.min(minY, landmark.y);
    maxY = Math.max(maxY, landmark.y);
  }
  const width = Math.max(maxX - minX, 1e-6);
  const height = Math.max(maxY - minY, 1e-6);
  return {
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
    width,
    height,
    scale: (width + height) / 2,
  };
}

function buildFeaturePayload(leftEye, rightEye, face, pose) {
  const avgX = (leftEye.xRatio + rightEye.xRatio) / 2;
  const avgY = (leftEye.yRatio + rightEye.yRatio) / 2;
  const headYaw = USE_BROWSER_HEAD_POSE_FEATURES
    ? signedPoseValue(pose?.yawDeg, HEAD_FEATURE_DEFAULTS.head_yaw_deg, BROWSER_HEAD_POSE_SIGNS.yaw)
    : HEAD_FEATURE_DEFAULTS.head_yaw_deg;
  const headPitch = USE_BROWSER_HEAD_POSE_FEATURES
    ? signedPoseValue(
        pose?.pitchDeg,
        HEAD_FEATURE_DEFAULTS.head_pitch_deg,
        BROWSER_HEAD_POSE_SIGNS.pitch,
      )
    : HEAD_FEATURE_DEFAULTS.head_pitch_deg;
  const headRoll = USE_BROWSER_HEAD_POSE_FEATURES
    ? signedPoseValue(pose?.rollDeg, HEAD_FEATURE_DEFAULTS.head_roll_deg, BROWSER_HEAD_POSE_SIGNS.roll)
    : HEAD_FEATURE_DEFAULTS.head_roll_deg;
  const headTx = USE_BROWSER_HEAD_POSE_FEATURES
    ? signedPoseValue(pose?.tx, HEAD_FEATURE_DEFAULTS.head_tx, BROWSER_HEAD_POSE_SIGNS.tx)
    : HEAD_FEATURE_DEFAULTS.head_tx;
  const headTy = USE_BROWSER_HEAD_POSE_FEATURES
    ? signedPoseValue(pose?.ty, HEAD_FEATURE_DEFAULTS.head_ty, BROWSER_HEAD_POSE_SIGNS.ty)
    : HEAD_FEATURE_DEFAULTS.head_ty;
  const headTz = USE_BROWSER_HEAD_POSE_FEATURES
    ? signedPoseValue(pose?.tz, HEAD_FEATURE_DEFAULTS.head_tz, BROWSER_HEAD_POSE_SIGNS.tz)
    : HEAD_FEATURE_DEFAULTS.head_tz;
  return {
    left_x: leftEye.xRatio,
    left_y: leftEye.yRatio,
    left_orth_y: leftEye.orthY,
    left_openness: leftEye.eyeOpenness,
    left_upper_gap: leftEye.upperGap,
    left_lower_gap: leftEye.lowerGap,
    right_x: rightEye.xRatio,
    right_y: rightEye.yRatio,
    right_orth_y: rightEye.orthY,
    right_openness: rightEye.eyeOpenness,
    right_upper_gap: rightEye.upperGap,
    right_lower_gap: rightEye.lowerGap,
    avg_x: avgX,
    avg_y: avgY,
    face_center_x: finiteOrDefault(face.centerX, HEAD_FEATURE_DEFAULTS.face_center_x),
    face_center_y: finiteOrDefault(face.centerY, HEAD_FEATURE_DEFAULTS.face_center_y),
    face_width: face.width,
    face_height: face.height,
    face_scale: finiteOrDefault(face.scale, HEAD_FEATURE_DEFAULTS.face_scale),
    head_yaw_deg: headYaw,
    head_pitch_deg: headPitch,
    head_roll_deg: headRoll,
    head_tx: headTx,
    head_ty: headTy,
    head_tz: headTz,
  };
}

function firstMatrix(result) {
  const matrix = result?.facialTransformationMatrixes?.[0] || result?.facialTransformationMatrices?.[0];
  if (!matrix) {
    return null;
  }
  if (Array.isArray(matrix)) {
    return matrix.flat ? matrix.flat() : [].concat(...matrix);
  }
  if (matrix.data) {
    return Array.from(matrix.data);
  }
  return null;
}

function extractEyeCropTensor(
  frame,
  landmarks,
  cornerIndices,
  upperLidIndices,
  lowerLidIndices,
  flipHorizontal,
) {
  const [firstCorner, secondCorner] = cornerIndices.map((index) => point(landmarks, index));
  const [leftCorner, rightCorner] =
    firstCorner[0] <= secondCorner[0] ? [firstCorner, secondCorner] : [secondCorner, firstCorner];
  const upperLid = meanPoint(landmarks, upperLidIndices);
  const lowerLid = meanPoint(landmarks, lowerLidIndices);

  const horizontalAxis = sub(rightCorner, leftCorner);
  const eyeWidth = Math.max(norm(horizontalAxis), 1e-6);
  const horizontalUnit = scale(horizontalAxis, 1 / eyeWidth);
  let verticalUnit = [-horizontalUnit[1], horizontalUnit[0]];
  if (dot(sub(lowerLid, upperLid), verticalUnit) < 0) {
    verticalUnit = scale(verticalUnit, -1);
  }

  const eyeHeight = Math.max(dot(sub(lowerLid, upperLid), verticalUnit), 1e-6);
  const center = scale(add(add(leftCorner, rightCorner), add(upperLid, lowerLid)), 0.25);
  const sourceWidth = frame.width;
  const sourceHeight = frame.height;
  const centerPx = [center[0] * sourceWidth, center[1] * sourceHeight];
  const cropWidthPx = eyeWidth * sourceWidth * 1.8;
  const cropHeightPx = Math.max(
    eyeHeight * sourceHeight * 3.2,
    cropWidthPx * (EYE_CROP_HEIGHT / EYE_CROP_WIDTH),
  );
  const xAxisPx = scale(horizontalUnit, cropWidthPx / 2);
  const yAxisPx = scale(verticalUnit, cropHeightPx / 2);
  const sourcePoints = [
    sub(sub(centerPx, xAxisPx), yAxisPx),
    sub(add(centerPx, xAxisPx), yAxisPx),
    add(sub(centerPx, xAxisPx), yAxisPx),
  ];

  return affineCropToTensor(frame, sourcePoints, flipHorizontal);
}

function affineCropToTensor(frame, sourcePoints, flipHorizontal) {
  const [s0, s1, s2] = sourcePoints;
  const ax = (s1[0] - s0[0]) / (EYE_CROP_WIDTH - 1);
  const ay = (s1[1] - s0[1]) / (EYE_CROP_WIDTH - 1);
  const bx = (s2[0] - s0[0]) / (EYE_CROP_HEIGHT - 1);
  const by = (s2[1] - s0[1]) / (EYE_CROP_HEIGHT - 1);
  const tensor = new Float32Array(EYE_CROP_WIDTH * EYE_CROP_HEIGHT);
  for (let y = 0; y < EYE_CROP_HEIGHT; y += 1) {
    for (let x = 0; x < EYE_CROP_WIDTH; x += 1) {
      const cropX = flipHorizontal ? EYE_CROP_WIDTH - 1 - x : x;
      const sourceX = s0[0] + ax * cropX + bx * y;
      const sourceY = s0[1] + ay * cropX + by * y;
      const gray = sampleReplicatedGray(frame, sourceX, sourceY);
      tensor[y * EYE_CROP_WIDTH + x] = gray / 127.5 - 1.0;
    }
  }
  return tensor;
}

function sampleReplicatedGray(frame, x, y) {
  const width = frame.width;
  const height = frame.height;
  const clampedX = Math.max(0, Math.min(width - 1, x));
  const clampedY = Math.max(0, Math.min(height - 1, y));
  const x0 = Math.floor(clampedX);
  const y0 = Math.floor(clampedY);
  const x1 = Math.min(x0 + 1, width - 1);
  const y1 = Math.min(y0 + 1, height - 1);
  const tx = clampedX - x0;
  const ty = clampedY - y0;

  const top = lerp(sampleGrayAt(frame, x0, y0), sampleGrayAt(frame, x1, y0), tx);
  const bottom = lerp(sampleGrayAt(frame, x0, y1), sampleGrayAt(frame, x1, y1), tx);
  return lerp(top, bottom, ty);
}

function sampleGrayAt(frame, x, y) {
  const index = (y * frame.width + x) * 4;
  const data = frame.data;
  return data[index] * 0.299 + data[index + 1] * 0.587 + data[index + 2] * 0.114;
}

function point(landmarks, index) {
  const landmark = landmarks[index];
  return [landmark.x, landmark.y];
}

function meanPoint(landmarks, indices) {
  let x = 0;
  let y = 0;
  for (const index of indices) {
    x += landmarks[index].x;
    y += landmarks[index].y;
  }
  return [x / indices.length, y / indices.length];
}

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1]];
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1]];
}

function scale(a, value) {
  return [a[0] * value, a[1] * value];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1];
}

function norm(a) {
  return Math.sqrt(dot(a, a));
}

function featureValue(payload, key, fallback) {
  return finiteOrDefault(payload?.[key], fallback);
}

function finiteOrDefault(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function signedPoseValue(value, fallback, sign) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return number * sign;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function lerp(a, b, alpha) {
  return a * (1 - alpha) + b * alpha;
}
