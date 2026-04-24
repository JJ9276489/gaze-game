import {
  PERSONAL_MIN_SAMPLES,
  appendTrainingSamples,
  buildPersonalFeatureVector,
  createTrainingSample,
  distancePx,
  loadPersonalModels,
  loadTrainingSamples,
  mean,
  predictWithPersonalModel,
  samplesForKind,
  savePersonalModels,
  saveTrainingSamples,
  trainPersonalModel,
} from "./personal_model.js";

const TASKS_VERSION = "0.10.34";
const TASKS_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}`;
const WASM_URL = `${TASKS_URL}/wasm`;
const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const ORT_VERSION = "1.24.3";
const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.wasm.min.mjs`;
const ORT_WASM_PATH = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
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

const GAZE_MODELS = Object.freeze({
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
const DEFAULT_GAZE_MODEL_KEY = "spatial_geom";

const CALIBRATION_VERSION = 3;
const CALIBRATION_SETTLE_MS = 600;
const CALIBRATION_CAPTURE_MS = 900;
const CALIBRATION_MIN_SAMPLES = 10;
const CALIBRATION_INSIDE_TOLERANCE = -0.04;
const CALIBRATION_TARGETS = [
  { id: "center", x: 0.5, y: 0.5 },
  { id: "topLeft", x: 0.14, y: 0.16 },
  { id: "topRight", x: 0.86, y: 0.16 },
  { id: "bottomRight", x: 0.86, y: 0.84 },
  { id: "bottomLeft", x: 0.14, y: 0.84 },
];

const TRAIN_TARGETS = [
  { id: "train-01", x: 0.15, y: 0.16 },
  { id: "train-02", x: 0.36, y: 0.16 },
  { id: "train-03", x: 0.64, y: 0.16 },
  { id: "train-04", x: 0.85, y: 0.16 },
  { id: "train-05", x: 0.22, y: 0.34 },
  { id: "train-06", x: 0.48, y: 0.34 },
  { id: "train-07", x: 0.73, y: 0.34 },
  { id: "train-08", x: 0.15, y: 0.52 },
  { id: "train-09", x: 0.36, y: 0.52 },
  { id: "train-10", x: 0.64, y: 0.52 },
  { id: "train-11", x: 0.85, y: 0.52 },
  { id: "train-12", x: 0.27, y: 0.68 },
  { id: "train-13", x: 0.52, y: 0.68 },
  { id: "train-14", x: 0.78, y: 0.68 },
  { id: "train-15", x: 0.15, y: 0.78 },
  { id: "train-16", x: 0.36, y: 0.78 },
  { id: "train-17", x: 0.64, y: 0.78 },
  { id: "train-18", x: 0.85, y: 0.78 },
];
const EVAL_TARGETS = [
  { id: "eval-01", x: 0.24, y: 0.24 },
  { id: "eval-02", x: 0.5, y: 0.22 },
  { id: "eval-03", x: 0.76, y: 0.24 },
  { id: "eval-04", x: 0.22, y: 0.5 },
  { id: "eval-05", x: 0.51, y: 0.49 },
  { id: "eval-06", x: 0.78, y: 0.5 },
  { id: "eval-07", x: 0.24, y: 0.74 },
  { id: "eval-08", x: 0.5, y: 0.76 },
  { id: "eval-09", x: 0.76, y: 0.74 },
];
const TRAIN_SETTLE_MS = 360;
const TRAIN_CAPTURE_MS = 640;
const TRAIN_MIN_SAMPLES_PER_TARGET = 6;
const EVAL_SETTLE_MS = 420;
const EVAL_CAPTURE_MS = 700;
const EVAL_MIN_SAMPLES_PER_TARGET = 6;
const CHALLENGE_DURATION_MS = 30000;
const CHALLENGE_TARGET_RADIUS_PX = 34;
const CHALLENGE_DWELL_MS = 240;
const WAVE_MODES = new Set(["solo", "multiplayer"]);

const $ = (id) => document.getElementById(id);

const elements = {
  canvas: $("stageCanvas"),
  video: $("camera"),
  lobby: $("lobby"),
  hud: $("hud"),
  form: $("joinForm"),
  createButton: $("createButton"),
  joinButton: $("joinButton"),
  trainButton: $("trainButton"),
  evaluateButton: $("evaluateButton"),
  challengeButton: $("challengeButton"),
  multiplayerButton: $("multiplayerButton"),
  calibrateButton: $("calibrateButton"),
  resetPersonalButton: $("resetPersonalButton"),
  leaveButton: $("leaveButton"),
  nameInput: $("nameInput"),
  roomInput: $("roomInput"),
  relayInput: $("relayInput"),
  mouseModeInput: $("mouseModeInput"),
  statusLine: $("statusLine"),
  toast: $("toast"),
  secureBadge: $("secureBadge"),
  roomLabel: $("roomLabel"),
  copyRoomButton: $("copyRoomButton"),
  fullscreenButton: $("fullscreenButton"),
  toggleControlsButton: $("toggleControlsButton"),
  nameLabel: $("nameLabel"),
  trackingLabel: $("trackingLabel"),
  modelSelect: $("modelSelect"),
  personalModelLabel: $("personalModelLabel"),
  personalModelMeta: $("personalModelMeta"),
  personalProgressFill: $("personalProgressFill"),
  calibrationOverlay: $("calibrationOverlay"),
  calibrationStep: $("calibrationStep"),
  calibrationTitle: $("calibrationTitle"),
  calibrationBody: $("calibrationBody"),
  calibrationCancelButton: $("calibrationCancelButton"),
  trainerOverlay: $("trainerOverlay"),
  trainerStep: $("trainerStep"),
  trainerTitle: $("trainerTitle"),
  trainerBody: $("trainerBody"),
  trainerProgressFill: $("trainerProgressFill"),
  trainerCancelButton: $("trainerCancelButton"),
};

const ctx = elements.canvas.getContext("2d");

const state = {
  animationFrame: 0,
  cameraFrame: 0,
  sendTimer: 0,
  ws: null,
  connected: false,
  running: false,
  source: "gaze",
  faceLandmarker: null,
  gazeModel: null,
  modelKey: DEFAULT_GAZE_MODEL_KEY,
  mediaStream: null,
  lastVideoTime: -1,
  inferenceBusy: false,
  seq: 0,
  rawReading: null,
  local: {
    id: "local",
    name: "Guest",
    room: "",
    color: [117, 216, 255],
    x: 0.5,
    y: 0.5,
    tracking: false,
  },
  peers: new Map(),
  calibration: loadCalibration(),
  calibrationSession: null,
  trainingSamples: loadTrainingSamples(),
  personalModels: loadPersonalModels(),
  trainerSession: null,
  controlsHidden: loadControlsHidden(),
  processingCanvas: document.createElement("canvas"),
  processingFrameData: null,
  cropCanvas: document.createElement("canvas"),
};

let mediaPipePromise = null;
const gazeModelPromises = new Map();

init();

function init() {
  elements.nameInput.value = localStorage.getItem("gazeGame.name") || "";
  elements.roomInput.value = normalizeRoom(new URLSearchParams(location.search).get("room") || "");
  elements.relayInput.value =
    new URLSearchParams(location.search).get("relay") ||
    localStorage.getItem("gazeGame.relay") ||
    defaultRelayUrl();
  state.modelKey = normalizeModelKey(
    new URLSearchParams(location.search).get("model") ||
      localStorage.getItem("gazeGame.model") ||
      DEFAULT_GAZE_MODEL_KEY,
  );
  elements.modelSelect.value = state.modelKey;
  setControlsHidden(state.controlsHidden, false);
  refreshFullscreenButton();
  refreshPersonalModelLabel();

  if (!window.isSecureContext) {
    elements.secureBadge.textContent = "HTTPS needed";
    elements.secureBadge.classList.add("warn");
  }

  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    void startSession(false);
  });
  elements.createButton.addEventListener("click", () => {
    elements.roomInput.value = generateRoomCode();
    void startSession(true);
  });
  elements.leaveButton.addEventListener("click", stopSession);
  elements.copyRoomButton.addEventListener("click", copyRoomCode);
  elements.fullscreenButton.addEventListener("click", () => {
    void toggleFullscreen();
  });
  elements.toggleControlsButton.addEventListener("click", () => {
    setControlsHidden(!state.controlsHidden);
  });
  elements.trainButton.addEventListener("click", () => {
    void startTrainerRun("dojo");
  });
  elements.evaluateButton.addEventListener("click", () => {
    void startTrainerRun("trial");
  });
  elements.challengeButton.addEventListener("click", () => {
    void startTrainerRun("solo");
  });
  elements.multiplayerButton.addEventListener("click", () => {
    void startTrainerRun("multiplayer");
  });
  elements.resetPersonalButton.addEventListener("click", resetCurrentPersonalModel);
  elements.calibrateButton.addEventListener("click", () => {
    void startCalibrationFromUser();
  });
  elements.calibrationCancelButton.addEventListener("click", () => cancelCalibration(true));
  elements.trainerCancelButton.addEventListener("click", () => cancelTrainerRun(true));
  elements.modelSelect.addEventListener("change", () => {
    void switchGazeModel(elements.modelSelect.value);
  });
  window.addEventListener("resize", drawStage);
  window.addEventListener("pointermove", updateMouseSource);
  window.addEventListener("keydown", handleGlobalKeydown);
  document.addEventListener("fullscreenchange", refreshFullscreenButton);
  document.addEventListener("webkitfullscreenchange", refreshFullscreenButton);

  drawLoop();
}

async function startSession(createRoom) {
  setBusy(true);
  hideToast();
  state.source = elements.mouseModeInput.checked ? "mouse" : "gaze";

  const name = (elements.nameInput.value || "Guest").trim().slice(0, 32) || "Guest";
  const room = normalizeRoom(elements.roomInput.value || (createRoom ? generateRoomCode() : ""));
  if (!room) {
    setBusy(false);
    setStatus("Room code required");
    return;
  }

  state.local.name = name;
  state.local.room = room;
  state.local.color = colorForName(name);
  elements.roomInput.value = room;
  elements.roomLabel.textContent = room;
  elements.roomLabel.title = room;
  elements.nameLabel.textContent = name;

  localStorage.setItem("gazeGame.name", name);
  localStorage.setItem("gazeGame.relay", elements.relayInput.value.trim());

  try {
    setStatus(state.source === "mouse" ? "Starting pointer" : "Starting camera");
    if (state.source === "mouse") {
      startMouseSource();
    } else {
      await startGazeSource();
    }

    setStatus("Connecting");
    await connectRelay();
    showHud();
    state.running = true;
    startSending();
    if (state.source === "gaze" && !hasAnyCalibrationMapping()) {
      showToast("Click Calibrate and keep the page fullscreen for best results.");
    }
  } catch (error) {
    stopSession();
    const message = error instanceof Error ? error.message : String(error);
    setStatus("Could not start");
    showToast(message);
  } finally {
    setBusy(false);
  }
}

function stopSession() {
  state.running = false;
  state.connected = false;
  state.peers.clear();
  window.clearInterval(state.sendTimer);
  state.sendTimer = 0;
  window.cancelAnimationFrame(state.cameraFrame);
  state.cameraFrame = 0;
  cancelCalibration(false);
  cancelTrainerRun(false);

  if (state.ws) {
    state.ws.onclose = null;
    state.ws.close();
    state.ws = null;
  }
  if (state.mediaStream) {
    for (const track of state.mediaStream.getTracks()) {
      track.stop();
    }
    state.mediaStream = null;
  }
  elements.video.srcObject = null;
  state.local.tracking = false;
  state.rawReading = null;
  showLobby();
  setStatus("Ready");
}

function setBusy(isBusy) {
  elements.createButton.disabled = isBusy;
  elements.joinButton.disabled = isBusy;
}

function showHud() {
  elements.lobby.classList.add("hidden");
  elements.hud.classList.remove("hidden");
  syncHudSuppression();
}

function showLobby() {
  elements.hud.classList.add("hidden");
  elements.hud.classList.remove("hud-suppressed");
  elements.lobby.classList.remove("hidden");
}

function syncHudSuppression() {
  const isTargetRunActive = Boolean(state.calibrationSession?.active || state.trainerSession?.active);
  elements.hud.classList.toggle("hud-suppressed", isTargetRunActive);
}

function setControlsHidden(hidden, persist = true) {
  state.controlsHidden = Boolean(hidden);
  elements.hud.classList.toggle("hud-controls-hidden", state.controlsHidden);
  elements.toggleControlsButton.textContent = state.controlsHidden ? "Show buttons" : "Hide buttons";
  elements.toggleControlsButton.setAttribute("aria-expanded", String(!state.controlsHidden));
  elements.toggleControlsButton.setAttribute("aria-pressed", String(state.controlsHidden));
  elements.toggleControlsButton.setAttribute(
    "aria-label",
    state.controlsHidden ? "Show buttons" : "Hide buttons",
  );
  if (persist) {
    localStorage.setItem("gazeGame.controlsHidden", state.controlsHidden ? "1" : "0");
  }
}

function handleGlobalKeydown(event) {
  if (!state.running || event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) {
    return;
  }
  if (isEditableTarget(event.target)) {
    return;
  }

  const key = event.key.toLowerCase();
  if (key === "h") {
    event.preventDefault();
    setControlsHidden(!state.controlsHidden);
  } else if (key === "f") {
    event.preventDefault();
    void toggleFullscreen();
  }
}

function isEditableTarget(target) {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName.toLowerCase();
  return target.isContentEditable || tag === "input" || tag === "select" || tag === "textarea";
}

function setStatus(message) {
  elements.statusLine.textContent = message;
}

function setTrackingStatus(message) {
  elements.trackingLabel.textContent = message;
}

async function copyRoomCode() {
  const room = state.local.room || elements.roomLabel.textContent || "";
  if (!room) {
    return;
  }
  try {
    await navigator.clipboard.writeText(room);
    showToast("Room code copied.");
  } catch {
    showToast(`Room code: ${room}`);
  }
  window.setTimeout(hideToast, 1000);
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.remove("hidden");
}

function hideToast() {
  elements.toast.classList.add("hidden");
  elements.toast.textContent = "";
}

async function startGazeSource() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser cannot access a camera.");
  }
  if (!window.isSecureContext) {
    throw new Error("Camera access needs HTTPS or localhost.");
  }

  state.mediaStream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user",
    },
    audio: false,
  });
  elements.video.srcObject = state.mediaStream;
  await elements.video.play();

  setStatus("Loading gaze model");
  const [faceLandmarker, gazeModel] = await Promise.all([
    loadFaceLandmarker(),
    loadGazeModel(state.modelKey).catch((error) => {
      console.warn("Gaze model unavailable; using heuristic fallback.", error);
      return null;
    }),
  ]);
  state.faceLandmarker = faceLandmarker;
  state.gazeModel = gazeModel;
  state.cropCanvas.width = EYE_CROP_WIDTH;
  state.cropCanvas.height = EYE_CROP_HEIGHT;
  if (!state.gazeModel) {
    showToast("Browser model unavailable; using fallback gaze.");
  }
  state.lastVideoTime = -1;
  state.running = true;
  readGazeFrame();
}

function startMouseSource() {
  state.local.tracking = true;
  setTrackingStatus("Mouse");
}

async function loadFaceLandmarker() {
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

async function loadGazeModel(modelKey) {
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

async function switchGazeModel(modelKey) {
  const config = modelConfig(modelKey);
  state.modelKey = config.key;
  elements.modelSelect.value = config.key;
  localStorage.setItem("gazeGame.model", config.key);
  cancelCalibration(false);
  cancelTrainerRun(false);
  state.rawReading = null;
  state.local.tracking = false;
  refreshPersonalModelLabel();
  if (state.source !== "gaze") {
    return;
  }
  setTrackingStatus("Loading model");
  try {
    state.gazeModel = await loadGazeModel(config.key);
    showToast(`Using ${config.label}. Calibrate this model.`);
    window.setTimeout(hideToast, 1400);
  } catch (error) {
    console.warn("Gaze model switch failed; using fallback.", error);
    state.gazeModel = null;
    showToast(`${config.label} unavailable; using fallback gaze.`);
  }
}

function readGazeFrame() {
  if (!state.running || state.source !== "gaze" || !state.faceLandmarker) {
    return;
  }

  const video = elements.video;
  if (
    !state.inferenceBusy &&
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    video.currentTime !== state.lastVideoTime
  ) {
    state.lastVideoTime = video.currentTime;
    state.inferenceBusy = true;
    processGazeFrame()
      .then((reading) => applyGazeReading(reading))
      .catch((error) => {
        console.warn("Gaze frame failed", error);
        state.local.tracking = false;
        setTrackingStatus("Camera paused");
      })
      .finally(() => {
        state.inferenceBusy = false;
      });
  }

  state.cameraFrame = window.requestAnimationFrame(readGazeFrame);
}

async function processGazeFrame() {
  drawMirroredVideoFrame();
  const result = state.faceLandmarker.detectForVideo(state.processingCanvas, performance.now());
  return estimateGaze(result);
}

function drawMirroredVideoFrame() {
  const width = elements.video.videoWidth || 1;
  const height = elements.video.videoHeight || 1;
  if (state.processingCanvas.width !== width || state.processingCanvas.height !== height) {
    state.processingCanvas.width = width;
    state.processingCanvas.height = height;
  }
  const context = state.processingCanvas.getContext("2d", { willReadFrequently: true });
  context.save();
  context.clearRect(0, 0, width, height);
  context.translate(width, 0);
  context.scale(-1, 1);
  context.drawImage(elements.video, 0, 0, width, height);
  context.restore();
  state.processingFrameData = context.getImageData(0, 0, width, height);
}

function applyGazeReading(reading) {
  if (!reading?.tracking) {
    state.local.tracking = false;
    state.rawReading = null;
    setTrackingStatus("No face");
    if (state.calibrationSession?.active) {
      elements.calibrationBody.textContent = "Face lost. Look back at the camera, then keep staring at the dot.";
    }
    return;
  }

  state.rawReading = reading;
  updateCalibrationSession(reading);
  const basePrediction = calibrateReading(reading);
  const personalPrediction = predictWithPersonalModel(state.personalModels[reading.kind], reading);
  const predicted = personalPrediction || basePrediction;
  const nextX = predicted.x;
  const nextY = predicted.y;

  const alpha = 0.32;
  state.local.x = state.local.tracking ? lerp(state.local.x, nextX, alpha) : nextX;
  state.local.y = state.local.tracking ? lerp(state.local.y, nextY, alpha) : nextY;
  state.local.tracking = true;
  updateTrainerRun(reading, {
    active: {
      x: state.local.x,
      y: state.local.y,
      method: predicted.method,
    },
    base: basePrediction,
    personal: personalPrediction,
  });
  setTrackingStatus(trackingStatusFor(reading, predicted.method));
}

function calibrateReading(reading) {
  const mapping = getCalibrationMapping(reading.kind);
  if (mapping) {
    const mapped = mapPointWithCalibration([reading.rawX, reading.rawY], mapping);
    if (mapped) {
      return {
        x: clamp01(mapped[0]),
        y: clamp01(mapped[1]),
        method: "calibration",
      };
    }
  }

  if (isModelReading(reading.kind)) {
    return {
      x: clamp01(0.5 + (reading.rawX - state.calibration.centerX)),
      y: clamp01(0.5 + (reading.rawY - state.calibration.centerY)),
      method: "base",
    };
  }

  return {
    x: clamp01(0.5 + (reading.rawX - state.calibration.centerX) * state.calibration.gainX),
    y: clamp01(0.5 + (reading.rawY - state.calibration.centerY) * state.calibration.gainY),
    method: "fallback",
  };
}

function trackingStatusFor(reading, method) {
  const config = GAZE_MODELS[reading.kind];
  if (config) {
    if (method === "personal") return `${config.label} + personal NN`;
    if (method === "calibration") return `${config.label} calibrated`;
    return config.label;
  }
  return method === "calibration" ? "Fallback calibrated" : "Fallback";
}

async function startCalibrationFromUser() {
  if (state.source !== "gaze" || !state.running) {
    return;
  }
  if (!state.rawReading?.tracking) {
    showToast("Wait for face tracking before calibrating.");
    return;
  }

  hideToast();
  await maybeEnterFullscreen();
  cancelTrainerRun(false);
  state.calibrationSession = {
    active: true,
    kind: state.rawReading.kind || state.modelKey,
    stepIndex: 0,
    phase: "settle",
    phaseStartedAt: performance.now(),
    samples: [],
    capturedPoints: [],
  };
  refreshCalibrationOverlay();
}

function cancelCalibration(showMessage) {
  const wasActive = Boolean(state.calibrationSession?.active);
  state.calibrationSession = null;
  refreshCalibrationOverlay();
  if (showMessage && wasActive) {
    showToast("Calibration canceled.");
    window.setTimeout(hideToast, 1000);
  }
}

function updateCalibrationSession(reading) {
  const session = state.calibrationSession;
  if (!session?.active || reading.kind !== session.kind) {
    return;
  }

  const now = performance.now();
  if (session.phase === "settle") {
    if (now - session.phaseStartedAt < CALIBRATION_SETTLE_MS) {
      return;
    }
    session.phase = "capture";
    session.phaseStartedAt = now;
    session.samples = [];
    refreshCalibrationOverlay();
    return;
  }

  session.samples.push([reading.rawX, reading.rawY]);
  if (
    now - session.phaseStartedAt < CALIBRATION_CAPTURE_MS ||
    session.samples.length < CALIBRATION_MIN_SAMPLES
  ) {
    return;
  }

  const target = visibleOverlayTarget(CALIBRATION_TARGETS[session.stepIndex], ".calibration-card");
  const average = averagePoint(session.samples);
  session.capturedPoints.push({
    id: target.id,
    rawX: average[0],
    rawY: average[1],
    targetX: target.x,
    targetY: target.y,
  });

  session.stepIndex += 1;
  if (session.stepIndex >= CALIBRATION_TARGETS.length) {
    finishCalibration(session);
    return;
  }

  session.phase = "settle";
  session.phaseStartedAt = now;
  session.samples = [];
  refreshCalibrationOverlay();
}

function finishCalibration(session) {
  const mapping = buildCalibrationMapping(session.kind, session.capturedPoints);
  if (!mapping) {
    cancelCalibration(false);
    showToast("Calibration failed. Try again in better light.");
    return;
  }

  state.calibration.centerX = mapping.center?.rawX ?? state.calibration.centerX;
  state.calibration.centerY = mapping.center?.rawY ?? state.calibration.centerY;
  state.calibration.mappings[session.kind] = mapping;
  saveCalibration(state.calibration);
  state.calibrationSession = null;
  refreshCalibrationOverlay();
  showToast("Calibration saved.");
  window.setTimeout(hideToast, 1200);
}

function refreshCalibrationOverlay() {
  const session = state.calibrationSession;
  syncHudSuppression();
  if (!session?.active) {
    elements.calibrationOverlay.classList.add("hidden");
    return;
  }

  const target = visibleOverlayTarget(CALIBRATION_TARGETS[session.stepIndex], ".calibration-card");
  elements.calibrationOverlay.classList.remove("hidden");
  elements.calibrationStep.textContent = `Calibration ${session.stepIndex + 1} of ${CALIBRATION_TARGETS.length}`;
  elements.calibrationTitle.textContent = "Look at the dot";
  elements.calibrationBody.textContent =
    session.phase === "capture"
      ? "Keep staring at the target until it moves."
      : `Hold still on the ${calibrationTargetLabel(target.id)} target.`;
}

function isFullscreenActive() {
  return Boolean(document.fullscreenElement || document.webkitFullscreenElement);
}

function refreshFullscreenButton() {
  const active = isFullscreenActive();
  elements.fullscreenButton.textContent = active ? "Exit full" : "Full screen";
  elements.fullscreenButton.setAttribute("aria-pressed", String(active));
  elements.fullscreenButton.setAttribute(
    "aria-label",
    active ? "Exit full screen" : "Enter full screen",
  );
}

async function toggleFullscreen() {
  if (isFullscreenActive()) {
    await exitFullscreen();
  } else {
    await enterFullscreen(true);
  }
  refreshFullscreenButton();
}

async function enterFullscreen(showError) {
  const element = document.documentElement;
  if (isFullscreenActive()) {
    return true;
  }
  const request = element.requestFullscreen || element.webkitRequestFullscreen;
  if (!request) {
    if (showError) {
      showToast("Full screen is not available in this browser.");
    }
    return false;
  }
  try {
    await Promise.resolve(request.call(element));
    return true;
  } catch (error) {
    console.warn("Could not enter fullscreen.", error);
    if (showError) {
      showToast("Use the browser full screen control if this button is blocked.");
    }
    return false;
  }
}

async function exitFullscreen() {
  const exit = document.exitFullscreen || document.webkitExitFullscreen;
  if (!exit || !isFullscreenActive()) {
    return;
  }
  try {
    await Promise.resolve(exit.call(document));
  } catch (error) {
    console.warn("Could not exit fullscreen.", error);
  }
}

async function maybeEnterFullscreen() {
  await enterFullscreen(false);
}

function isWaveMode(mode) {
  return WAVE_MODES.has(mode);
}

function isMultiplayerWaveMode(mode) {
  return mode === "multiplayer";
}

function trainerModeLabel(mode) {
  if (mode === "dojo") return "Dojo";
  if (mode === "trial") return "Trial";
  if (mode === "solo") return "Solo";
  if (mode === "multiplayer") return "Multiplayer wave";
  return "Run";
}

function trainerTargetNoun(mode) {
  if (mode === "dojo") return "dummy";
  if (mode === "trial") return "mark";
  return "enemy";
}

async function startTrainerRun(mode) {
  if (state.source !== "gaze" || !state.running) {
    return;
  }
  if (!state.rawReading?.tracking) {
    showToast("Wait for face tracking first.");
    return;
  }

  hideToast();
  await maybeEnterFullscreen();
  cancelCalibration(false);

  const kind = state.rawReading.kind || state.modelKey;
  if ((mode === "trial" || isWaveMode(mode)) && !state.personalModels[kind]) {
    showToast("Enter the Dojo and train your personal NN first.");
    return;
  }
  const now = performance.now();
  const targets =
    isWaveMode(mode)
      ? [randomEnemyTarget()]
      : shuffledTargets(mode === "trial" ? EVAL_TARGETS : TRAIN_TARGETS);

  state.trainerSession = {
    active: true,
    mode,
    kind,
    targets,
    stepIndex: 0,
    phase: isWaveMode(mode) ? "active" : "settle",
    phaseStartedAt: now,
    startedAt: now,
    endsAt: isWaveMode(mode) ? now + CHALLENGE_DURATION_MS : 0,
    samples: [],
    capturedSamples: [],
    errors: [],
    baseErrors: [],
    personalErrors: [],
    score: 0,
    dwellStartedAt: 0,
    lastErrorPx: null,
    lastUiAt: 0,
  };
  refreshTrainerOverlay();
}

function cancelTrainerRun(showMessage) {
  const wasActive = Boolean(state.trainerSession?.active);
  state.trainerSession = null;
  refreshTrainerOverlay();
  if (showMessage && wasActive) {
    showToast("Run canceled.");
    window.setTimeout(hideToast, 1000);
  }
}

function updateTrainerRun(reading, point) {
  const session = state.trainerSession;
  if (!session?.active) {
    return;
  }
  if (reading.kind !== session.kind) {
    elements.trainerBody.textContent = "Model changed. Start a new run.";
    return;
  }

  if (isWaveMode(session.mode)) {
    updateWaveRun(session, point.active);
    return;
  }

  const now = performance.now();
  const settleMs = session.mode === "trial" ? EVAL_SETTLE_MS : TRAIN_SETTLE_MS;
  const captureMs = session.mode === "trial" ? EVAL_CAPTURE_MS : TRAIN_CAPTURE_MS;
  const minSamples =
    session.mode === "trial" ? EVAL_MIN_SAMPLES_PER_TARGET : TRAIN_MIN_SAMPLES_PER_TARGET;
  const target = visibleOverlayTarget(session.targets[session.stepIndex], ".trainer-dock");
  if (!target) {
    finishTrainerRun(session);
    return;
  }

  if (session.phase === "settle") {
    if (now - session.phaseStartedAt >= settleMs) {
      session.phase = "capture";
      session.phaseStartedAt = now;
      session.samples = [];
      refreshTrainerOverlay();
    }
    return;
  }

  if (session.mode === "dojo") {
    const sample = createTrainingSample(reading, target, {
      width: window.innerWidth,
      height: window.innerHeight,
    });
    if (sample) {
      session.samples.push(sample);
    }
  } else {
    const activeError = distanceToTargetPx(point.active, target);
    session.samples.push(activeError);
    session.baseErrors.push(distanceToTargetPx(point.base, target));
    if (point.personal) {
      session.personalErrors.push(distanceToTargetPx(point.personal, target));
    }
  }

  if (now - session.lastUiAt > 150) {
    session.lastUiAt = now;
    refreshTrainerOverlay();
  }

  if (now - session.phaseStartedAt < captureMs || session.samples.length < minSamples) {
    return;
  }

  if (session.mode === "dojo") {
    session.capturedSamples.push(...session.samples);
  } else {
    session.errors.push(...session.samples);
  }

  session.stepIndex += 1;
  if (session.stepIndex >= session.targets.length) {
    finishTrainerRun(session);
    return;
  }

  session.phase = "settle";
  session.phaseStartedAt = now;
  session.samples = [];
  refreshTrainerOverlay();
}

function updateWaveRun(session, point) {
  const now = performance.now();
  if (now >= session.endsAt) {
    finishTrainerRun(session);
    return;
  }

  const target = visibleOverlayTarget(session.targets[0], ".trainer-dock");
  const errorPx = distanceToTargetPx(point, target);
  session.lastErrorPx = errorPx;
  if (errorPx <= CHALLENGE_TARGET_RADIUS_PX) {
    if (!session.dwellStartedAt) {
      session.dwellStartedAt = now;
    }
    if (now - session.dwellStartedAt >= CHALLENGE_DWELL_MS) {
      session.score += 1;
      session.targets = [randomEnemyTarget(target)];
      session.dwellStartedAt = 0;
      session.phaseStartedAt = now;
    }
  } else {
    session.dwellStartedAt = 0;
  }

  if (now - session.lastUiAt > 150) {
    session.lastUiAt = now;
    refreshTrainerOverlay();
  }
}

function finishTrainerRun(session) {
  state.trainerSession = null;
  refreshTrainerOverlay();
  if (session.mode === "dojo") {
    void finishTrainingRun(session);
    return;
  }
  if (session.mode === "trial") {
    finishEvaluationRun(session);
    return;
  }
  showToast(`${trainerModeLabel(session.mode)} complete: ${session.score} takedowns.`);
  window.setTimeout(hideToast, 1800);
}

async function finishTrainingRun(session) {
  const trainingUpdate = appendTrainingSamples(state.trainingSamples, session.capturedSamples);
  state.trainingSamples = saveTrainingSamples(trainingUpdate.samples);
  refreshPersonalModelLabel();
  if (trainingUpdate.addedCount < 1) {
    showToast("No usable samples collected.");
    return;
  }

  const samples = samplesForKind(state.trainingSamples, session.kind);
  if (samples.length < PERSONAL_MIN_SAMPLES) {
    showToast(`Saved ${samples.length} samples. Need ${PERSONAL_MIN_SAMPLES} to train.`);
    return;
  }

  showToast(`Training personal NN from ${samples.length} Dojo samples...`);
  await nextFrame();
  try {
    const model = trainPersonalModel(session.kind, samples);
    state.personalModels[session.kind] = model;
    savePersonalModels(state.personalModels);
    refreshPersonalModelLabel();
    showToast(
      `Dojo complete: ${samples.length} samples, ${Math.round(model.fitMeanPx)} px fit.`,
    );
    window.setTimeout(hideToast, 2200);
  } catch (error) {
    console.warn("Personal NN training failed", error);
    showToast("Personal NN training failed.");
  }
}

function finishEvaluationRun(session) {
  if (!session.errors.length) {
    showToast("No test samples collected.");
    return;
  }
  const meanPx = mean(session.personalErrors.length ? session.personalErrors : session.errors);
  const baseMeanPx = session.baseErrors.length ? mean(session.baseErrors) : null;
  const improvementPx = baseMeanPx === null ? null : baseMeanPx - meanPx;
  const model = state.personalModels[session.kind];
  if (model) {
    model.lastEvalMeanPx = meanPx;
    model.lastBaseEvalMeanPx = baseMeanPx;
    model.lastEvalDeltaPx = improvementPx;
    model.lastEvalAt = Date.now();
    savePersonalModels(state.personalModels);
  }
  refreshPersonalModelLabel();
  if (improvementPx !== null) {
    const direction = improvementPx >= 0 ? "better" : "worse";
    showToast(`Trial: ${Math.round(meanPx)} px, ${Math.abs(Math.round(improvementPx))} px ${direction}.`);
  } else {
    showToast(`Trial complete: ${Math.round(meanPx)} px mean error.`);
  }
  window.setTimeout(hideToast, 2200);
}

function refreshTrainerOverlay() {
  const session = state.trainerSession;
  syncHudSuppression();
  if (!session?.active) {
    elements.trainerOverlay.classList.add("hidden");
    elements.trainerProgressFill.style.width = "0%";
    return;
  }

  elements.trainerOverlay.classList.remove("hidden");
  if (isWaveMode(session.mode)) {
    const secondsLeft = Math.max(0, Math.ceil((session.endsAt - performance.now()) / 1000));
    const progress = clamp01((performance.now() - session.startedAt) / CHALLENGE_DURATION_MS);
    elements.trainerProgressFill.style.width = `${Math.round(progress * 100)}%`;
    elements.trainerStep.textContent = `${trainerModeLabel(session.mode)} · ${secondsLeft}s`;
    elements.trainerTitle.textContent = `${session.score} takedowns`;
    elements.trainerBody.textContent =
      session.lastErrorPx === null ? "Acquire enemies." : `${Math.round(session.lastErrorPx)} px`;
    return;
  }

  const total = session.targets.length;
  const label = trainerModeLabel(session.mode);
  const targetProgress = session.stepIndex / total;
  const captureMs = session.mode === "trial" ? EVAL_CAPTURE_MS : TRAIN_CAPTURE_MS;
  const phaseProgress =
    session.phase === "capture"
      ? Math.min(1, (performance.now() - session.phaseStartedAt) / captureMs)
      : 0;
  const progress = targetProgress + phaseProgress / total;
  elements.trainerProgressFill.style.width = `${Math.round(progress * 100)}%`;
  elements.trainerStep.textContent = `${label} ${session.stepIndex + 1} of ${total}`;
  const targetNoun = trainerTargetNoun(session.mode);
  elements.trainerTitle.textContent =
    session.phase === "capture" ? `Hold ${targetNoun}` : `Acquire ${targetNoun}`;
  if (session.mode === "dojo") {
    elements.trainerBody.textContent = `${session.capturedSamples.length + session.samples.length} samples`;
  } else {
    elements.trainerBody.textContent = `${session.errors.length + session.samples.length} readings`;
  }
}

function resetCurrentPersonalModel() {
  const kind = state.modelKey;
  const hadModel = Boolean(state.personalModels[kind]);
  const before = state.trainingSamples.length;
  if (!hadModel && !samplesForKind(state.trainingSamples, kind).length) {
    showToast("No personal NN data.");
    window.setTimeout(hideToast, 1200);
    return;
  }
  if (!window.confirm("Reset personal NN data for the selected model?")) {
    return;
  }
  delete state.personalModels[kind];
  state.trainingSamples = state.trainingSamples.filter((sample) => sample.kind !== kind);
  savePersonalModels(state.personalModels);
  state.trainingSamples = saveTrainingSamples(state.trainingSamples);
  refreshPersonalModelLabel();
  showToast(
    hadModel || before !== state.trainingSamples.length ? "Personal NN reset." : "No personal NN data.",
  );
  window.setTimeout(hideToast, 1400);
}

function updateMouseSource(event) {
  if (state.source !== "mouse" || !state.connected) {
    return;
  }
  state.local.x = clamp01(event.clientX / Math.max(window.innerWidth, 1));
  state.local.y = clamp01(event.clientY / Math.max(window.innerHeight, 1));
  state.local.tracking = true;
}

function connectRelay() {
  const relayUrl = normalizeRelayUrl(elements.relayInput.value);
  elements.relayInput.value = relayUrl;

  return new Promise((resolve, reject) => {
    const ws = new WebSocket(relayUrl);
    state.ws = ws;
    let settled = false;
    const timeout = window.setTimeout(() => {
      if (!settled) {
        settled = true;
        ws.close();
        reject(new Error("Relay connection timed out."));
      }
    }, 8000);

    ws.addEventListener("open", () => {
      ws.send(
        JSON.stringify({
          type: "join",
          room: state.local.room,
          name: state.local.name,
          color: state.local.color,
        }),
      );
    });

    ws.addEventListener("message", (event) => {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }

      if (message.type === "welcome") {
        state.local.id = message.id;
        state.connected = true;
        state.peers.clear();
        for (const peer of message.peers || []) {
          state.peers.set(peer.id, {
            id: peer.id,
            name: peer.name || "Guest",
            color: peer.color || [255, 255, 255],
            x: null,
            y: null,
            tracking: false,
            lastSeen: 0,
          });
        }
        setTrackingStatus(state.source === "mouse" ? "Mouse" : "Gaze");
        if (!settled) {
          settled = true;
          window.clearTimeout(timeout);
          resolve();
        }
        return;
      }

      handleRelayMessage(message);
    });

    ws.addEventListener("error", () => {
      if (!settled) {
        settled = true;
        window.clearTimeout(timeout);
        reject(new Error(`Could not connect to ${relayUrl}`));
      }
    });

    ws.addEventListener("close", () => {
      if (!settled) {
        settled = true;
        window.clearTimeout(timeout);
        reject(new Error("Relay closed before joining."));
        return;
      }
      if (state.running) {
        state.connected = false;
        setTrackingStatus("Disconnected");
        showToast("Relay disconnected");
      }
    });
  });
}

function handleRelayMessage(message) {
  if (message.type === "peer_join") {
    state.peers.set(message.id, {
      id: message.id,
      name: message.name || "Guest",
      color: message.color || [255, 255, 255],
      x: null,
      y: null,
      tracking: false,
      lastSeen: performance.now(),
    });
    return;
  }

  if (message.type === "peer_leave") {
    state.peers.delete(message.id);
    return;
  }

  if (message.type === "cursor") {
    const hasCoordinates = typeof message.x === "number" && typeof message.y === "number";
    const peer = state.peers.get(message.id) || {
      id: message.id,
      name: message.name || "Guest",
      color: message.color || [255, 255, 255],
      x: null,
      y: null,
      tracking: false,
      lastSeen: 0,
    };
    peer.name = message.name || peer.name;
    peer.color = message.color || peer.color;
    peer.x = hasCoordinates ? clamp01(message.x) : null;
    peer.y = hasCoordinates ? clamp01(message.y) : null;
    peer.tracking = Boolean(message.tracking) && hasCoordinates;
    peer.lastSeen = performance.now();
    state.peers.set(message.id, peer);
  }
}

function startSending() {
  window.clearInterval(state.sendTimer);
  state.sendTimer = window.setInterval(sendCursor, 33);
}

function sendCursor() {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN || !state.connected) {
    return;
  }
  const calibrationActive = Boolean(state.calibrationSession?.active);
  const trainerHidden = Boolean(
    state.trainerSession?.active && !isMultiplayerWaveMode(state.trainerSession.mode),
  );
  state.seq += 1;
  state.ws.send(
    JSON.stringify({
      type: "cursor",
      room: state.local.room,
      x: state.local.tracking && !calibrationActive && !trainerHidden ? state.local.x : null,
      y: state.local.tracking && !calibrationActive && !trainerHidden ? state.local.y : null,
      tracking: state.local.tracking && !calibrationActive && !trainerHidden,
      seq: state.seq,
      ts: Date.now(),
    }),
  );
}

function drawLoop() {
  drawStage();
  state.animationFrame = window.requestAnimationFrame(drawLoop);
}

function drawStage() {
  const dpr = Math.max(window.devicePixelRatio || 1, 1);
  const width = Math.floor(elements.canvas.clientWidth * dpr);
  const height = Math.floor(elements.canvas.clientHeight * dpr);
  if (elements.canvas.width !== width || elements.canvas.height !== height) {
    elements.canvas.width = width;
    elements.canvas.height = height;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = elements.canvas.clientWidth;
  const h = elements.canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#05080d";
  ctx.fillRect(0, 0, w, h);
  drawGrid(w, h);

  for (const peer of state.peers.values()) {
    if (peer.x === null || peer.y === null) {
      continue;
    }
    const age = performance.now() - peer.lastSeen;
    const alpha = peer.tracking ? Math.max(0.25, 1 - age / 3000) : 0.2;
    drawCursor(peer.x, peer.y, peer.color, peer.name, alpha, false);
  }

  const trainerHidesCursor = Boolean(
    state.trainerSession?.active && !isWaveMode(state.trainerSession.mode),
  );
  if (state.local.tracking && !state.calibrationSession?.active && !trainerHidesCursor) {
    drawCursor(state.local.x, state.local.y, state.local.color, state.local.name, 1, true);
  }
  if (state.calibrationSession?.active) {
    drawCalibrationTarget(w, h);
  }
  if (state.trainerSession?.active) {
    drawTrainerTarget(w, h);
  }
}

function drawGrid(width, height) {
  const minor = 96;
  const major = minor * 2;
  ctx.lineWidth = 1;
  for (let x = 0; x <= width; x += minor) {
    ctx.strokeStyle = x % major === 0 ? "rgba(122, 170, 255, 0.18)" : "rgba(122, 170, 255, 0.1)";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y <= height; y += minor) {
    ctx.strokeStyle = y % major === 0 ? "rgba(122, 170, 255, 0.18)" : "rgba(122, 170, 255, 0.1)";
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function drawCursor(x, y, color, label, alpha, isLocal) {
  const px = x * elements.canvas.clientWidth;
  const py = y * elements.canvas.clientHeight;
  const rgb = `${color[0]}, ${color[1]}, ${color[2]}`;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.shadowBlur = isLocal ? 24 : 16;
  ctx.shadowColor = `rgba(${rgb}, 0.85)`;
  ctx.fillStyle = `rgba(${rgb}, ${isLocal ? 0.98 : 0.86})`;
  ctx.beginPath();
  ctx.arc(px, py, isLocal ? 9 : 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = `rgba(${rgb}, 0.72)`;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(px, py, isLocal ? 18 : 14, 0, Math.PI * 2);
  ctx.stroke();
  ctx.font = "13px Inter, ui-sans-serif, system-ui, sans-serif";
  ctx.textBaseline = "middle";
  const text = label || "Guest";
  const textWidth = ctx.measureText(text).width;
  const labelX = Math.min(px + 18, elements.canvas.clientWidth - textWidth - 18);
  const labelY = Math.max(18, py - 22);
  ctx.fillStyle = "rgba(5, 8, 13, 0.82)";
  roundRect(ctx, labelX - 8, labelY - 12, textWidth + 16, 24, 6);
  ctx.fill();
  ctx.fillStyle = `rgba(${rgb}, 0.98)`;
  ctx.fillText(text, labelX, labelY);
  ctx.restore();
}

function roundRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.arcTo(x + width, y, x + width, y + height, radius);
  context.arcTo(x + width, y + height, x, y + height, radius);
  context.arcTo(x, y + height, x, y, radius);
  context.arcTo(x, y, x + width, y, radius);
  context.closePath();
}

function drawCalibrationTarget(width, height) {
  const session = state.calibrationSession;
  if (!session?.active) {
    return;
  }
  const target = visibleOverlayTarget(CALIBRATION_TARGETS[session.stepIndex], ".calibration-card");
  if (!target) {
    return;
  }

  const px = target.x * width;
  const py = target.y * height;
  const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 180);

  ctx.save();
  ctx.shadowBlur = 28;
  ctx.shadowColor = "rgba(117, 216, 255, 0.75)";
  ctx.strokeStyle = "rgba(117, 216, 255, 0.92)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(px, py, 26 + pulse * 6, 0, Math.PI * 2);
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.fillStyle = "rgba(156, 255, 210, 0.98)";
  ctx.beginPath();
  ctx.arc(px, py, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(156, 255, 210, 0.68)";
  ctx.beginPath();
  ctx.moveTo(px - 18, py);
  ctx.lineTo(px + 18, py);
  ctx.moveTo(px, py - 18);
  ctx.lineTo(px, py + 18);
  ctx.stroke();
  ctx.restore();
}

function drawTrainerTarget(width, height) {
  const session = state.trainerSession;
  if (!session?.active) {
    return;
  }
  const target = visibleOverlayTarget(session.targets[session.stepIndex] || session.target, ".trainer-dock");
  if (!target) {
    return;
  }

  const px = target.x * width;
  const py = target.y * height;
  const now = performance.now();
  const phaseElapsed = now - (session.phaseStartedAt || now);
  const captureMs = session.mode === "trial" ? EVAL_CAPTURE_MS : TRAIN_CAPTURE_MS;
  const warmup = Math.min(1, phaseElapsed / captureMs);
  const isCapture = session.phase === "capture" || isWaveMode(session.mode);

  if (session.mode === "dojo") {
    drawDojoDummy(px, py, warmup, isCapture);
  } else if (session.mode === "trial") {
    drawTrialMark(px, py, warmup, isCapture);
  } else if (isWaveMode(session.mode)) {
    const dwellProgress = session.dwellStartedAt
      ? clamp01((now - session.dwellStartedAt) / CHALLENGE_DWELL_MS)
      : 0;
    drawEnemy(px, py, session.score, dwellProgress, session.mode);
  }
}

function drawDojoDummy(px, py, warmup, isCapture) {
  const radius = 24 + warmup * 8;
  const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 130);
  const stroke = isCapture ? "rgba(156, 255, 210, 0.95)" : "rgba(117, 216, 255, 0.94)";

  ctx.save();
  ctx.shadowBlur = 28;
  ctx.shadowColor = stroke;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(px, py, radius + pulse * 4, 0, Math.PI * 2);
  ctx.stroke();
  ctx.shadowBlur = 0;

  ctx.strokeStyle = "rgba(255, 213, 112, 0.58)";
  ctx.lineWidth = 6;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(px - 28, py + 10);
  ctx.lineTo(px + 28, py + 10);
  ctx.moveTo(px, py + 2);
  ctx.lineTo(px, py + 46);
  ctx.stroke();

  ctx.fillStyle = "rgba(174, 111, 58, 0.96)";
  ctx.strokeStyle = "rgba(255, 213, 112, 0.86)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(px, py - 12, 18, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "rgba(106, 66, 36, 0.92)";
  roundRect(ctx, px - 16, py + 6, 32, 34, 8);
  ctx.fill();

  drawTargetGlyph(px, py, radius, "rgba(156, 255, 210, 0.98)");
  ctx.restore();
}

function drawTrialMark(px, py, warmup, isCapture) {
  const radius = 20 + warmup * 14;
  const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 120);
  const stroke = isCapture ? "rgba(156, 255, 210, 0.96)" : "rgba(117, 216, 255, 0.94)";

  ctx.save();
  ctx.shadowBlur = 30;
  ctx.shadowColor = stroke;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(px, py, radius + pulse * 5, 0, Math.PI * 2);
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = "rgba(238, 243, 255, 0.58)";
  ctx.beginPath();
  ctx.arc(px, py, radius * 0.58, 0, Math.PI * 2);
  ctx.stroke();
  drawTargetGlyph(px, py, radius, "rgba(156, 255, 210, 0.98)");
  ctx.restore();
}

function drawEnemy(px, py, score, dwellProgress, mode) {
  const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 105);
  const radius = CHALLENGE_TARGET_RADIUS_PX;
  const accent = mode === "multiplayer" ? "rgba(255, 111, 145, 0.95)" : "rgba(255, 213, 112, 0.95)";

  ctx.save();
  ctx.shadowBlur = 32;
  ctx.shadowColor = accent;
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(px, py, radius + pulse * 5, 0, Math.PI * 2);
  ctx.stroke();
  ctx.shadowBlur = 0;

  ctx.fillStyle = "rgba(15, 20, 32, 0.98)";
  ctx.strokeStyle = "rgba(238, 243, 255, 0.3)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(px, py - 2, 25, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "rgba(238, 243, 255, 0.88)";
  roundRect(ctx, px - 16, py - 9, 32, 10, 5);
  ctx.fill();
  ctx.fillStyle = "rgba(5, 8, 13, 0.92)";
  ctx.beginPath();
  ctx.arc(px - 7, py - 4, 2.2, 0, Math.PI * 2);
  ctx.arc(px + 7, py - 4, 2.2, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "rgba(238, 243, 255, 0.7)";
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(px + 15, py + 17);
  ctx.lineTo(px + 36, py - 18);
  ctx.stroke();
  ctx.strokeStyle = "rgba(255, 213, 112, 0.82)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(px + 30, py - 26);
  ctx.lineTo(px + 39, py - 12);
  ctx.stroke();

  if (dwellProgress > 0) {
    ctx.strokeStyle = "rgba(156, 255, 210, 0.98)";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(px, py, radius + 10, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * dwellProgress);
    ctx.stroke();
  }

  drawTargetGlyph(px, py, radius, "rgba(255, 213, 112, 0.98)");
  ctx.font = "700 14px Inter, ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(5, 8, 13, 0.86)";
  ctx.fillText(String(score), px, py + 18);
  ctx.restore();
}

function drawTargetGlyph(px, py, radius, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(px, py, Math.max(5, radius * 0.16), 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(238, 243, 255, 0.58)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(px - radius * 0.64, py);
  ctx.lineTo(px + radius * 0.64, py);
  ctx.moveTo(px, py - radius * 0.64);
  ctx.lineTo(px, py + radius * 0.64);
  ctx.stroke();
}

async function estimateGaze(result) {
  const landmarks = result?.faceLandmarks?.[0];
  if (!landmarks) {
    return { tracking: false };
  }

  const featureFrame = buildFeatureFrame(landmarks, result);
  if (state.gazeModel) {
    try {
      return await predictWithGazeModel(featureFrame);
    } catch (error) {
      console.warn("ONNX gaze inference failed; using fallback.", error);
    }
  }
  return estimateHeuristicGaze(featureFrame);
}

function buildFeatureFrame(landmarks, result) {
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

async function predictWithGazeModel(featureFrame) {
  const leftEyeTensor = extractEyeCropTensor(
    featureFrame.landmarks,
    LEFT_EYE_CORNER_POINTS,
    LEFT_EYE_UPPER_LID_POINTS,
    LEFT_EYE_LOWER_LID_POINTS,
    false,
  );
  const rightEyeTensor = extractEyeCropTensor(
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
  const extraFeatureKeys = state.gazeModel.extraFeatureKeys || [];
  const extraFeatures = new Float32Array(
    extraFeatureKeys.map((key) =>
      featureValue(featureFrame.payload, key, EXTRA_FEATURE_DEFAULTS[key] ?? 0),
    ),
  );
  const feeds = {};
  if (state.gazeModel.inputNames.has("left_eye")) {
    feeds.left_eye = new state.gazeModel.ort.Tensor("float32", leftEyeTensor, [
      1,
      1,
      EYE_CROP_HEIGHT,
      EYE_CROP_WIDTH,
    ]);
  }
  if (state.gazeModel.inputNames.has("right_eye")) {
    feeds.right_eye = new state.gazeModel.ort.Tensor("float32", rightEyeTensor, [
      1,
      1,
      EYE_CROP_HEIGHT,
      EYE_CROP_WIDTH,
    ]);
  }
  if (state.gazeModel.inputNames.has("head_features")) {
    feeds.head_features = new state.gazeModel.ort.Tensor("float32", headFeatures, [
      1,
      HEAD_FEATURE_KEYS.length,
    ]);
  }
  if (state.gazeModel.inputNames.has("extra_features")) {
    feeds.extra_features = new state.gazeModel.ort.Tensor("float32", extraFeatures, [
      1,
      extraFeatureKeys.length,
    ]);
  }
  const outputs = await state.gazeModel.session.run(feeds);
  const gaze = outputs.gaze || outputs[state.gazeModel.session.outputNames[0]];
  const rawX = Number(gaze.data[0]);
  const rawY = Number(gaze.data[1]);
  return {
    tracking: true,
    kind: state.gazeModel.key,
    rawX,
    rawY,
    features: buildPersonalFeatureVector(featureFrame.payload, rawX, rawY),
  };
}

function estimateHeuristicGaze(featureFrame) {
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

function computeHeadPose(matrix) {
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

function extractEyeCropTensor(
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
  const sourceWidth = state.processingCanvas.width;
  const sourceHeight = state.processingCanvas.height;
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

  return affineCropToTensor(sourcePoints, flipHorizontal);
}

function affineCropToTensor(sourcePoints, flipHorizontal) {
  const frame = state.processingFrameData || state.processingCanvas
    .getContext("2d", { willReadFrequently: true })
    .getImageData(0, 0, state.processingCanvas.width, state.processingCanvas.height);
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

  const top = lerpGray(sampleGrayAt(frame, x0, y0), sampleGrayAt(frame, x1, y0), tx);
  const bottom = lerpGray(sampleGrayAt(frame, x0, y1), sampleGrayAt(frame, x1, y1), tx);
  return lerpGray(top, bottom, ty);
}

function sampleGrayAt(frame, x, y) {
  const index = (y * frame.width + x) * 4;
  const data = frame.data;
  return data[index] * 0.299 + data[index + 1] * 0.587 + data[index + 2] * 0.114;
}

function lerpGray(a, b, alpha) {
  return a * (1 - alpha) + b * alpha;
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

function averagePoint(samples) {
  let sumX = 0;
  let sumY = 0;
  for (const [x, y] of samples) {
    sumX += x;
    sumY += y;
  }
  return [sumX / Math.max(samples.length, 1), sumY / Math.max(samples.length, 1)];
}

function hasAnyCalibrationMapping() {
  return Object.values(state.calibration.mappings || {}).some(Boolean);
}

function getCalibrationMapping(kind) {
  return state.calibration.mappings?.[kind] || null;
}

function buildCalibrationMapping(kind, points) {
  const normalizedPoints = CALIBRATION_TARGETS.map((target) => {
    const point = points.find((item) => item.id === target.id);
    if (!point) {
      return null;
    }
    if (!Number.isFinite(point.rawX) || !Number.isFinite(point.rawY)) {
      return null;
    }
    return {
      id: target.id,
      rawX: Number(point.rawX),
      rawY: Number(point.rawY),
      targetX: Number(point.targetX ?? target.x),
      targetY: Number(point.targetY ?? target.y),
    };
  }).filter(Boolean);
  if (normalizedPoints.length !== CALIBRATION_TARGETS.length) {
    return null;
  }

  const affine = solveAffineTransform(normalizedPoints);
  if (!affine) {
    return null;
  }

  return {
    kind,
    version: CALIBRATION_VERSION,
    points: normalizedPoints,
    center: normalizedPoints.find((point) => point.id === "center") || null,
    affine,
  };
}

function normalizeCalibrationMapping(mapping, fallbackKind) {
  if (!mapping || !Array.isArray(mapping.points)) {
    return null;
  }
  if (mapping.version !== CALIBRATION_VERSION) {
    return null;
  }
  return buildCalibrationMapping(mapping.kind || fallbackKind || "onnx", mapping.points);
}

function solveAffineTransform(points) {
  const ata = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const atbx = [0, 0, 0];
  const atby = [0, 0, 0];

  for (const point of points) {
    const row = [point.rawX, point.rawY, 1];
    for (let r = 0; r < 3; r += 1) {
      atbx[r] += row[r] * point.targetX;
      atby[r] += row[r] * point.targetY;
      for (let c = 0; c < 3; c += 1) {
        ata[r][c] += row[r] * row[c];
      }
    }
  }

  const inverse = invert3x3(ata);
  if (!inverse) {
    return null;
  }
  return {
    x: multiply3x3Vec(inverse, atbx),
    y: multiply3x3Vec(inverse, atby),
  };
}

function invert3x3(matrix) {
  const [
    [a, b, c],
    [d, e, f],
    [g, h, i],
  ] = matrix;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const D = -(b * i - c * h);
  const E = a * i - c * g;
  const F = -(a * h - b * g);
  const G = b * f - c * e;
  const H = -(a * f - c * d);
  const I = a * e - b * d;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-8) {
    return null;
  }
  const invDet = 1 / det;
  return [
    [A * invDet, D * invDet, G * invDet],
    [B * invDet, E * invDet, H * invDet],
    [C * invDet, F * invDet, I * invDet],
  ];
}

function multiply3x3Vec(matrix, vector) {
  return matrix.map((row) => row[0] * vector[0] + row[1] * vector[1] + row[2] * vector[2]);
}

function mapPointWithCalibration(rawPoint, mapping) {
  const piecewise = mapPointWithPiecewiseTriangles(rawPoint, mapping.points);
  if (piecewise) {
    return piecewise;
  }
  return applyAffineTransform(mapping.affine, rawPoint);
}

function mapPointWithPiecewiseTriangles(rawPoint, points) {
  const triangles = [
    ["topLeft", "topRight", "center"],
    ["topRight", "bottomRight", "center"],
    ["bottomRight", "bottomLeft", "center"],
    ["bottomLeft", "topLeft", "center"],
  ];
  let best = null;

  for (const ids of triangles) {
    const source = ids.map((id) => findCalibrationPoint(points, id));
    if (source.some((point) => !point)) {
      continue;
    }
    const weights = barycentricWeights(
      rawPoint,
      [source[0].rawX, source[0].rawY],
      [source[1].rawX, source[1].rawY],
      [source[2].rawX, source[2].rawY],
    );
    if (!weights) {
      continue;
    }

    const minWeight = Math.min(...weights);
    if (minWeight >= CALIBRATION_INSIDE_TOLERANCE) {
      return combineTrianglePoint(weights, source, "targetX", "targetY");
    }
    if (!best || minWeight > best.minWeight) {
      best = { minWeight, weights, source };
    }
  }

  if (!best) {
    return null;
  }
  return combineTrianglePoint(best.weights, best.source, "targetX", "targetY");
}

function findCalibrationPoint(points, id) {
  return points.find((point) => point.id === id) || null;
}

function barycentricWeights(point, a, b, c) {
  const denominator =
    (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
  if (Math.abs(denominator) < 1e-8) {
    return null;
  }
  const w1 =
    ((b[1] - c[1]) * (point[0] - c[0]) + (c[0] - b[0]) * (point[1] - c[1])) /
    denominator;
  const w2 =
    ((c[1] - a[1]) * (point[0] - c[0]) + (a[0] - c[0]) * (point[1] - c[1])) /
    denominator;
  const w3 = 1 - w1 - w2;
  return [w1, w2, w3];
}

function combineTrianglePoint(weights, points, xKey, yKey) {
  const x =
    weights[0] * points[0][xKey] + weights[1] * points[1][xKey] + weights[2] * points[2][xKey];
  const y =
    weights[0] * points[0][yKey] + weights[1] * points[1][yKey] + weights[2] * points[2][yKey];
  return [x, y];
}

function applyAffineTransform(affine, rawPoint) {
  if (!affine?.x || !affine?.y) {
    return null;
  }
  const vector = [rawPoint[0], rawPoint[1], 1];
  return [
    affine.x[0] * vector[0] + affine.x[1] * vector[1] + affine.x[2],
    affine.y[0] * vector[0] + affine.y[1] * vector[1] + affine.y[2],
  ];
}

function calibrationTargetLabel(id) {
  switch (id) {
    case "center":
      return "center";
    case "topLeft":
      return "top-left";
    case "topRight":
      return "top-right";
    case "bottomRight":
      return "bottom-right";
    case "bottomLeft":
      return "bottom-left";
    default:
      return "target";
  }
}

function defaultRelayUrl() {
  if (location.protocol === "https:") {
    return `wss://${location.host}/ws`;
  }
  if (location.protocol === "http:") {
    return `ws://${location.host}/ws`;
  }
  return "ws://127.0.0.1:8765/ws";
}

function normalizeRelayUrl(value) {
  const raw = (value || defaultRelayUrl()).trim();
  if (raw.startsWith("https://")) {
    return `wss://${raw.slice("https://".length)}`;
  }
  if (raw.startsWith("http://")) {
    return `ws://${raw.slice("http://".length)}`;
  }
  if (raw.startsWith("ws://") || raw.startsWith("wss://")) {
    return raw;
  }
  return `ws://${raw}`;
}

function normalizeRoom(value) {
  const compact = (value || "").toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 12);
  if (!compact) {
    return "";
  }
  if (compact.length === 6) {
    return `${compact.slice(0, 3)}-${compact.slice(3)}`;
  }
  return compact;
}

function generateRoomCode() {
  const letters = "ABCDEFGHJKLMNPQRSTUVWXYZ";
  let prefix = "";
  for (let index = 0; index < 3; index += 1) {
    prefix += letters[Math.floor(Math.random() * letters.length)];
  }
  const suffix = String(Math.floor(Math.random() * 1000)).padStart(3, "0");
  return `${prefix}-${suffix}`;
}

function colorForName(name) {
  let hash = 0;
  for (let index = 0; index < name.length; index += 1) {
    hash = (hash * 31 + name.charCodeAt(index)) >>> 0;
  }
  const hue = (hash % 360) / 360;
  return hslToRgb(hue, 0.74, 0.62);
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

function refreshPersonalModelLabel() {
  const kind = state.modelKey;
  const sampleCount = samplesForKind(state.trainingSamples, kind).length;
  const model = state.personalModels[kind];
  const hasPersonalData = Boolean(model) || sampleCount > 0;
  elements.evaluateButton.disabled = !model;
  elements.challengeButton.disabled = !model;
  elements.multiplayerButton.disabled = !model;
  elements.resetPersonalButton.disabled = !hasPersonalData;
  const progress = clamp01(sampleCount / PERSONAL_MIN_SAMPLES);
  elements.personalProgressFill.style.width = `${Math.round(progress * 100)}%`;
  if (model) {
    const fit = Number.isFinite(model.fitMeanPx) ? `fit ${Math.round(model.fitMeanPx)} px` : "trained";
    const evalText = Number.isFinite(model.lastEvalMeanPx)
      ? ` · test ${Math.round(model.lastEvalMeanPx)} px`
      : "";
    elements.personalModelLabel.textContent = `${model.sampleCount || sampleCount} samples · ${fit}${evalText}`;
    if (Number.isFinite(model.lastEvalDeltaPx)) {
      const direction = model.lastEvalDeltaPx >= 0 ? "better than base" : "worse than base";
      elements.personalModelMeta.textContent = `${Math.abs(Math.round(model.lastEvalDeltaPx))} px ${direction}`;
    } else {
      elements.personalModelMeta.textContent = "Run Trial for held-out accuracy";
    }
    elements.personalProgressFill.style.width = "100%";
    return;
  }
  if (sampleCount > 0) {
    elements.personalModelLabel.textContent = `${sampleCount}/${PERSONAL_MIN_SAMPLES} samples`;
    elements.personalModelMeta.textContent = "Keep training";
    return;
  }
  elements.personalModelLabel.textContent = "No data";
  elements.personalModelMeta.textContent = "Local only";
}

function shuffledTargets(targets) {
  const copy = targets.map((target) => ({ ...target }));
  shuffleInPlace(copy, Math.random);
  return copy;
}

function randomEnemyTarget(previous) {
  let target = null;
  for (let attempt = 0; attempt < 8; attempt += 1) {
    target = {
      id: `enemy-${Date.now()}-${attempt}`,
      x: 0.12 + Math.random() * 0.76,
      y: 0.14 + Math.random() * 0.62,
    };
    if (!previous || Math.hypot(target.x - previous.x, target.y - previous.y) > 0.28) {
      break;
    }
  }
  return target;
}

function shuffleInPlace(items, random) {
  for (let index = items.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [items[index], items[swapIndex]] = [items[swapIndex], items[index]];
  }
  return items;
}

function distanceToTargetPx(point, target) {
  return distancePx(
    point.x,
    point.y,
    target.x,
    target.y,
    Math.max(1, window.innerWidth),
    Math.max(1, window.innerHeight),
  );
}

function visibleOverlayTarget(target, dockSelector) {
  if (!target) {
    return target;
  }
  const dock = document.querySelector(dockSelector);
  const dockHeight = dock?.getBoundingClientRect().height || 0;
  const viewportHeight = Math.max(1, window.innerHeight);
  const bottomReserved = dockHeight > 0 ? (dockHeight + 58) / viewportHeight : 0;
  const maxY = Math.max(0.56, 1 - bottomReserved);
  return {
    ...target,
    y: Math.min(target.y, maxY),
  };
}

function nextFrame() {
  return new Promise((resolve) => window.requestAnimationFrame(resolve));
}

function loadControlsHidden() {
  return localStorage.getItem("gazeGame.controlsHidden") === "1";
}

function defaultCalibration() {
  return {
    version: CALIBRATION_VERSION,
    centerX: 0.5,
    centerY: 0.5,
    gainX: 1.85,
    gainY: 1.75,
    mappings: {},
  };
}

function loadCalibration() {
  try {
    const saved = JSON.parse(localStorage.getItem("gazeGame.calibration") || "{}");
    const calibration = defaultCalibration();
    if (saved.version !== CALIBRATION_VERSION) {
      return calibration;
    }
    calibration.centerX = typeof saved.centerX === "number" ? saved.centerX : calibration.centerX;
    calibration.centerY = typeof saved.centerY === "number" ? saved.centerY : calibration.centerY;
    calibration.gainX = typeof saved.gainX === "number" ? saved.gainX : calibration.gainX;
    calibration.gainY = typeof saved.gainY === "number" ? saved.gainY : calibration.gainY;

    if (saved.mappings && typeof saved.mappings === "object") {
      for (const [kind, mapping] of Object.entries(saved.mappings)) {
        const normalized = normalizeCalibrationMapping(mapping, kind);
        if (normalized) {
          calibration.mappings[kind] = normalized;
        }
      }
    } else if (Array.isArray(saved.points)) {
      const normalized = normalizeCalibrationMapping(saved, saved.kind || "onnx");
      if (normalized) {
        calibration.mappings[normalized.kind] = normalized;
      }
    }
    return calibration;
  } catch {
    return defaultCalibration();
  }
}

function saveCalibration(calibration) {
  localStorage.setItem("gazeGame.calibration", JSON.stringify(calibration));
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function finiteOrDefault(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function featureValue(payload, key, fallback) {
  return finiteOrDefault(payload?.[key], fallback);
}

function signedPoseValue(value, fallback, sign) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return number * sign;
}

function normalizeModelKey(value) {
  return GAZE_MODELS[value] ? value : DEFAULT_GAZE_MODEL_KEY;
}

function modelConfig(modelKey) {
  return GAZE_MODELS[normalizeModelKey(modelKey)];
}

function isModelReading(kind) {
  return Boolean(GAZE_MODELS[kind]);
}

function lerp(a, b, alpha) {
  return a * (1 - alpha) + b * alpha;
}
