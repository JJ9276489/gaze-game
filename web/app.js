import {
  PERSONAL_MIN_SAMPLES,
  appendTrainingSamples,
  clearPersonalStatsForKind,
  loadPersonalModels,
  loadPersonalStats,
  loadTrainingSamples,
  personalStatsForKind,
  predictWithPersonalModel,
  recordTrainingSamples,
  samplesForKind,
  savePersonalModels,
  saveTrainingSamples,
  trainPersonalModel,
} from "./personal_model.js";
import {
  CHALLENGE_DURATION_MS,
  createWaveSeed,
  isMultiplayerWaveMode,
  isWaveMode,
  normalizeRelayWave,
} from "./game_logic.js";
import {
  CALIBRATION_TARGETS,
  buildCalibrationMapping,
  calibrationTargetLabel,
  getCalibrationMapping,
  hasAnyCalibrationMapping,
  loadCalibration,
  mapPointWithCalibration,
  saveCalibration,
} from "./calibration_logic.js";
import {
  RELAY_SEND_INTERVAL_MS,
  buildCursorMessage,
  buildWaveHitMessage,
  buildWaveStartMessage,
  connectRelaySocket,
  defaultRelayUrl,
  normalizeRelayUrl,
  relayIsOpen,
  sendRelayMessage,
} from "./relay_client.js";
import {
  DEFAULT_GAZE_MODEL_KEY,
  drawMirroredVideoFrame,
  estimateGaze,
  gazeModelLabel,
  isModelReading,
  loadFaceLandmarker,
  loadGazeModel,
  modelConfig,
  normalizeModelKey,
} from "./gaze_runtime.js";
import { renderStage } from "./renderer.js";
import {
  CHALLENGE_DWELL_MS,
  CHALLENGE_TARGET_RADIUS_PX,
  TRAIN_CAPTURE_MS,
  advanceTrainerSession,
  createTrainerSession,
  currentTrainerTarget,
  trainerModeLabel,
  trainerOverlayView,
  trainerWaveScoreMap,
} from "./trainer_session.js";
import {
  colorForName,
  controlsHiddenView,
  generateRoomCode,
  hudContextView,
  normalizePlayerName,
  normalizeRoom,
  personalModelView,
} from "./ui_state.js";

const CALIBRATION_SETTLE_MS = 600;
const CALIBRATION_CAPTURE_MS = 900;
const CALIBRATION_MIN_SAMPLES = 10;
const DEBUG_REFRESH_MS = 250;
const DEBUG_SAMPLE_LIMIT = 180;

const $ = (id) => document.getElementById(id);

const elements = {
  canvas: $("stageCanvas"),
  video: $("camera"),
  lobby: $("lobby"),
  hud: $("hud"),
  form: $("joinForm"),
  dojoButton: $("dojoButton"),
  createButton: $("createButton"),
  joinButton: $("joinButton"),
  trainButton: $("trainButton"),
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
  roomScopeLabel: $("roomScopeLabel"),
  roomLabel: $("roomLabel"),
  copyRoomButton: $("copyRoomButton"),
  fullscreenButton: $("fullscreenButton"),
  toggleControlsButton: $("toggleControlsButton"),
  debugButton: $("debugButton"),
  exportDebugButton: $("exportDebugButton"),
  debugPanel: $("debugPanel"),
  debugMetrics: $("debugMetrics"),
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
const initialTrainingSamples = loadTrainingSamples();

const state = {
  animationFrame: 0,
  cameraFrame: 0,
  sendTimer: 0,
  ws: null,
  sessionMode: "lobby",
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
  trainingSamples: initialTrainingSamples,
  personalModels: loadPersonalModels(),
  personalStats: loadPersonalStats(initialTrainingSamples),
  trainerSession: null,
  waveScores: new Map(),
  pendingWave: null,
  controlsHidden: loadControlsHidden(),
  debug: {
    enabled: loadDebugEnabled(),
    frameCount: 0,
    fpsStartedAt: performance.now(),
    fps: 0,
    inferenceCount: 0,
    inferenceFailures: 0,
    lastInferenceMs: 0,
    avgInferenceMs: 0,
    lastDebugRefreshAt: 0,
    lastError: "",
    lastPrediction: null,
    samples: [],
  },
  processingCanvas: document.createElement("canvas"),
};

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
  setDebugVisible(state.debug.enabled, false);
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
  elements.dojoButton.addEventListener("click", () => {
    void startLocalDojoSession();
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
  elements.debugButton.addEventListener("click", () => {
    setDebugVisible(!state.debug.enabled);
  });
  elements.exportDebugButton.addEventListener("click", () => {
    void exportDebugLog();
  });
  elements.trainButton.addEventListener("click", () => {
    void startTrainerRun("dojo");
  });
  elements.challengeButton.addEventListener("click", () => {
    void startTrainerRun("solo");
  });
  elements.multiplayerButton.addEventListener("click", () => {
    void requestMultiplayerWaveFromUser();
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

async function startLocalDojoSession() {
  setBusy(true);
  hideToast();
  state.source = "gaze";
  elements.mouseModeInput.checked = false;

  const name = normalizePlayerName(elements.nameInput.value);
  state.local.name = name;
  state.local.room = "DOJO";
  state.sessionMode = "dojo";
  state.local.color = colorForName(name);
  elements.roomLabel.textContent = "DOJO";
  elements.roomLabel.title = "Local Dojo";
  elements.nameLabel.textContent = name;
  localStorage.setItem("gazeGame.name", name);

  try {
    setStatus("Starting Dojo");
    await startGazeSource();
    state.connected = false;
    state.running = true;
    showHud();
    showToast("Dojo ready. Calibrate, then start Dojo.");
    window.setTimeout(hideToast, 1800);
  } catch (error) {
    stopSession();
    const message = error instanceof Error ? error.message : String(error);
    setStatus("Could not start Dojo");
    showToast(message);
  } finally {
    setBusy(false);
  }
}

async function startSession(createRoom) {
  setBusy(true);
  hideToast();
  state.source = elements.mouseModeInput.checked ? "mouse" : "gaze";

  const name = normalizePlayerName(elements.nameInput.value);
  const room = normalizeRoom(elements.roomInput.value || (createRoom ? generateRoomCode() : ""));
  if (!room) {
    setBusy(false);
    setStatus("Room code required");
    return;
  }

  state.local.name = name;
  state.local.room = room;
  state.sessionMode = "room";
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
    if (state.pendingWave) {
      const wave = state.pendingWave;
      state.pendingWave = null;
      handleWaveStart(wave);
    }
    if (state.source === "gaze" && !hasAnyCalibrationMapping(state.calibration)) {
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
  state.sessionMode = "lobby";
  state.connected = false;
  state.peers.clear();
  state.waveScores.clear();
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
  elements.dojoButton.disabled = isBusy;
  elements.createButton.disabled = isBusy;
  elements.joinButton.disabled = isBusy;
}

function showHud() {
  elements.lobby.classList.add("hidden");
  elements.hud.classList.remove("hidden");
  syncHudContext();
  syncHudSuppression();
}

function showLobby() {
  elements.hud.classList.add("hidden");
  elements.hud.classList.remove("hud-suppressed");
  elements.lobby.classList.remove("hidden");
  syncHudContext();
}

function syncHudContext() {
  const view = hudContextView(state.sessionMode);
  elements.roomScopeLabel.textContent = view.roomScopeLabel;
  elements.copyRoomButton.classList.toggle("hidden", view.copyRoomHidden);
  elements.trainButton.classList.toggle("hidden", view.trainHidden);
  elements.trainButton.textContent = view.trainText;
  elements.challengeButton.classList.toggle("hidden", view.challengeHidden);
  elements.multiplayerButton.classList.toggle("hidden", view.multiplayerHidden);
  elements.resetPersonalButton.classList.toggle("hidden", view.resetPersonalHidden);
  refreshPersonalModelLabel();
}

function syncHudSuppression() {
  const isTargetRunActive = Boolean(state.calibrationSession?.active || state.trainerSession?.active);
  elements.hud.classList.toggle("hud-suppressed", isTargetRunActive);
}

function setControlsHidden(hidden, persist = true) {
  state.controlsHidden = Boolean(hidden);
  const view = controlsHiddenView(state.controlsHidden);
  elements.hud.classList.toggle("hud-controls-hidden", view.hidden);
  elements.toggleControlsButton.textContent = view.buttonText;
  elements.toggleControlsButton.setAttribute("aria-expanded", view.ariaExpanded);
  elements.toggleControlsButton.setAttribute("aria-pressed", view.ariaPressed);
  elements.toggleControlsButton.setAttribute("aria-label", view.ariaLabel);
  if (persist) {
    localStorage.setItem("gazeGame.controlsHidden", view.storageValue);
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
  } else if (key === "d") {
    event.preventDefault();
    setDebugVisible(!state.debug.enabled);
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
        recordInferenceFailureDebug(error);
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
  const startedAt = performance.now();
  const frame = drawMirroredVideoFrame(elements.video, state.processingCanvas);
  const result = state.faceLandmarker.detectForVideo(state.processingCanvas, performance.now());
  const reading = await estimateGaze(result, {
    gazeModel: state.gazeModel,
    frame,
    onModelError(error) {
      console.warn("ONNX gaze inference failed; using fallback.", error);
      recordInferenceFailureDebug(error);
    },
  });
  recordInferenceDebug(performance.now() - startedAt);
  return reading;
}

function applyGazeReading(reading) {
  if (!reading?.tracking) {
    state.local.tracking = false;
    state.rawReading = null;
    state.debug.lastPrediction = {
      tracking: false,
      method: "none",
      rawX: null,
      rawY: null,
      x: state.local.x,
      y: state.local.y,
      kind: state.modelKey,
    };
    recordDebugSample({ event: "gaze", tracking: false, method: "none" });
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
  state.debug.lastPrediction = {
    tracking: true,
    method: predicted.method,
    rawX: reading.rawX,
    rawY: reading.rawY,
    x: state.local.x,
    y: state.local.y,
    kind: reading.kind,
  };
  recordDebugSample({
    event: "gaze",
    tracking: true,
    kind: reading.kind,
    method: predicted.method,
    rawX: reading.rawX,
    rawY: reading.rawY,
    x: state.local.x,
    y: state.local.y,
  });
  updateTrainerRun(reading, {
    active: {
      x: state.local.x,
      y: state.local.y,
      method: predicted.method,
    },
  });
  setTrackingStatus(trackingStatusFor(reading, predicted.method));
}

function calibrateReading(reading) {
  const mapping = getCalibrationMapping(state.calibration, reading.kind);
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
  const label = gazeModelLabel(reading.kind);
  if (label) {
    if (method === "personal") return `${label} + personal NN`;
    if (method === "calibration") return `${label} calibrated`;
    return label;
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

async function requestMultiplayerWaveFromUser() {
  if (!state.connected || !relayIsOpen(state.ws)) {
    showToast("Join or create a room before starting multiplayer.");
    return;
  }
  if (state.source !== "gaze" || !state.running) {
    return;
  }
  if (!state.rawReading?.tracking) {
    showToast("Wait for face tracking first.");
    return;
  }

  const kind = state.rawReading.kind || state.modelKey;
  if (!state.personalModels[kind]) {
    showToast("Enter the Dojo and train your personal NN first.");
    return;
  }

  hideToast();
  await maybeEnterFullscreen();
  const seed = createWaveSeed();
  sendRelayMessage(
    state.ws,
    buildWaveStartMessage({
      room: state.local.room,
      seed,
      durationMs: CHALLENGE_DURATION_MS,
    }),
  );
}

async function startTrainerRun(mode, options = {}) {
  if (state.source !== "gaze" || !state.running) {
    return;
  }
  if (!state.rawReading?.tracking) {
    showToast("Wait for face tracking first.");
    return;
  }

  hideToast();
  if (!options.skipFullscreen) {
    await maybeEnterFullscreen();
  }
  cancelCalibration(false);

  const kind = state.rawReading.kind || state.modelKey;
  if (isWaveMode(mode) && !state.personalModels[kind]) {
    showToast("Enter the Dojo and train your personal NN first.");
    return;
  }
  const wave = options.wave || null;
  state.waveScores.clear();
  for (const score of trainerWaveScoreMap(wave).values()) {
    state.waveScores.set(score.id, score);
  }
  state.trainerSession = createTrainerSession({
    mode,
    kind,
    wave,
    now: performance.now(),
    wallNow: Date.now(),
  });
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
  const result = advanceTrainerSession(session, {
    reading,
    activePoint: point.active,
    now: performance.now(),
    viewportWidth: window.innerWidth,
    viewportHeight: window.innerHeight,
    targetMapper: trainerTargetForOverlay,
  });

  if (result.type === "kind_mismatch") {
    elements.trainerBody.textContent = result.message;
    return;
  }
  if (result.hitTarget && session.mode === "multiplayer") {
    sendWaveHit(session, result.hitTarget);
  }
  if (result.complete) {
    finishTrainerRun(session);
    return;
  }
  if (result.refresh) {
    refreshTrainerOverlay();
  }
}

function trainerTargetForOverlay(target) {
  return visibleOverlayTarget(target, ".trainer-dock");
}

function sendWaveHit(session, target) {
  if (!state.connected || !relayIsOpen(state.ws)) {
    return;
  }
  sendRelayMessage(
    state.ws,
    buildWaveHitMessage({
      room: state.local.room,
      waveId: session.waveId,
      targetId: target?.id || "",
      score: session.score,
    }),
  );
}

function finishTrainerRun(session) {
  state.trainerSession = null;
  refreshTrainerOverlay();
  if (session.mode === "dojo") {
    void finishTrainingRun(session);
    return;
  }
  showToast(`${trainerModeLabel(session.mode)} complete: ${session.score} takedowns.`);
  window.setTimeout(hideToast, 1800);
}

async function finishTrainingRun(session) {
  const trainingUpdate = appendTrainingSamples(state.trainingSamples, session.capturedSamples);
  state.trainingSamples = saveTrainingSamples(trainingUpdate.samples);
  const samples = samplesForKind(state.trainingSamples, session.kind);
  state.personalStats = recordTrainingSamples(
    state.personalStats,
    session.kind,
    trainingUpdate.addedCount,
    samples.length,
  );
  if (trainingUpdate.addedCount < 1) {
    refreshPersonalModelLabel();
    showToast("No usable samples collected.");
    return;
  }

  refreshPersonalModelLabel();
  if (samples.length < PERSONAL_MIN_SAMPLES) {
    showToast(`Saved ${samples.length} samples. Need ${PERSONAL_MIN_SAMPLES} to train.`);
    return;
  }

  showToast(`Training personal NN from ${samples.length} Dojo samples...`);
  await nextFrame();
  try {
    const model = trainPersonalModel(session.kind, samples);
    const stats = personalStatsForKind(state.personalStats, session.kind, state.trainingSamples);
    model.totalSampleCount = stats.totalSamples;
    model.retainedSampleCount = samples.length;
    state.personalModels[session.kind] = model;
    savePersonalModels(state.personalModels);
    refreshPersonalModelLabel();
    showToast(
      `Dojo complete: ${stats.totalSamples} total samples, ${Math.round(model.fitMeanPx)} px fit.`,
    );
    window.setTimeout(hideToast, 2200);
  } catch (error) {
    console.warn("Personal NN training failed", error);
    showToast("Personal NN training failed.");
  }
}

function refreshTrainerOverlay() {
  const view = trainerOverlayView(state.trainerSession, { now: performance.now() });
  syncHudSuppression();
  if (view.hidden) {
    elements.trainerOverlay.classList.add("hidden");
    elements.trainerProgressFill.style.width = "0%";
    return;
  }

  elements.trainerOverlay.classList.remove("hidden");
  elements.trainerProgressFill.style.width = `${view.progressPercent}%`;
  elements.trainerStep.textContent = view.step;
  elements.trainerTitle.textContent = view.title;
  elements.trainerBody.textContent = view.body;
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
  state.personalStats = clearPersonalStatsForKind(
    state.personalStats,
    kind,
    samplesForKind(state.trainingSamples, kind).length,
  );
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
  state.debug.lastPrediction = {
    tracking: true,
    method: "mouse",
    rawX: null,
    rawY: null,
    x: state.local.x,
    y: state.local.y,
    kind: "mouse",
  };
  recordDebugSample({
    event: "mouse",
    tracking: true,
    method: "mouse",
    x: state.local.x,
    y: state.local.y,
  });
}

function connectRelay() {
  const relayUrl = normalizeRelayUrl(elements.relayInput.value);
  elements.relayInput.value = relayUrl;
  localStorage.setItem("gazeGame.relay", relayUrl);

  return connectRelaySocket({
    url: relayUrl,
    room: state.local.room,
    name: state.local.name,
    color: state.local.color,
    onWelcome(message) {
      state.local.id = message.id;
      state.connected = true;
      state.pendingWave = normalizeRelayWave(message.wave);
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
    },
    onMessage: handleRelayMessage,
    onDisconnect() {
      if (state.running) {
        state.connected = false;
        setTrackingStatus("Disconnected");
        showToast("Relay disconnected");
      }
    },
  }).then((ws) => {
    state.ws = ws;
  });
}

function handleRelayMessage(message) {
  if (message.type === "error") {
    showToast(`Relay error: ${message.message || "unknown_error"}`);
    window.setTimeout(hideToast, 1800);
    return;
  }

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
    return;
  }

  if (message.type === "wave_start") {
    handleWaveStart(normalizeRelayWave(message));
    return;
  }

  if (message.type === "wave_score") {
    handleWaveScore(message);
  }
}

function handleWaveStart(wave) {
  if (!wave) {
    return;
  }
  if (state.trainerSession?.active) {
    if (state.trainerSession.waveId !== wave.id) {
      showToast(`${wave.startedByName || "A player"} started a wave.`);
      window.setTimeout(hideToast, 1600);
    }
    return;
  }
  if (!state.running || state.source !== "gaze") {
    return;
  }
  const kind = state.rawReading?.kind || state.modelKey;
  if (!state.personalModels[kind]) {
    showToast("Multiplayer wave started. Finish Dojo training to join waves.");
    window.setTimeout(hideToast, 2200);
    return;
  }
  if (!state.rawReading?.tracking) {
    showToast("Multiplayer wave started. Wait for face tracking, then start the next wave.");
    window.setTimeout(hideToast, 2200);
    return;
  }
  void startTrainerRun("multiplayer", { wave, skipFullscreen: true });
}

function handleWaveScore(message) {
  const session = state.trainerSession;
  if (!session?.active || session.mode !== "multiplayer" || message.wave_id !== session.waveId) {
    return;
  }
  const id = String(message.id || "");
  if (!id) {
    return;
  }
  const score = {
    id,
    name: message.name || "Guest",
    color: Array.isArray(message.color) ? message.color : [255, 255, 255],
    score: Math.max(0, Number(message.score) || 0),
  };
  state.waveScores.set(id, score);
  if (id === state.local.id) {
    session.score = score.score;
  }
}

function startSending() {
  window.clearInterval(state.sendTimer);
  state.sendTimer = window.setInterval(sendCursor, RELAY_SEND_INTERVAL_MS);
}

function sendCursor() {
  if (!state.connected || !relayIsOpen(state.ws)) {
    return;
  }
  const calibrationActive = Boolean(state.calibrationSession?.active);
  const trainerHidden = Boolean(
    state.trainerSession?.active && !isMultiplayerWaveMode(state.trainerSession.mode),
  );
  state.seq += 1;
  sendRelayMessage(
    state.ws,
    buildCursorMessage({
      room: state.local.room,
      x: state.local.tracking && !calibrationActive && !trainerHidden ? state.local.x : null,
      y: state.local.tracking && !calibrationActive && !trainerHidden ? state.local.y : null,
      tracking: state.local.tracking && !calibrationActive && !trainerHidden,
      seq: state.seq,
    }),
  );
}

function drawLoop() {
  drawStage();
  refreshDebugPanel();
  state.animationFrame = window.requestAnimationFrame(drawLoop);
}

function drawStage() {
  renderStage({
    canvas: elements.canvas,
    context: ctx,
    dpr: Math.max(window.devicePixelRatio || 1, 1),
    now: performance.now(),
    peers: state.peers.values(),
    local: state.local,
    calibrationActive: Boolean(state.calibrationSession?.active),
    calibrationTarget: state.calibrationSession?.active
      ? visibleOverlayTarget(CALIBRATION_TARGETS[state.calibrationSession.stepIndex], ".calibration-card")
      : null,
    trainerSession: state.trainerSession,
    trainerTarget: state.trainerSession?.active
      ? currentTrainerTarget(state.trainerSession, trainerTargetForOverlay)
      : null,
    waveScores: state.waveScores,
    trainCaptureMs: TRAIN_CAPTURE_MS,
    challengeDwellMs: CHALLENGE_DWELL_MS,
    challengeTargetRadiusPx: CHALLENGE_TARGET_RADIUS_PX,
  });
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

function refreshPersonalModelLabel() {
  const kind = state.modelKey;
  const view = personalModelView({
    kind,
    trainingSamples: state.trainingSamples,
    personalStats: state.personalStats,
    personalModels: state.personalModels,
    isDojo: state.sessionMode === "dojo",
  });
  elements.challengeButton.disabled = view.challengeDisabled;
  elements.multiplayerButton.disabled = view.multiplayerDisabled;
  elements.resetPersonalButton.disabled = view.resetDisabled;
  elements.personalProgressFill.style.width = `${view.progressPercent}%`;
  elements.personalModelLabel.textContent = view.label;
  elements.personalModelMeta.textContent = view.meta;
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

function loadDebugEnabled() {
  const debugParam = new URLSearchParams(location.search).get("debug");
  if (debugParam === "1" || debugParam === "true") {
    return true;
  }
  if (debugParam === "0" || debugParam === "false") {
    return false;
  }
  return localStorage.getItem("gazeGame.debug") === "1";
}

function loadControlsHidden() {
  return localStorage.getItem("gazeGame.controlsHidden") === "1";
}

function setDebugVisible(visible, persist = true) {
  state.debug.enabled = Boolean(visible);
  elements.debugPanel.classList.toggle("hidden", !state.debug.enabled);
  elements.debugButton.setAttribute("aria-expanded", String(state.debug.enabled));
  elements.debugButton.setAttribute("aria-pressed", String(state.debug.enabled));
  if (persist) {
    localStorage.setItem("gazeGame.debug", state.debug.enabled ? "1" : "0");
  }
  refreshDebugPanel(true);
}

function recordInferenceDebug(durationMs) {
  const debug = state.debug;
  const now = performance.now();
  debug.inferenceCount += 1;
  debug.frameCount += 1;
  debug.lastInferenceMs = durationMs;
  debug.avgInferenceMs =
    debug.avgInferenceMs > 0 ? debug.avgInferenceMs * 0.86 + durationMs * 0.14 : durationMs;
  const windowMs = now - debug.fpsStartedAt;
  if (windowMs >= 1000) {
    debug.fps = (debug.frameCount * 1000) / windowMs;
    debug.frameCount = 0;
    debug.fpsStartedAt = now;
  }
}

function recordInferenceFailureDebug(error) {
  state.debug.inferenceFailures += 1;
  state.debug.lastError = error instanceof Error ? error.message : String(error || "unknown");
  recordDebugSample({
    event: "error",
    message: state.debug.lastError,
  });
}

function recordDebugSample(sample) {
  state.debug.samples.push({
    ts: new Date().toISOString(),
    ...sample,
  });
  if (state.debug.samples.length > DEBUG_SAMPLE_LIMIT) {
    state.debug.samples.splice(0, state.debug.samples.length - DEBUG_SAMPLE_LIMIT);
  }
}

function refreshDebugPanel(force = false) {
  if (!state.debug.enabled) {
    return;
  }
  const now = performance.now();
  if (!force && now - state.debug.lastDebugRefreshAt < DEBUG_REFRESH_MS) {
    return;
  }
  state.debug.lastDebugRefreshAt = now;
  const rows = debugMetricRows();
  const fragment = document.createDocumentFragment();
  for (const [label, value] of rows) {
    const term = document.createElement("dt");
    const details = document.createElement("dd");
    term.textContent = label;
    details.textContent = value;
    fragment.append(term, details);
  }
  elements.debugMetrics.replaceChildren(fragment);
}

function debugMetricRows() {
  const prediction = state.debug.lastPrediction || {};
  const rawPoint =
    prediction.rawX == null || prediction.rawY == null
      ? "n/a"
      : `${formatNumber(prediction.rawX)}, ${formatNumber(prediction.rawY)}`;
  const cursorPoint = `${formatNumber(state.local.x)}, ${formatNumber(state.local.y)}`;
  const calibration = getCalibrationMapping(state.calibration, prediction.kind || state.modelKey);
  const samples = samplesForKind(state.trainingSamples, state.modelKey).length;
  return [
    ["Status", elements.trackingLabel.textContent || "Ready"],
    ["Session", `${state.sessionMode}/${state.source}`],
    ["Relay", state.connected ? "connected" : "local"],
    ["Peers", String(state.peers.size)],
    ["Model", `${modelConfig(state.modelKey).label}${state.gazeModel ? "" : " fallback"}`],
    ["Reading", prediction.kind || "none"],
    ["Method", prediction.method || "none"],
    ["Tracking", state.local.tracking ? "yes" : "no"],
    ["Cursor", cursorPoint],
    ["Raw", rawPoint],
    ["FPS", formatNumber(state.debug.fps)],
    ["Infer ms", formatNumber(state.debug.lastInferenceMs)],
    ["Avg ms", formatNumber(state.debug.avgInferenceMs)],
    ["Failures", String(state.debug.inferenceFailures)],
    ["Video", `${elements.video.videoWidth || 0}x${elements.video.videoHeight || 0}`],
    ["Calib", calibration ? "saved" : "none"],
    ["Samples", String(samples)],
    ["Personal", state.personalModels[state.modelKey] ? "trained" : "none"],
    ["Last err", state.debug.lastError || "none"],
  ];
}

async function exportDebugLog() {
  const report = buildDebugReport();
  const text = JSON.stringify(report, null, 2);
  downloadTextFile(text, `gaze-ninja-debug-${Date.now()}.json`);
  try {
    await navigator.clipboard?.writeText(text);
    showToast("Debug log downloaded and copied.");
  } catch {
    showToast("Debug log downloaded.");
  }
  window.setTimeout(hideToast, 1600);
}

function buildDebugReport() {
  const prediction = state.debug.lastPrediction || {};
  const calibration = getCalibrationMapping(state.calibration, prediction.kind || state.modelKey);
  const relay = summarizeRelayUrl(elements.relayInput.value);
  return {
    generatedAt: new Date().toISOString(),
    app: {
      name: "Gaze Ninja",
      url: `${location.origin}${location.pathname}`,
      secureContext: window.isSecureContext,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
        dpr: window.devicePixelRatio || 1,
      },
      userAgent: navigator.userAgent,
    },
    runtime: {
      source: state.source,
      sessionMode: state.sessionMode,
      connected: state.connected,
      relay,
      roomCodePresent: Boolean(state.local.room),
      roomCodeLength: state.local.room.length,
      peerCount: state.peers.size,
      runtimeAssets: "local vendor",
    },
    video: {
      width: elements.video.videoWidth || 0,
      height: elements.video.videoHeight || 0,
      readyState: elements.video.readyState,
      tracks: state.mediaStream
        ? state.mediaStream.getVideoTracks().map((track) => ({
            label: track.label,
            readyState: track.readyState,
            enabled: track.enabled,
            settings: safeTrackSettings(track),
          }))
        : [],
    },
    model: {
      selectedKey: state.modelKey,
      selectedLabel: modelConfig(state.modelKey).label,
      loaded: Boolean(state.gazeModel),
      activeReadingKind: prediction.kind || null,
      status: elements.trackingLabel.textContent || "",
      calibrationSaved: Boolean(calibration),
      personalTrained: Boolean(state.personalModels[state.modelKey]),
      retainedSamples: samplesForKind(state.trainingSamples, state.modelKey).length,
      totalSamples:
        personalStatsForKind(state.personalStats, state.modelKey, state.trainingSamples)
          .totalSamples || 0,
    },
    cursor: {
      tracking: state.local.tracking,
      method: prediction.method || "none",
      x: state.local.x,
      y: state.local.y,
      rawX: prediction.rawX ?? null,
      rawY: prediction.rawY ?? null,
    },
    metrics: {
      fps: state.debug.fps,
      lastInferenceMs: state.debug.lastInferenceMs,
      avgInferenceMs: state.debug.avgInferenceMs,
      inferenceCount: state.debug.inferenceCount,
      inferenceFailures: state.debug.inferenceFailures,
      lastError: state.debug.lastError,
    },
    recentSamples: state.debug.samples.slice(-DEBUG_SAMPLE_LIMIT),
  };
}

function downloadTextFile(text, filename) {
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function summarizeRelayUrl(value) {
  try {
    const url = new URL(normalizeRelayUrl(value));
    return {
      protocol: url.protocol,
      host: url.host,
      pathname: url.pathname,
    };
  } catch {
    return { host: "invalid" };
  }
}

function safeTrackSettings(track) {
  try {
    const settings = track.getSettings();
    return {
      width: settings.width,
      height: settings.height,
      frameRate: settings.frameRate,
      facingMode: settings.facingMode,
      deviceIdPresent: Boolean(settings.deviceId),
    };
  } catch {
    return {};
  }
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "0.00";
  }
  return number.toFixed(2);
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function lerp(a, b, alpha) {
  return a * (1 - alpha) + b * alpha;
}
