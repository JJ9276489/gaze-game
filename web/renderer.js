import { isWaveMode } from "./game_logic.js";

const BACKGROUND = "#05080d";
const FONT_STACK = "Inter, ui-sans-serif, system-ui, sans-serif";

export function renderStage({
  canvas,
  context,
  dpr = 1,
  now = performance.now(),
  peers = [],
  local,
  calibrationActive = false,
  calibrationTarget = null,
  trainerSession = null,
  trainerTarget = null,
  waveScores = new Map(),
  trainCaptureMs,
  challengeDwellMs,
  challengeTargetRadiusPx,
}) {
  const { width, height } = resizeCanvasForDpr(canvas, context, dpr);
  context.clearRect(0, 0, width, height);
  context.fillStyle = BACKGROUND;
  context.fillRect(0, 0, width, height);
  drawGrid(context, width, height);

  for (const peer of peers) {
    if (peer.x === null || peer.y === null) {
      continue;
    }
    const age = now - peer.lastSeen;
    const alpha = peer.tracking ? Math.max(0.25, 1 - age / 3000) : 0.2;
    drawCursor(context, width, height, peer.x, peer.y, peer.color, peer.name, alpha, false);
  }

  if (shouldShowLocalCursor({ local, calibrationActive, trainerSession })) {
    drawCursor(context, width, height, local.x, local.y, local.color, local.name, 1, true);
  }
  if (calibrationTarget) {
    drawCalibrationTarget(context, width, height, calibrationTarget, now);
  }
  if (trainerSession?.active) {
    drawTrainerTarget(context, width, height, {
      session: trainerSession,
      target: trainerTarget,
      now,
      trainCaptureMs,
      challengeDwellMs,
      challengeTargetRadiusPx,
    });
    drawWaveLeaderboard(
      context,
      width,
      buildLeaderboardRows({ waveScores, local, session: trainerSession }),
    );
  }
}

export function resizeCanvasForDpr(canvas, context, dpr = 1) {
  const normalizedDpr = Math.max(dpr || 1, 1);
  const backingWidth = Math.floor(canvas.clientWidth * normalizedDpr);
  const backingHeight = Math.floor(canvas.clientHeight * normalizedDpr);
  if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
    canvas.width = backingWidth;
    canvas.height = backingHeight;
  }
  context.setTransform(normalizedDpr, 0, 0, normalizedDpr, 0, 0);
  return {
    width: canvas.clientWidth,
    height: canvas.clientHeight,
    backingWidth,
    backingHeight,
    dpr: normalizedDpr,
  };
}

export function shouldShowLocalCursor({ local, calibrationActive, trainerSession }) {
  const trainerHidesCursor = Boolean(trainerSession?.active && !isWaveMode(trainerSession.mode));
  return Boolean(local?.tracking && !calibrationActive && !trainerHidesCursor);
}

export function buildLeaderboardRows({ waveScores, local, session, limit = 5 }) {
  if (!session?.active || session.mode !== "multiplayer") {
    return [];
  }
  const scores = new Map(waveScores || []);
  scores.set(local.id, {
    id: local.id,
    name: local.name,
    color: local.color,
    score: session.score,
  });
  return [...scores.values()]
    .sort(
      (a, b) =>
        b.score - a.score ||
        String(a.name || "Guest").localeCompare(String(b.name || "Guest")),
    )
    .slice(0, limit)
    .map((row) => (row.id === local.id ? { ...row, label: "You" } : row));
}

function drawGrid(context, width, height) {
  const minor = 96;
  const major = minor * 2;
  context.lineWidth = 1;
  for (let x = 0; x <= width; x += minor) {
    context.strokeStyle =
      x % major === 0 ? "rgba(122, 170, 255, 0.18)" : "rgba(122, 170, 255, 0.1)";
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
  for (let y = 0; y <= height; y += minor) {
    context.strokeStyle =
      y % major === 0 ? "rgba(122, 170, 255, 0.18)" : "rgba(122, 170, 255, 0.1)";
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
}

function drawCursor(context, width, height, x, y, color, label, alpha, isLocal) {
  const px = x * width;
  const py = y * height;
  const rgb = `${color[0]}, ${color[1]}, ${color[2]}`;
  context.save();
  context.globalAlpha = alpha;
  context.shadowBlur = isLocal ? 24 : 16;
  context.shadowColor = `rgba(${rgb}, 0.85)`;
  context.fillStyle = `rgba(${rgb}, ${isLocal ? 0.98 : 0.86})`;
  context.beginPath();
  context.arc(px, py, isLocal ? 9 : 7, 0, Math.PI * 2);
  context.fill();
  context.shadowBlur = 0;
  context.strokeStyle = `rgba(${rgb}, 0.72)`;
  context.lineWidth = 2;
  context.beginPath();
  context.arc(px, py, isLocal ? 18 : 14, 0, Math.PI * 2);
  context.stroke();
  context.font = `13px ${FONT_STACK}`;
  context.textBaseline = "middle";
  const text = label || "Guest";
  const textWidth = context.measureText(text).width;
  const labelX = Math.min(px + 18, width - textWidth - 18);
  const labelY = Math.max(18, py - 22);
  context.fillStyle = "rgba(5, 8, 13, 0.82)";
  roundRect(context, labelX - 8, labelY - 12, textWidth + 16, 24, 6);
  context.fill();
  context.fillStyle = `rgba(${rgb}, 0.98)`;
  context.fillText(text, labelX, labelY);
  context.restore();
}

function drawWaveLeaderboard(context, width, rows) {
  if (!rows.length) {
    return;
  }

  const panelWidth = Math.min(240, Math.max(180, width * 0.22));
  const rowHeight = 24;
  const panelHeight = 34 + rows.length * rowHeight;
  const x = Math.max(12, width - panelWidth - 16);
  const y = 88;

  context.save();
  context.fillStyle = "rgba(5, 8, 13, 0.58)";
  context.strokeStyle = "rgba(122, 170, 255, 0.22)";
  context.lineWidth = 1;
  roundRect(context, x, y, panelWidth, panelHeight, 8);
  context.fill();
  context.stroke();
  context.font = `700 11px ${FONT_STACK}`;
  context.fillStyle = "rgba(117, 216, 255, 0.95)";
  context.textBaseline = "middle";
  context.fillText("WAVE SCORE", x + 12, y + 17);

  rows.forEach((row, index) => {
    const rowY = y + 36 + index * rowHeight;
    const rgb = Array.isArray(row.color) ? row.color : [255, 255, 255];
    context.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.95)`;
    context.beginPath();
    context.arc(x + 14, rowY - 1, 4, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "rgba(238, 243, 255, 0.92)";
    context.font = `600 13px ${FONT_STACK}`;
    context.fillText(row.label || row.name || "Guest", x + 26, rowY);
    context.textAlign = "right";
    context.fillText(String(row.score), x + panelWidth - 12, rowY);
    context.textAlign = "left";
  });
  context.restore();
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

function drawCalibrationTarget(context, width, height, target, now) {
  const px = target.x * width;
  const py = target.y * height;
  const pulse = 0.5 + 0.5 * Math.sin(now / 180);

  context.save();
  context.shadowBlur = 28;
  context.shadowColor = "rgba(117, 216, 255, 0.75)";
  context.strokeStyle = "rgba(117, 216, 255, 0.92)";
  context.lineWidth = 2;
  context.beginPath();
  context.arc(px, py, 26 + pulse * 6, 0, Math.PI * 2);
  context.stroke();
  context.shadowBlur = 0;
  context.fillStyle = "rgba(156, 255, 210, 0.98)";
  context.beginPath();
  context.arc(px, py, 6, 0, Math.PI * 2);
  context.fill();
  context.strokeStyle = "rgba(156, 255, 210, 0.68)";
  context.beginPath();
  context.moveTo(px - 18, py);
  context.lineTo(px + 18, py);
  context.moveTo(px, py - 18);
  context.lineTo(px, py + 18);
  context.stroke();
  context.restore();
}

function drawTrainerTarget(
  context,
  width,
  height,
  { session, target, now, trainCaptureMs, challengeDwellMs, challengeTargetRadiusPx },
) {
  if (!target) {
    return;
  }

  const px = target.x * width;
  const py = target.y * height;
  const phaseElapsed = now - (session.phaseStartedAt || now);
  const warmup = Math.min(1, phaseElapsed / trainCaptureMs);
  const isCapture = session.phase === "capture" || isWaveMode(session.mode);

  if (session.mode === "dojo") {
    drawDojoDummy(context, px, py, warmup, isCapture, now);
  } else if (isWaveMode(session.mode)) {
    const dwellProgress = session.dwellStartedAt
      ? clamp01((now - session.dwellStartedAt) / challengeDwellMs)
      : 0;
    drawEnemy(
      context,
      px,
      py,
      session.score,
      dwellProgress,
      session.mode,
      challengeTargetRadiusPx,
      now,
    );
  }
}

function drawDojoDummy(context, px, py, warmup, isCapture, now) {
  const radius = 24 + warmup * 8;
  const pulse = 0.5 + 0.5 * Math.sin(now / 130);
  const stroke = isCapture ? "rgba(156, 255, 210, 0.95)" : "rgba(117, 216, 255, 0.94)";

  context.save();
  context.shadowBlur = 28;
  context.shadowColor = stroke;
  context.strokeStyle = stroke;
  context.lineWidth = 2.5;
  context.beginPath();
  context.arc(px, py, radius + pulse * 4, 0, Math.PI * 2);
  context.stroke();
  context.shadowBlur = 0;

  context.strokeStyle = "rgba(255, 213, 112, 0.58)";
  context.lineWidth = 6;
  context.lineCap = "round";
  context.beginPath();
  context.moveTo(px - 28, py + 10);
  context.lineTo(px + 28, py + 10);
  context.moveTo(px, py + 2);
  context.lineTo(px, py + 46);
  context.stroke();

  context.fillStyle = "rgba(174, 111, 58, 0.96)";
  context.strokeStyle = "rgba(255, 213, 112, 0.86)";
  context.lineWidth = 2;
  context.beginPath();
  context.arc(px, py - 12, 18, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.fillStyle = "rgba(106, 66, 36, 0.92)";
  roundRect(context, px - 16, py + 6, 32, 34, 8);
  context.fill();

  drawTargetGlyph(context, px, py, radius, "rgba(156, 255, 210, 0.98)");
  context.restore();
}

function drawEnemy(context, px, py, score, dwellProgress, mode, radius, now) {
  const pulse = 0.5 + 0.5 * Math.sin(now / 105);
  const accent =
    mode === "multiplayer" ? "rgba(255, 111, 145, 0.95)" : "rgba(255, 213, 112, 0.95)";

  context.save();
  context.shadowBlur = 32;
  context.shadowColor = accent;
  context.strokeStyle = accent;
  context.lineWidth = 2.5;
  context.beginPath();
  context.arc(px, py, radius + pulse * 5, 0, Math.PI * 2);
  context.stroke();
  context.shadowBlur = 0;

  context.fillStyle = "rgba(15, 20, 32, 0.98)";
  context.strokeStyle = "rgba(238, 243, 255, 0.3)";
  context.lineWidth = 2;
  context.beginPath();
  context.arc(px, py - 2, 25, 0, Math.PI * 2);
  context.fill();
  context.stroke();

  context.fillStyle = "rgba(238, 243, 255, 0.88)";
  roundRect(context, px - 16, py - 9, 32, 10, 5);
  context.fill();
  context.fillStyle = "rgba(5, 8, 13, 0.92)";
  context.beginPath();
  context.arc(px - 7, py - 4, 2.2, 0, Math.PI * 2);
  context.arc(px + 7, py - 4, 2.2, 0, Math.PI * 2);
  context.fill();

  context.strokeStyle = "rgba(238, 243, 255, 0.7)";
  context.lineWidth = 3;
  context.lineCap = "round";
  context.beginPath();
  context.moveTo(px + 15, py + 17);
  context.lineTo(px + 36, py - 18);
  context.stroke();
  context.strokeStyle = "rgba(255, 213, 112, 0.82)";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(px + 30, py - 26);
  context.lineTo(px + 39, py - 12);
  context.stroke();

  if (dwellProgress > 0) {
    context.strokeStyle = "rgba(156, 255, 210, 0.98)";
    context.lineWidth = 4;
    context.beginPath();
    context.arc(px, py, radius + 10, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * dwellProgress);
    context.stroke();
  }

  drawTargetGlyph(context, px, py, radius, "rgba(255, 213, 112, 0.98)");
  context.font = `700 14px ${FONT_STACK}`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillStyle = "rgba(5, 8, 13, 0.86)";
  context.fillText(String(score), px, py + 18);
  context.restore();
}

function drawTargetGlyph(context, px, py, radius, color) {
  context.fillStyle = color;
  context.beginPath();
  context.arc(px, py, Math.max(5, radius * 0.16), 0, Math.PI * 2);
  context.fill();
  context.strokeStyle = "rgba(238, 243, 255, 0.58)";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(px - radius * 0.64, py);
  context.lineTo(px + radius * 0.64, py);
  context.moveTo(px, py - radius * 0.64);
  context.lineTo(px, py + radius * 0.64);
  context.stroke();
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}
