export const CALIBRATION_VERSION = 3;
export const CALIBRATION_STORAGE_KEY = "gazeGame.calibration";
export const CALIBRATION_TARGETS = [
  { id: "center", x: 0.5, y: 0.5 },
  { id: "topLeft", x: 0.14, y: 0.16 },
  { id: "topRight", x: 0.86, y: 0.16 },
  { id: "bottomRight", x: 0.86, y: 0.84 },
  { id: "bottomLeft", x: 0.14, y: 0.84 },
];

const CALIBRATION_INSIDE_TOLERANCE = -0.04;

export function defaultCalibration() {
  return {
    version: CALIBRATION_VERSION,
    centerX: 0.5,
    centerY: 0.5,
    gainX: 1.85,
    gainY: 1.75,
    mappings: {},
  };
}

export function loadCalibration() {
  try {
    const saved = JSON.parse(localStorage.getItem(CALIBRATION_STORAGE_KEY) || "{}");
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

export function saveCalibration(calibration) {
  localStorage.setItem(CALIBRATION_STORAGE_KEY, JSON.stringify(calibration));
}

export function hasAnyCalibrationMapping(calibration) {
  return Object.values(calibration?.mappings || {}).some(Boolean);
}

export function getCalibrationMapping(calibration, kind) {
  return calibration?.mappings?.[kind] || null;
}

export function buildCalibrationMapping(kind, points) {
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

export function normalizeCalibrationMapping(mapping, fallbackKind) {
  if (!mapping || !Array.isArray(mapping.points)) {
    return null;
  }
  if (mapping.version !== CALIBRATION_VERSION) {
    return null;
  }
  return buildCalibrationMapping(mapping.kind || fallbackKind || "onnx", mapping.points);
}

export function mapPointWithCalibration(rawPoint, mapping) {
  const piecewise = mapPointWithPiecewiseTriangles(rawPoint, mapping.points);
  if (piecewise) {
    return piecewise;
  }
  return applyAffineTransform(mapping.affine, rawPoint);
}

export function calibrationTargetLabel(id) {
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
