#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_browser_onnx import BrowserGazeModel, DEFAULT_OUTPUT_PATH
from shared_gaze.config import DEFAULT_CHECKPOINT_PATH, EYE_CROP_HEIGHT, EYE_CROP_WIDTH
from shared_gaze.vision_runtime import load_vision_predictor


def verify_browser_onnx(checkpoint_path: Path, model_path: Path) -> None:
    predictor = load_vision_predictor(checkpoint_path=checkpoint_path, requested_device="cpu")
    predictor.model.eval()
    wrapped = BrowserGazeModel(
        predictor.model,
        predictor.head_mean,
        predictor.head_scale,
        predictor.extra_mean,
        predictor.extra_scale,
    ).eval()

    rng = np.random.default_rng(9276489)
    inputs = {
        "left_eye": rng.uniform(-1.0, 1.0, (1, 1, EYE_CROP_HEIGHT, EYE_CROP_WIDTH)).astype(
            np.float32
        ),
        "right_eye": rng.uniform(-1.0, 1.0, (1, 1, EYE_CROP_HEIGHT, EYE_CROP_WIDTH)).astype(
            np.float32
        ),
        "head_features": rng.normal(0.0, 1.0, (1, len(predictor.head_feature_keys))).astype(
            np.float32
        ),
        "extra_features": rng.normal(0.0, 1.0, (1, len(predictor.extra_feature_keys))).astype(
            np.float32
        ),
    }

    with torch.no_grad():
        torch_output = (
            wrapped(*(torch.from_numpy(inputs[name]) for name in inputs))
            .detach()
            .cpu()
            .numpy()
        )

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["gaze"], inputs)[0]
    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))

    print(f"PyTorch output: {torch_output.tolist()}")
    print(f"ONNX output:    {onnx_output.tolist()}")
    print(f"Max abs diff:   {max_abs_diff:.8f}")
    if max_abs_diff > 1e-5:
        raise SystemExit("ONNX output differs from PyTorch output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare browser ONNX output to PyTorch output.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_browser_onnx(args.checkpoint, args.model)


if __name__ == "__main__":
    main()
