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

from scripts.browser_model_manifest import selected_browser_model_specs
from scripts.export_browser_onnx import BrowserGazeModel
from shared_gaze.config import EYE_CROP_HEIGHT, EYE_CROP_WIDTH
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
    feeds = {input_meta.name: inputs[input_meta.name] for input_meta in session.get_inputs()}
    onnx_output = session.run(["gaze"], feeds)[0]
    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))

    print(f"PyTorch output: {torch_output.tolist()}")
    print(f"ONNX output:    {onnx_output.tolist()}")
    print(f"Max abs diff:   {max_abs_diff:.8f}")
    if max_abs_diff > 1e-5:
        raise SystemExit("ONNX output differs from PyTorch output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare browser ONNX output to PyTorch output.")
    parser.add_argument(
        "--model-key",
        choices=[spec.key for spec in selected_browser_model_specs(None, True)],
        help="Verify one configured browser model. Defaults to all models.",
    )
    parser.add_argument("--all", action="store_true", help="Verify all configured browser models.")
    parser.add_argument("--checkpoint", type=Path, help="Override checkpoint path for single-model verification.")
    parser.add_argument("--model", type=Path, help="Override ONNX path for single-model verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = selected_browser_model_specs(args.model_key, args.all or args.model_key is None)
    if (args.checkpoint or args.model) and len(specs) != 1:
        raise SystemExit("--checkpoint and --model can only be used with --model-key")
    for spec in specs:
        checkpoint_path = args.checkpoint.expanduser().resolve() if args.checkpoint else spec.resolve_checkpoint()
        model_path = args.model.expanduser().resolve() if args.model else spec.output
        print(f"Verifying {spec.key}: {checkpoint_path} -> {model_path}")
        verify_browser_onnx(checkpoint_path, model_path)


if __name__ == "__main__":
    main()
