#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.browser_model_manifest import (
    browser_model_spec,
    selected_browser_model_specs,
)
from shared_gaze.config import EYE_CROP_HEIGHT, EYE_CROP_WIDTH
from shared_gaze.vision_runtime import load_vision_predictor


DEFAULT_OUTPUT_PATH = browser_model_spec("spatial_geom").output


class BrowserGazeModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        head_mean: np.ndarray,
        head_scale: np.ndarray,
        extra_mean: np.ndarray,
        extra_scale: np.ndarray,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("head_mean", torch.from_numpy(head_mean.astype(np.float32)).unsqueeze(0))
        self.register_buffer(
            "head_scale", torch.from_numpy(head_scale.astype(np.float32)).unsqueeze(0)
        )
        self.register_buffer(
            "extra_mean", torch.from_numpy(extra_mean.astype(np.float32)).unsqueeze(0)
        )
        self.register_buffer(
            "extra_scale", torch.from_numpy(extra_scale.astype(np.float32)).unsqueeze(0)
        )

    def forward(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        head_features: torch.Tensor,
        extra_features: torch.Tensor,
    ) -> torch.Tensor:
        normalized_head = (head_features - self.head_mean) / self.head_scale
        normalized_extra = (extra_features - self.extra_mean) / self.extra_scale
        return self.model(left_eye, right_eye, normalized_head, normalized_extra)


def export_browser_onnx(checkpoint_path: Path, output_path: Path, opset: int) -> None:
    predictor = load_vision_predictor(checkpoint_path=checkpoint_path, requested_device="cpu")
    predictor.model.eval()

    wrapped = BrowserGazeModel(
        predictor.model,
        predictor.head_mean,
        predictor.head_scale,
        predictor.extra_mean,
        predictor.extra_scale,
    ).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    left_eye = torch.zeros(1, 1, EYE_CROP_HEIGHT, EYE_CROP_WIDTH, dtype=torch.float32)
    right_eye = torch.zeros(1, 1, EYE_CROP_HEIGHT, EYE_CROP_WIDTH, dtype=torch.float32)
    head_features = torch.zeros(1, len(predictor.head_feature_keys), dtype=torch.float32)
    extra_features = torch.zeros(1, len(predictor.extra_feature_keys), dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (left_eye, right_eye, head_features, extra_features),
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["left_eye", "right_eye", "head_features", "extra_features"],
            output_names=["gaze"],
            dynamic_axes={
                "left_eye": {0: "batch"},
                "right_eye": {0: "batch"},
                "head_features": {0: "batch"},
                "extra_features": {0: "batch"},
                "gaze": {0: "batch"},
            },
            dynamo=False,
        )

    print(f"Exported {output_path}")
    print(f"Head features: {', '.join(predictor.head_feature_keys)}")
    print(f"Extra features: {', '.join(predictor.extra_feature_keys)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the gaze checkpoint for browser ONNX.")
    parser.add_argument(
        "--model",
        choices=[spec.key for spec in selected_browser_model_specs(None, True)],
        help="Export one configured browser model. Defaults to all models.",
    )
    parser.add_argument("--all", action="store_true", help="Export all configured browser models.")
    parser.add_argument("--checkpoint", type=Path, help="Override checkpoint path for single-model export.")
    parser.add_argument("--output", type=Path, help="Override output path for single-model export.")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = selected_browser_model_specs(args.model, args.all or args.model is None)
    if (args.checkpoint or args.output) and len(specs) != 1:
        raise SystemExit("--checkpoint and --output can only be used with --model")
    for spec in specs:
        checkpoint_path = args.checkpoint.expanduser().resolve() if args.checkpoint else spec.resolve_checkpoint()
        output_path = args.output.expanduser().resolve() if args.output else spec.output
        print(f"Exporting {spec.key}: {checkpoint_path} -> {output_path}")
        export_browser_onnx(checkpoint_path, output_path, args.opset)


if __name__ == "__main__":
    main()
