from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from shared_gaze.config import DEFAULT_CHECKPOINT_PATH
from shared_gaze.vision_model import EyeCropModelConfig, EyeCropRegressor


def choose_device(requested_device: str | None = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_checkpoint(checkpoint_path: Path) -> dict:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


@dataclass
class VisionGazePredictor:
    checkpoint_path: Path
    head_feature_keys: list[str]
    head_mean: np.ndarray
    head_scale: np.ndarray
    extra_feature_keys: list[str]
    extra_mean: np.ndarray
    extra_scale: np.ndarray
    model: EyeCropRegressor
    device: torch.device


def load_vision_predictor(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    requested_device: str | None = None,
) -> VisionGazePredictor:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = _load_checkpoint(checkpoint_path)
    head_feature_keys = list(checkpoint.get("head_feature_keys") or [])
    if not head_feature_keys:
        raise ValueError("Vision checkpoint is missing auxiliary feature keys")

    model_config = EyeCropModelConfig.from_dict(checkpoint.get("model_config"))
    extra_feature_keys = list(checkpoint.get("extra_feature_keys") or model_config.extra_feature_keys)
    model = EyeCropRegressor(
        head_feature_dim=len(head_feature_keys),
        extra_feature_dim=len(extra_feature_keys),
        config=model_config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = choose_device(requested_device)
    model.to(device)
    model.eval()

    head_mean = np.array(checkpoint["head_mean"], dtype=np.float32)
    head_scale = np.array(checkpoint["head_scale"], dtype=np.float32)
    head_scale[head_scale < 1e-6] = 1.0
    extra_mean = np.array(
        checkpoint.get("extra_mean") or [0.0] * len(extra_feature_keys),
        dtype=np.float32,
    )
    extra_scale = np.array(
        checkpoint.get("extra_scale") or [1.0] * len(extra_feature_keys),
        dtype=np.float32,
    )
    extra_scale[extra_scale < 1e-6] = 1.0

    return VisionGazePredictor(
        checkpoint_path=checkpoint_path,
        head_feature_keys=head_feature_keys,
        head_mean=head_mean,
        head_scale=head_scale,
        extra_feature_keys=extra_feature_keys,
        extra_mean=extra_mean,
        extra_scale=extra_scale,
        model=model,
        device=device,
    )
