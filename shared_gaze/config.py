from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

_LOCAL_CHECKPOINT_PATH = MODELS_DIR / "vision_gaze_spatial_geom.pt"
_SIBLING_CHECKPOINT_PATH = (
    PROJECT_ROOT.parent / "eye-cursor" / "models" / "vision_gaze_spatial_geom.pt"
)
DEFAULT_CHECKPOINT_PATH = (
    _LOCAL_CHECKPOINT_PATH
    if _LOCAL_CHECKPOINT_PATH.exists()
    else _SIBLING_CHECKPOINT_PATH
)

EYE_CROP_WIDTH = 96
EYE_CROP_HEIGHT = 64

VISION_HEAD_FEATURE_KEYS = [
    "face_center_x",
    "face_center_y",
    "face_scale",
    "head_yaw_deg",
    "head_pitch_deg",
    "head_roll_deg",
    "head_tx",
    "head_ty",
    "head_tz",
]
