from pathlib import Path
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
APP_RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))


def _relay_urls_from_env() -> list[str]:
    raw_urls = os.environ.get("GAZE_GAME_RELAY_URLS")
    if raw_urls:
        return [item.strip() for item in raw_urls.replace(";", ",").split(",") if item.strip()]
    raw_url = os.environ.get("GAZE_GAME_RELAY_URL")
    if raw_url:
        return [raw_url.strip()]
    return []


DEFAULT_RELAY_URLS = _relay_urls_from_env() or [
    "ws://127.0.0.1:8765",
]
DEFAULT_RELAY_URL = DEFAULT_RELAY_URLS[0]

_BUNDLED_CHECKPOINT_PATH = APP_RESOURCE_ROOT / "models" / "vision_gaze_spatial_geom.pt"
_SIBLING_CHECKPOINT_PATH = (
    PROJECT_ROOT.parent / "eye-cursor" / "models" / "vision_gaze_spatial_geom.pt"
)
_SIBLING_FACE_LANDMARKER_PATH = (
    PROJECT_ROOT.parent / "eye-cursor" / "models" / "face_landmarker.task"
)
DEFAULT_CHECKPOINT_PATH = (
    _BUNDLED_CHECKPOINT_PATH
    if _BUNDLED_CHECKPOINT_PATH.exists()
    else _SIBLING_CHECKPOINT_PATH
)
DEFAULT_FACE_LANDMARKER_PATH = (
    _SIBLING_FACE_LANDMARKER_PATH
    if _SIBLING_FACE_LANDMARKER_PATH.exists()
    else MODELS_DIR / "face_landmarker.task"
)

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

EYE_CROP_WIDTH = 96
EYE_CROP_HEIGHT = 64

PREDICTION_ALPHA = 0.35

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

VISION_EYE_GEOMETRY_FEATURE_KEYS = [
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
]

RIGHT_IRIS_POINTS = [469, 470, 471, 472]
LEFT_IRIS_POINTS = [474, 475, 476, 477]

RIGHT_EYE_CORNER_POINTS = (33, 133)
RIGHT_EYE_UPPER_LID_POINTS = [159, 158, 160, 161]
RIGHT_EYE_LOWER_LID_POINTS = [145, 153, 144, 163]

LEFT_EYE_CORNER_POINTS = (263, 362)
LEFT_EYE_UPPER_LID_POINTS = [386, 385, 387, 388]
LEFT_EYE_LOWER_LID_POINTS = [374, 380, 373, 390]
