from pathlib import Path
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
APP_RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))


def _parse_relay_urls(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def _relay_urls_from_text(text: str) -> list[str]:
    urls: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            urls.extend(_parse_relay_urls(line))
    return urls


def _app_bundle_path() -> Path | None:
    if not getattr(sys, "frozen", False):
        return None
    executable = Path(sys.executable).resolve()
    for parent in executable.parents:
        if parent.suffix == ".app":
            return parent
    return None


def _relay_url_config_paths() -> list[Path]:
    paths: list[Path] = []
    app_bundle = _app_bundle_path()
    if app_bundle is not None:
        paths.extend(
            [
                app_bundle.parent / "relay_urls.txt",
                app_bundle.parent / "Gaze Game Relay URLs.txt",
                app_bundle / "Contents" / "Resources" / "relay_urls.txt",
            ]
        )

    paths.extend(
        [
            PROJECT_ROOT / "relay_urls.local.txt",
            PROJECT_ROOT / "relay_urls.txt",
            Path.home() / "Library" / "Application Support" / "Gaze Game" / "relay_urls.txt",
        ]
    )

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return unique_paths


def _relay_urls_from_config_files() -> list[str]:
    for path in _relay_url_config_paths():
        try:
            urls = _relay_urls_from_text(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except OSError:
            continue
        if urls:
            return urls
    return []


def _relay_urls_from_env() -> list[str]:
    raw_urls = os.environ.get("GAZE_GAME_RELAY_URLS")
    if raw_urls:
        return _parse_relay_urls(raw_urls)
    raw_url = os.environ.get("GAZE_GAME_RELAY_URL")
    if raw_url:
        return _parse_relay_urls(raw_url)
    return []


DEFAULT_RELAY_URLS = _relay_urls_from_env() or _relay_urls_from_config_files() or [
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
