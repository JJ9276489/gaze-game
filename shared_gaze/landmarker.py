from pathlib import Path
import urllib.error
import urllib.request

import mediapipe as mp

from shared_gaze.config import DEFAULT_FACE_LANDMARKER_PATH, FACE_LANDMARKER_URL


def ensure_face_landmarker(model_path: Path = DEFAULT_FACE_LANDMARKER_PATH) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return model_path

    print(f"Downloading MediaPipe face landmarker to {model_path}...")
    try:
        with urllib.request.urlopen(FACE_LANDMARKER_URL) as response:
            model_path.write_bytes(response.read())
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Could not download the MediaPipe face landmarker model. "
            f"Download it manually from {FACE_LANDMARKER_URL} and save it to {model_path}."
        ) from error

    return model_path


def create_landmarker(model_path: Path = DEFAULT_FACE_LANDMARKER_PATH):
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)

