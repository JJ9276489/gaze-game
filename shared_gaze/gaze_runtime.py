from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import mediapipe as mp

from shared_gaze.camera import get_screen_size, open_camera
from shared_gaze.config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FACE_LANDMARKER_PATH,
    PREDICTION_ALPHA,
)
from shared_gaze.eye_crops import extract_eye_crops
from shared_gaze.features import extract_feature_frame
from shared_gaze.landmarker import create_landmarker, ensure_face_landmarker
from shared_gaze.vision_runtime import VisionGazePredictor, load_vision_predictor


@dataclass
class GazeReading:
    x: float | None
    y: float | None
    tracking: bool
    timestamp_ms: int
    raw_x: float | None = None
    raw_y: float | None = None
    model_label: str | None = None


def smooth_prediction(
    previous: tuple[float, float] | None,
    current: tuple[float, float] | None,
    alpha: float,
) -> tuple[float, float] | None:
    if current is None:
        return None
    if previous is None:
        return current
    return (
        previous[0] * (1.0 - alpha) + current[0] * alpha,
        previous[1] * (1.0 - alpha) + current[1] * alpha,
    )


class GazeRuntime:
    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        face_landmarker_path: Path = DEFAULT_FACE_LANDMARKER_PATH,
        requested_device: str | None = None,
        camera_index: int = 0,
        smoothing_alpha: float = PREDICTION_ALPHA,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.face_landmarker_path = face_landmarker_path
        self.requested_device = requested_device
        self.camera_index = camera_index
        self.smoothing_alpha = smoothing_alpha

        self.screen_size = get_screen_size()
        self.predictor: VisionGazePredictor | None = None
        self.cap = None
        self.landmarker = None
        self.smoothed_prediction: tuple[float, float] | None = None

    def start(self) -> "GazeRuntime":
        face_model = ensure_face_landmarker(self.face_landmarker_path)
        self.predictor = load_vision_predictor(
            checkpoint_path=self.checkpoint_path,
            requested_device=self.requested_device,
        )
        if self.predictor.checkpoint_screen_size != self.screen_size:
            print(
                "Warning: checkpoint was trained for "
                f"{self.predictor.checkpoint_screen_size[0]}x{self.predictor.checkpoint_screen_size[1]}, "
                f"current screen is {self.screen_size[0]}x{self.screen_size[1]}. "
                "The game will still render normalized coordinates."
            )
        self.cap = open_camera(self.camera_index)
        self.landmarker = create_landmarker(face_model)
        return self

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "GazeRuntime":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def model_label(self) -> str | None:
        if self.predictor is None:
            return None
        return self.predictor.label

    def read(self) -> GazeReading:
        timestamp_ms = int(time.monotonic() * 1000)
        if self.cap is None or self.landmarker is None or self.predictor is None:
            raise RuntimeError("GazeRuntime.start() must be called before read()")

        ok, frame = self.cap.read()
        if not ok:
            self.smoothed_prediction = None
            return GazeReading(None, None, False, timestamp_ms, model_label=self.model_label)

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        feature_frame = extract_feature_frame(result)
        if feature_frame is None:
            self.smoothed_prediction = None
            return GazeReading(None, None, False, timestamp_ms, model_label=self.model_label)

        eye_crops = extract_eye_crops(frame, feature_frame.face_landmarks)
        raw_prediction = self.predictor.predict_normalized(eye_crops, feature_frame.payload)
        self.smoothed_prediction = smooth_prediction(
            self.smoothed_prediction,
            raw_prediction,
            self.smoothing_alpha,
        )
        if self.smoothed_prediction is None:
            return GazeReading(None, None, False, timestamp_ms, model_label=self.model_label)
        return GazeReading(
            self.smoothed_prediction[0],
            self.smoothed_prediction[1],
            True,
            timestamp_ms,
            raw_x=raw_prediction[0],
            raw_y=raw_prediction[1],
            model_label=self.model_label,
        )
