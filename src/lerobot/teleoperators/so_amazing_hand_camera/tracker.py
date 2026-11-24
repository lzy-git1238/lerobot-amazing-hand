"""Read a front-facing RGB camera and estimate four finger flexion values."""

import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from lerobot.cameras.opencv import OpenCVCamera
from lerobot.utils.amazinghand import finger_flexion
from lerobot.utils.import_utils import _mediapipe_available, require_package

from .configuration import HandCameraConfig
from .dashboard import HandDashboard, load_vision

if TYPE_CHECKING or _mediapipe_available:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision


class HandTrackingLostError(RuntimeError):
    """The selected hand is unavailable for teleoperation."""


class CameraHandTracker:
    def __init__(self, config: HandCameraConfig):
        self.config = config
        self.dashboard: HandDashboard | None = None
        self.vision_bounds = load_vision(config)
        self.camera = OpenCVCamera(config.camera)
        self._landmarker: vision.HandLandmarker | None = None
        self._last_timestamp_ms = -1

    @property
    def is_connected(self) -> bool:
        return self._landmarker is not None and self.camera.is_connected

    def connect(self) -> None:
        require_package("mediapipe", extra="amazinghand-camera")
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.config.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        try:
            self.camera.connect()
            if self.dashboard:
                self.dashboard.start()
        except BaseException:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        try:
            if self.dashboard:
                self.dashboard.close()
            if self.camera.is_connected:
                self.camera.disconnect()
        finally:
            if self._landmarker is not None:
                self._landmarker.close()
                self._landmarker = None
            self._last_timestamp_ms = -1

    def read(self) -> tuple[NDArray[np.float64] | None, float]:
        if self._landmarker is None:
            raise RuntimeError("AmazingHand camera tracker is disconnected.")
        try:
            frame = self.camera.async_read(timeout_ms=min(100.0, self.config.hold_timeout_s * 1000))
        except TimeoutError:
            if self.dashboard:
                self.dashboard.frame(None, [], None, None)
            return None, time.monotonic()

        captured_at = time.monotonic()
        timestamp_ms = max(int(captured_at * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        for index, (handedness, landmarks) in enumerate(
            zip(result.handedness, result.hand_world_landmarks, strict=True)
        ):
            if not handedness or handedness[0].category_name != self.config.handedness:
                continue
            try:
                raw = finger_flexion([(p.x, p.y, p.z) for p in landmarks])
            except ValueError:
                continue
            bounds = self.dashboard.bounds if self.dashboard else self.vision_bounds
            flexion = raw if bounds is None else np.clip((raw - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)
            if self.dashboard:
                points = [(p.x, p.y) for p in result.hand_landmarks[index]]
                self.dashboard.frame(frame, points, raw, flexion)
            return flexion, captured_at
        if self.dashboard:
            self.dashboard.frame(frame, [], None, None)
        return None, captured_at
