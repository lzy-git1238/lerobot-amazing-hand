"""Combine an SO-101 leader and a hand-tracking camera into one teleoperator."""

from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.utils.amazinghand import HAND_SERVO_NAMES, HandCalibration
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..so_leader import SO101Leader, SO101LeaderConfig
from ..teleoperator import Teleoperator
from .configuration import SOAmazingHandCameraConfig
from .juxi import JuxiHandBridge
from .tracker import CameraHandTracker, HandTrackingLostError


class SOAmazingHandCamera(Teleoperator):
    config_class = SOAmazingHandCameraConfig
    name = "so101_amazing_hand_camera"

    def __init__(self, config: SOAmazingHandCameraConfig):
        self.id = config.id
        self.config = config
        camera = config.hand_camera
        if camera.juxi_source is None:
            self.tracker = CameraHandTracker(camera)
            self.hand_calibration = HandCalibration.load(camera.calibration_path)
        else:
            self.tracker = JuxiHandBridge(camera.juxi_source, camera.dashboard_port, camera.hold_timeout_s)
        self.leader = SO101Leader(
            SO101LeaderConfig(
                port=config.port,
                id=config.id,
                calibration_dir=config.calibration_dir,
                use_degrees=config.use_degrees,
                num_read_retries=config.num_read_retries,
            )
        )
        self._last_flexion = None
        self._last_detection = None

    @property
    def action_features(self) -> dict[str, type]:
        return {k: v for k, v in self.leader.action_features.items() if k != "gripper.pos"} | dict.fromkeys(
            HAND_SERVO_NAMES, float
        )

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.leader.is_connected and self.tracker.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.leader.is_calibrated

    def calibrate(self) -> None:
        self.leader.calibrate()

    def configure(self) -> None:
        self.leader.configure()

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.tracker.connect()
        try:
            self.leader.connect(calibrate=calibrate)
            if isinstance(self.tracker, JuxiHandBridge):
                self.tracker.wait_for_targets()
        except BaseException:
            self.disconnect()
            raise

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        angles = self._read_hand_targets()
        arm = self.leader.get_action()
        del arm["gripper.pos"]
        return arm | dict(zip(HAND_SERVO_NAMES, angles, strict=True))

    def _read_hand_targets(self) -> list[float]:
        if isinstance(self.tracker, JuxiHandBridge):
            return self.tracker.read()
        flexion, timestamp = self.tracker.read()
        if flexion is not None:
            alpha = self.config.hand_camera.smoothing
            if self._last_flexion is not None:
                flexion = self._last_flexion + alpha * (flexion - self._last_flexion)
            self._last_flexion = flexion
            self._last_detection = timestamp
        if (
            self._last_flexion is None
            or self._last_detection is None
            or timestamp - self._last_detection > self.config.hand_camera.hold_timeout_s
        ):
            raise HandTrackingLostError(
                "AmazingHand tracking lost. Put the selected hand in the camera view."
            )

        return self.hand_calibration.angles(self._last_flexion).tolist()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        try:
            if self.leader.is_connected:
                self.leader.disconnect()
        finally:
            self.tracker.disconnect()
            self._last_flexion = None
            self._last_detection = None
