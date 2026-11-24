from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_amazing_hand import SO101AmazingHandFollowerConfig
from lerobot.utils.amazinghand import DEFAULT_HAND_CALIBRATION, DEFAULT_JUXI_SOURCE

from ..config import TeleoperatorConfig
from ..so_leader.config_so_leader import SOLeaderConfig


@dataclass(kw_only=True)
class HandCameraConfig:
    model_path: Path | None = None
    juxi_source: Path | None = None
    camera: OpenCVCameraConfig = field(default_factory=lambda: OpenCVCameraConfig(index_or_path=0))
    calibration_path: Path = DEFAULT_HAND_CALIBRATION
    handedness: Literal["Left", "Right"] = "Right"
    dashboard: bool = True
    dashboard_port: int = 8765
    smoothing: float = 0.35
    hold_timeout_s: float = 0.5

    def __post_init__(self) -> None:
        if self.model_path is None and self.juxi_source is None:
            self.juxi_source = DEFAULT_JUXI_SOURCE
        if self.camera.color_mode != "rgb":
            raise ValueError("AmazingHand tracking requires an RGB camera.")
        if self.handedness not in ("Left", "Right"):
            raise ValueError("handedness must be Left or Right.")
        if not 0 < self.smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1].")
        if not isfinite(self.hold_timeout_s) or self.hold_timeout_s <= 0:
            raise ValueError("hold_timeout_s must be finite and positive.")


@TeleoperatorConfig.register_subclass("so101_amazing_hand_camera")
@dataclass(kw_only=True)
class SOAmazingHandCameraConfig(TeleoperatorConfig, SOLeaderConfig):
    hand_camera: HandCameraConfig

    def __post_init__(self) -> None:
        if not self.use_degrees:
            raise ValueError("AmazingHand camera teleoperation requires arm angles in degrees.")

    def check_robot_config(self, robot_config: object) -> None:
        expected_mode = "servo_juxi" if self.hand_camera.juxi_source is not None else "servo"
        if (
            not isinstance(robot_config, SO101AmazingHandFollowerConfig)
            or robot_config.hand_control_mode != expected_mode
        ):
            raise ValueError(
                f"Camera hand teleoperation requires so101_amazing_hand with hand_control_mode={expected_mode}."
            )
        if expected_mode == "servo_juxi":
            return
        robot_path = DEFAULT_HAND_CALIBRATION
        if robot_config.calibration_dir is not None:
            robot_path = robot_config.calibration_dir / "hand_angles.json"
        if self.hand_camera.calibration_path.resolve() != robot_path.resolve():
            raise ValueError(
                f"Set teleop.hand_camera.calibration_path to the robot's calibration file: {robot_path}"
            )
