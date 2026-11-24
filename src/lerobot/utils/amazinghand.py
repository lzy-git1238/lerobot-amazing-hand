"""Calibration and landmark geometry shared by AmazingHand teleoperation and control."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import HF_LEROBOT_CALIBRATION, ROBOTS

HAND_SERVO_IDS = tuple(range(1, 9))
HAND_SERVO_NAMES = tuple(f"hand_servo_{servo_id}.pos" for servo_id in HAND_SERVO_IDS)
DEFAULT_HAND_CALIBRATION = HF_LEROBOT_CALIBRATION / ROBOTS / "so101_amazing_hand" / "hand_angles.json"
DEFAULT_JUXI_SOURCE = Path(__file__).resolve().parent.parent / "teleoperators/so_amazing_hand_camera/web_assets"

# The AmazingHand servo IDs are assigned in index, middle, ring, thumb order.
FINGER_LANDMARKS = np.array([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [1, 2, 3, 4]])

# Juxi Amazinghand-web-io SCS0009 right-hand configuration (IDs 1–8).
CALIBRATION_AXIS_LIMITS = ((-50, 90), (-45, 45))
JUXI_SHAFT_LIMIT_DEG = 120.0
CALIBRATION_PRESETS = {
    "open": (-35.0,) * 4,
    "middle": (-17.578125,) * 4,
    "close": (75.0, 75.0, 75.0, 18.0),
}


def calibration_servo_goals(flexion: float, spread: float) -> tuple[float, float]:
    """Convert Juxi right-hand finger axes to absolute shaft degrees."""
    odd_goal = max(-120.0, min(120.0, flexion - spread))
    even_goal = max(-120.0, min(120.0, -(flexion + spread)))
    return odd_goal, even_goal


def calibration_axes(odd_position: float, even_position: float) -> tuple[float, float]:
    """Convert absolute shaft degrees back to the right-hand finger axes."""
    return (odd_position - even_position) / 2, (-even_position - odd_position) / 2


def read_hand_positions(read_position: Callable[[int], float | list[float]]) -> NDArray[np.float64]:
    """Read all eight absolute actuator positions in radians in servo ID order."""
    # SCS0009 lacks Sync Read, and rustypot can return a one-element array for a single register.
    positions = np.asarray([read_position(servo_id) for servo_id in HAND_SERVO_IDS], dtype=np.float64)
    if positions.shape == (8, 1):
        positions = positions[:, 0]
    if positions.shape != (8,) or not np.all(np.isfinite(positions)):
        raise RuntimeError("Expected position feedback from all eight AmazingHand servos.")
    return positions


@dataclass(frozen=True)
class HandCalibration:
    open_angles: NDArray[np.float64]
    close_angles: NDArray[np.float64]

    @classmethod
    def load(cls, path: Path) -> "HandCalibration":
        try:
            data = json.loads(path.read_text())
            open_angles = np.asarray(data["hand_open_angles"], dtype=np.float64)
            close_angles = np.asarray(data["hand_close_angles"], dtype=np.float64)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Cannot load AmazingHand endpoints from {path}. Run lerobot-calibrate-amazing-hand."
            ) from exc
        if any(
            angles.shape != (8,) or not np.all(np.isfinite(angles)) for angles in (open_angles, close_angles)
        ):
            raise ValueError(f"Expected 8 finite open and close angles in {path}.")
        return cls(open_angles, close_angles)

    def angles(self, flexion: ArrayLike) -> NDArray[np.float64]:
        """Convert four flexion fractions to eight center-relative actuator degrees."""
        progress = np.asarray(flexion, dtype=np.float64)
        if progress.shape != (4,) or not np.all(np.isfinite(progress)):
            raise ValueError("Expected four finite finger flexion values.")
        return self.open_angles + np.repeat(np.clip(progress, 0.0, 1.0), 2) * (
            self.close_angles - self.open_angles
        )

    def project(self, angles: ArrayLike) -> NDArray[np.float64]:
        """Find the closest point on each finger's calibrated two-servo path."""
        target = np.asarray(angles, dtype=np.float64)
        if target.shape != (8,) or not np.all(np.isfinite(target)):
            raise ValueError("Expected eight finite AmazingHand servo targets.")
        travel = (self.close_angles - self.open_angles).reshape(4, 2)
        offset = (target - self.open_angles).reshape(4, 2)
        length_squared = np.sum(travel * travel, axis=1)
        progress = np.divide(
            np.sum(offset * travel, axis=1),
            length_squared,
            out=np.zeros(4),
            where=length_squared > 0,
        )
        return self.angles(progress)


def finger_flexion(landmarks: ArrayLike) -> NDArray[np.float64]:
    """Estimate index, middle, ring and thumb flexion from 21 MediaPipe world landmarks."""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        raise ValueError("Expected 21 finite hand landmarks in three dimensions.")
    bones = np.diff(points[FINGER_LANDMARKS], axis=1)
    lengths = np.linalg.norm(bones, axis=2)
    if np.any(lengths <= 1e-6):
        raise ValueError("Hand landmarks have overlapping joints.")
    unit_bones = bones / lengths[..., None]
    cosines = np.sum(unit_bones[:, :-1] * unit_bones[:, 1:], axis=2)
    bend = np.arccos(np.clip(cosines, -1.0, 1.0)).sum(axis=1)
    return np.clip(bend / (np.pi / 2), 0.0, 1.0)
