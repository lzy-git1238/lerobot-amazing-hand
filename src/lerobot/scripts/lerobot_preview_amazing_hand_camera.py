"""Preview, calibrate or drive AmazingHand from the local hand-tracking camera."""

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_amazing_hand.config_so_amazing_hand import SOAmazingHandFollowerConfig
from lerobot.teleoperators.so_amazing_hand_camera.configuration import HandCameraConfig
from lerobot.teleoperators.so_amazing_hand_camera.dashboard import HandDashboard
from lerobot.teleoperators.so_amazing_hand_camera.juxi import JuxiHandBridge
from lerobot.teleoperators.so_amazing_hand_camera.tracker import CameraHandTracker, HandTrackingLostError
from lerobot.utils.amazinghand import (
    DEFAULT_HAND_CALIBRATION,
    DEFAULT_JUXI_SOURCE,
    HAND_SERVO_IDS,
    HAND_SERVO_NAMES,
    HandCalibration,
    read_hand_positions,
)
from lerobot.utils.import_utils import _rustypot_available, require_package

if TYPE_CHECKING or _rustypot_available:
    from rustypot import Scs0009PyController


def run_juxi_follow(source: Path, port: str, dashboard_port: int) -> None:
    require_package("rustypot", extra="amazinghand")
    ids = list(HAND_SERVO_IDS)
    hand = Scs0009PyController(serial_port=port, baudrate=1_000_000, timeout=0.15)
    bridge = JuxiHandBridge(source, dashboard_port, timeout=0.5)
    target = None
    try:
        hand.sync_write_torque_enable(ids, [0] * 8)
        bridge.connect()
        bridge.wait_for_targets()
        while True:
            try:
                goals = bridge.read()
            except HandTrackingLostError:
                hand.sync_write_torque_enable(ids, [0] * 8)
                target = None
                bridge.wait_for_targets()
                continue
            if target is None:
                current = read_hand_positions(hand.read_present_position)
                target = np.rad2deg(current)
                hand.sync_write_goal_speed(ids, [3] * 8)
                hand.sync_write_goal_position(ids, current.tolist())
                hand.sync_write_torque_enable(ids, [1] * 8)
            change = (goals - target).reshape(4, 2)
            scale = np.maximum(1, np.max(np.abs(change), axis=1, keepdims=True) / 6)
            next_target = target + (change / scale).ravel()
            if np.max(np.abs(next_target - target)) >= 0.5:
                hand.sync_write_goal_position(ids, np.deg2rad(next_target).tolist())
                target = next_target
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            hand.sync_write_torque_enable(ids, [0] * 8)
        finally:
            bridge.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--juxi-source", type=Path, help="Override the bundled Juxi browser assets with a local checkout."
    )
    parser.add_argument("--calibration-path", type=Path)
    parser.add_argument("--camera", default="0", help="OpenCV camera index or capture device path.")
    parser.add_argument("--handedness", choices=["Left", "Right"], default="Right")
    parser.add_argument("--frames", type=int, help="Exit after this many frames.")
    parser.add_argument(
        "--dashboard", action="store_true", help="Open the local mapping and calibration page."
    )
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--hand-port", help="Connect the SCS0009 hand for calibration or tracking.")
    parser.add_argument(
        "--follow", action="store_true", help="Drive the eight servos from the selected hand."
    )
    args = parser.parse_args()
    if args.follow and args.model_path is None and args.juxi_source is None:
        args.juxi_source = DEFAULT_JUXI_SOURCE
    if args.juxi_source is not None:
        if not args.follow or not args.hand_port:
            parser.error("--juxi-source requires --follow and --hand-port")
        run_juxi_follow(args.juxi_source, args.hand_port, args.dashboard_port)
        return
    if args.model_path is None:
        parser.error("Provide --model-path or --juxi-source")
    if args.follow:
        if not args.hand_port:
            parser.error("--follow requires --hand-port")
        args.dashboard = True
    if args.hand_port and not args.dashboard:
        parser.error("--hand-port requires --dashboard")
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    camera_source = int(args.camera) if args.camera.isdecimal() else Path(args.camera)
    calibration_path = args.calibration_path
    if calibration_path is None and DEFAULT_HAND_CALIBRATION.is_file():
        calibration_path = DEFAULT_HAND_CALIBRATION
    calibration = HandCalibration.load(calibration_path) if calibration_path is not None else None
    tracker = CameraHandTracker(
        HandCameraConfig(
            model_path=args.model_path,
            camera=OpenCVCameraConfig(index_or_path=camera_source),
            handedness=args.handedness,
            dashboard_port=args.dashboard_port,
            calibration_path=calibration_path or DEFAULT_HAND_CALIBRATION,
        )
    )
    hand = None
    if args.hand_port:
        require_package("rustypot", extra="amazinghand")
        hand = Scs0009PyController(serial_port=args.hand_port, baudrate=1_000_000, timeout=0.15)
        center = SOAmazingHandFollowerConfig(hand_port=args.hand_port).hand_middle_pos
        start = np.rad2deg(read_hand_positions(hand.read_present_position)) - center
        if args.follow and calibration is None:
            spread = np.array([5.0, -5.0] * 4)
            calibration = HandCalibration(start - spread, start + spread)
        hand.sync_write_torque_enable(list(HAND_SERVO_IDS), [0] * 8)
    if args.dashboard:
        tracker.dashboard = HandDashboard(
            tracker.config,
            "实机跟随" if args.follow else "标定预览",
            hardware=hand is not None,
            servo_center=center if hand is not None and not args.follow else None,
        )
    tracker.connect()
    target = None
    last_seen = 0.0
    next_feedback = 0.0
    frame = 0
    try:
        while args.frames is None or frame < args.frames:
            flexion, captured_at = tracker.read()
            if tracker.dashboard:
                if tracker.dashboard.stop_requested:
                    break
                if hand is not None and not args.follow:
                    goals = tracker.dashboard.take_calibration_goals()
                    if goals:
                        if target is None:
                            current = read_hand_positions(hand.read_present_position)
                            target = np.rad2deg(current) - center
                            hand.sync_write_goal_speed(list(HAND_SERVO_IDS), [3] * 8)
                            hand.sync_write_goal_position(list(HAND_SERVO_IDS), current.tolist())
                            hand.sync_write_torque_enable(list(HAND_SERVO_IDS), [1] * 8)
                        indices = [index for finger in goals for index in (finger * 2, finger * 2 + 1)]
                        absolute = np.array(list(goals.values())).ravel()
                        target[indices] = absolute - np.asarray(center)[indices]
                        hand.sync_write_goal_position(
                            [HAND_SERVO_IDS[index] for index in indices], np.deg2rad(absolute).tolist()
                        )
                if args.follow and flexion is not None:
                    if target is None:
                        target = np.rad2deg(read_hand_positions(hand.read_present_position)) - center
                        hand.sync_write_goal_speed(list(HAND_SERVO_IDS), [3] * 8)
                        hand.sync_write_goal_position(
                            list(HAND_SERVO_IDS), np.deg2rad(center + target).tolist()
                        )
                        hand.sync_write_torque_enable(list(HAND_SERVO_IDS), [1] * 8)
                    desired = calibration.angles(flexion)
                    delta = (desired - target).reshape(4, 2)
                    scale = np.maximum(1.0, np.max(np.abs(delta), axis=1, keepdims=True) / 0.6)
                    target = target + (delta / scale).ravel()
                    hand.sync_write_goal_position(list(HAND_SERVO_IDS), np.deg2rad(center + target).tolist())
                    last_seen = captured_at
                elif (
                    args.follow
                    and target is not None
                    and captured_at - last_seen > tracker.config.hold_timeout_s
                ):
                    hand.sync_write_torque_enable(list(HAND_SERVO_IDS), [0] * 8)
                    target = None
                if hand is not None and captured_at >= next_feedback:
                    measured = np.rad2deg(read_hand_positions(hand.read_present_position)) - center
                    phase = None
                    if args.follow:
                        phase = "实机跟随 · 已使能" if target is not None else "实机跟随 · 等待人手"
                    tracker.dashboard.telemetry(
                        measured.tolist(), None if target is None else target.tolist(), phase
                    )
                    next_feedback = time.monotonic() + 0.1
                frame += 1
                continue
            output: dict[str, object] = {"timestamp_s": captured_at, "tracked": flexion is not None}
            if flexion is not None:
                output["flexion"] = dict(
                    zip(("index", "middle", "ring", "thumb"), flexion.tolist(), strict=True)
                )
                if calibration is not None:
                    angles = calibration.angles(flexion)
                    output["target_deg"] = dict(zip(HAND_SERVO_NAMES, angles.tolist(), strict=True))
            print(json.dumps(output), flush=True)
            frame += 1
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if hand is not None:
                hand.sync_write_torque_enable(list(HAND_SERVO_IDS), [0] * 8)
        finally:
            tracker.disconnect()


if __name__ == "__main__":
    main()
