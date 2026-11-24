"""Record one camera-controlled AmazingHand-only LeRobot episode."""

import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
from rustypot import Scs0009PyController

from lerobot.processor import make_default_processors
from lerobot.robots.so_amazing_hand import SO101AmazingHandFollower
from lerobot.teleoperators.so_amazing_hand_camera.juxi import JuxiHandBridge
from lerobot.teleoperators.so_amazing_hand_camera.so_amazing_hand_camera import SOAmazingHandCamera
from lerobot.teleoperators.so_amazing_hand_camera.tracker import HandTrackingLostError
from lerobot.utils.amazinghand import DEFAULT_JUXI_SOURCE, HAND_SERVO_IDS, HAND_SERVO_NAMES
from lerobot.utils.feature_utils import hw_to_dataset_features


class HandOnlyRobot(SO101AmazingHandFollower):
    def __init__(self, port: str):
        self.id = "hand-only-recording-test"
        self.config = SimpleNamespace(
            hand_control_mode="servo_juxi",
            hand_max_step_deg=6.0,
            num_read_retries=0,
        )
        self.bus = SimpleNamespace(is_connected=True, motors={}, sync_read=lambda *_args, **_kw: {})
        self.cameras = {}
        self.MiddlePos = [0.0] * 8
        self._last_hand_target = None
        self.hand = Scs0009PyController(serial_port=port, baudrate=1_000_000, timeout=0.15)

    def disconnect(self):
        try:
            self.hand.sync_write_torque_enable(list(HAND_SERVO_IDS), [0] * 8)
        finally:
            self.bus.is_connected = False


class BrowserTargets(SOAmazingHandCamera):
    def __init__(self, bridge):
        self.bridge = bridge

    @property
    def is_connected(self):
        return self.bridge.is_connected

    def get_action(self):
        return dict(zip(HAND_SERVO_NAMES, self.bridge.read(), strict=True))

    def disconnect(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-port", required=True)
    parser.add_argument("--juxi-source", type=Path, default=DEFAULT_JUXI_SOURCE)
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/datasets"))
    args = parser.parse_args()
    if args.seconds <= 0 or args.fps <= 0:
        parser.error("--seconds and --fps must be positive")

    # Newer PyAV removed API used by video-only type annotations and codec validation.
    # This example records only angle vectors, so those video APIs are never called.
    legacy_av = not hasattr(av, "option")
    annotation = patch.object(av, "option", SimpleNamespace(Option=object), create=True)
    with annotation if legacy_av else nullcontext():
        from lerobot.datasets import LeRobotDataset
        from lerobot.scripts.lerobot_record import record_loop

    fps = args.fps
    dataset_name = datetime.now().strftime("amazinghand_hand_only_live_%Y%m%d_%H%M%S")
    repo_id = "local/" + dataset_name
    root = args.output_root.resolve() / dataset_name
    hand = HandOnlyRobot(args.hand_port)
    bridge = JuxiHandBridge(args.juxi_source, port=8765, timeout=0.75)
    dataset = None
    try:
        hand.hand.sync_write_torque_enable(list(HAND_SERVO_IDS), [0] * 8)
        features = hw_to_dataset_features(hand.observation_features, "observation") | hw_to_dataset_features(
            hand.action_features, "action"
        )
        codec_check = patch("lerobot.datasets.check_video_encoder_parameters_pyav")
        with codec_check if legacy_av else nullcontext():
            dataset = LeRobotDataset.create(
                repo_id,
                fps,
                features,
                root=root,
                robot_type="amazinghand_juxi_hand_only_test",
                use_videos=False,
            )
        teleop_processor, action_processor, obs_processor = make_default_processors()
        bridge.connect()
        while True:
            bridge.wait_for_targets()
            start = hand._read_hand_angles()
            hand._last_hand_target = start
            hand._move_hand(start.tolist(), [3] * 8)
            hand.hand.sync_write_torque_enable(list(HAND_SERVO_IDS), [1] * 8)
            try:
                print(f"RECORDING: keep your hand in view for {args.seconds} seconds.", flush=True)
                record_loop(
                    robot=hand,
                    events={"exit_early": False},
                    fps=fps,
                    dataset=dataset,
                    teleop=BrowserTargets(bridge),
                    control_time_s=args.seconds,
                    single_task="Camera demonstration: open and close the AmazingHand",
                    teleop_action_processor=teleop_processor,
                    robot_action_processor=action_processor,
                    robot_observation_processor=obs_processor,
                )
            except HandTrackingLostError:
                hand.hand.sync_write_torque_enable(list(HAND_SERVO_IDS), [0] * 8)
                dataset.clear_episode_buffer()
                print(
                    "Tracking interrupted; the incomplete episode was discarded. Waiting to retry.",
                    flush=True,
                )
            else:
                hand.disconnect()
                dataset.save_episode()
                break
    finally:
        try:
            if hand.is_connected:
                hand.disconnect()
        finally:
            try:
                bridge.disconnect()
            finally:
                if dataset is not None:
                    dataset.finalize()

    print(f"Saved 1 episode and {dataset.num_frames} frames to {root}", flush=True)


if __name__ == "__main__":
    main()
