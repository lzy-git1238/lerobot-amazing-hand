"""Replay one hand-only Juxi LeRobot episode on the eight SCS0009 servos."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rustypot import Scs0009PyController

from lerobot.utils.amazinghand import (
    HAND_SERVO_IDS,
    HAND_SERVO_NAMES,
    JUXI_SHAFT_LIMIT_DEG,
    read_hand_positions,
)


def load_episode(root: Path, episode: int) -> tuple[np.ndarray, np.ndarray]:
    info = json.loads((root / "meta" / "info.json").read_text())
    if info["robot_type"] != "amazinghand_juxi_hand_only_test":
        raise ValueError("Expected a hand-only Juxi dataset with absolute servo angles.")
    if info["features"]["action"]["names"] != list(HAND_SERVO_NAMES):
        raise ValueError("Expected the eight servo targets in ID order.")

    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise ValueError("The dataset contains no Parquet files.")
    table = pq.read_table(
        [str(path) for path in files],
        columns=["frame_index", "timestamp", "action"],
        filters=[("episode_index", "=", episode)],
    ).sort_by("frame_index")
    timestamps = table["timestamp"].to_numpy().astype(float)
    actions = np.asarray(table["action"].to_pylist(), dtype=float)
    if (
        not len(timestamps)
        or actions.shape != (len(timestamps), len(HAND_SERVO_IDS))
        or not np.isfinite(actions).all()
        or np.max(np.abs(actions)) > JUXI_SHAFT_LIMIT_DEG
    ):
        raise ValueError("Episode is empty or contains invalid servo targets.")
    if (
        not np.isfinite(timestamps).all()
        or not np.array_equal(table["frame_index"].to_numpy(), np.arange(len(timestamps)))
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError("Episode has invalid timestamps or missing frames.")
    return timestamps - timestamps[0], actions


def replay(port: str, timestamps: np.ndarray, actions: np.ndarray) -> None:
    controller = Scs0009PyController(serial_port=port, baudrate=1_000_000, timeout=0.15)
    ids = list(HAND_SERVO_IDS)
    feedback = []
    try:
        current = read_hand_positions(controller.read_present_position)
        controller.sync_write_goal_speed(ids, [3] * len(ids))
        controller.sync_write_goal_position(ids, current.tolist())
        controller.sync_write_torque_enable(ids, [1] * len(ids))

        # Seed the recorded trajectory gradually from the actual starting pose.
        first = np.deg2rad(actions[0])
        steps = max(1, int(np.ceil(np.rad2deg(np.max(np.abs(first - current))) / 3)))
        for position in np.linspace(current, first, steps + 1)[1:]:
            controller.sync_write_goal_position(ids, position.tolist())
            time.sleep(0.05)
        time.sleep(0.15)

        start = time.monotonic()
        for timestamp, position in zip(timestamps, actions, strict=True):
            time.sleep(max(0.0, start + float(timestamp) - time.monotonic()))
            controller.sync_write_goal_position(ids, np.deg2rad(position).tolist())
            feedback.append(np.rad2deg(read_hand_positions(controller.read_present_position)))
        elapsed = time.monotonic() - start
        time.sleep(0.20)
        print(f"Replayed {len(actions)} frames in {elapsed:.2f} s.")
        print("Measured travel by servo (deg):", np.ptp(feedback, axis=0).round(2).tolist())
    finally:
        controller.sync_write_torque_enable(ids, [0] * len(ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="LeRobot dataset root")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--hand-port", help="Serial port, e.g. /dev/ttyACM0")
    args = parser.parse_args()
    timestamps, actions = load_episode(args.dataset, args.episode)
    print(f"Validated episode {args.episode}: {len(actions)} frames, {timestamps[-1]:.2f} s.")
    if args.hand_port:
        replay(args.hand_port, timestamps, actions)


if __name__ == "__main__":
    main()
