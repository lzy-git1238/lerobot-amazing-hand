from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.utils.amazinghand import (
    HandCalibration,
    calibration_axes,
    calibration_servo_goals,
    finger_flexion,
    read_hand_positions,
)


def test_right_hand_calibration_axes_match_reference_controller():
    # The reference swaps the right-hand pair, then negates the even-ID servo.
    assert calibration_servo_goals(30, 10) == (20, -40)
    assert calibration_axes(20, -40) == (30, 10)
    assert calibration_servo_goals(90, 45) == (45, -120)


def test_default_browser_assets_are_bundled_and_complete():
    import hashlib
    import json

    from lerobot.teleoperators.so_amazing_hand_camera.configuration import HandCameraConfig
    from lerobot.teleoperators.so_amazing_hand_camera.juxi import full_dof_tracking
    from lerobot.utils.amazinghand import DEFAULT_JUXI_SOURCE

    config = HandCameraConfig()
    assert config.juxi_source == DEFAULT_JUXI_SOURCE
    manifest = json.loads((DEFAULT_JUXI_SOURCE / "SHA256SUMS.json").read_text())
    for filename, expected_hash in manifest.items():
        assert hashlib.sha256((DEFAULT_JUXI_SOURCE / filename).read_bytes()).hexdigest() == expected_hash
    source = (DEFAULT_JUXI_SOURCE / "Amazinghand-web-io/tracking.js").read_text()
    full_dof_tracking(source)


def test_browser_thumb_splay_preserves_its_range_while_flexing():
    import json
    import shutil
    import subprocess

    from lerobot.teleoperators.so_amazing_hand_camera.juxi import full_dof_tracking
    from lerobot.utils.amazinghand import DEFAULT_JUXI_SOURCE

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the browser mapping regression")
    source = (DEFAULT_JUXI_SOURCE / "Amazinghand-web-io/tracking.js").read_text()
    algorithm = full_dof_tracking(source).split("async function listCameras()")[0]
    harness = """
const state={camera:{full_dof:true,flex_straight:-50,
  flex_fist:{ring:90,middle:90,pointer:90,thumb:18},
  side_together:{ring:-45,middle:-20,pointer:45,thumb:0},
  side_spread:{ring:45,middle:20,pointer:-45,thumb:-45}}};
const window={I18N:{t:()=>'',pick:(_g,k)=>k},
  AH:{target:()=> 'right',driver:()=> 'scs0009'}};
"""
    probe = """
handFrame=()=>({});
let flex=0;
flexionAt=()=>flex;
splayDeg=()=>20;
cal={flexLo:{ring:0,middle:0,pointer:0,thumb:0},
     flexHi:{ring:1,middle:1,pointer:1,thumb:1},
     splayLo:{ring:0,middle:0,pointer:0,thumb:0},
     splayHi:{ring:20,middle:20,pointer:20,thumb:20}};
const result=[];
for(const curl of [0,0.5,1]) {
  flex=curl;
  let output;
  for(let frame=0;frame<120;frame++) output=toAngles([],[], 'Right');
  result.push({thumb:(output.thumb[0]-output.thumb[1])/2,
               pointer:(output.pointer[0]-output.pointer[1])/2});
}
console.log(JSON.stringify(result));
"""
    result = subprocess.run(
        [node], input=harness + algorithm + probe, text=True, capture_output=True, check=True
    )
    values = json.loads(result.stdout)
    assert [value["thumb"] for value in values] == [-45, -45, -45]
    assert values[-1]["pointer"] == 0


@pytest.mark.parametrize("as_list", [False, True])
def test_position_feedback_accepts_single_register_values(as_list):
    def read_position(servo_id):
        value = servo_id / 10.0
        return [value] if as_list else value

    assert read_hand_positions(read_position) == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


def test_distinct_curls_survive_a_mirrored_camera():
    landmarks = np.zeros((21, 3))
    for mcp, x in ((1, -1), (5, 1), (9, 2), (13, 3)):
        landmarks[mcp : mcp + 4, 0] = x
        landmarks[mcp : mcp + 4, 1] = [1, 2, 3, 4]
    landmarks[7:9, 1:] = [[2, 1], [2, 2]]

    expected = [1, 0, 0, 0]
    assert finger_flexion(landmarks) == pytest.approx(expected)
    landmarks[:, 0] *= -1
    assert finger_flexion(landmarks) == pytest.approx(expected)


def test_degenerate_landmarks_are_unusable():
    with pytest.raises(ValueError, match="landmarks"):
        finger_flexion(np.zeros((21, 3)))


def test_coupled_servo_angles_respect_reversed_endpoints(tmp_path):
    path = tmp_path / "hand_angles.json"
    path.write_text(
        '{"hand_open_angles": [-20, 20, -20, 20, -20, 20, -20, 20],'
        ' "hand_close_angles": [40, -40, 40, -40, 40, -40, 40, -40]}'
    )
    calibration = HandCalibration.load(path)
    assert calibration.angles([0, 0.5, 1, 0]) == pytest.approx([-20, 20, 10, -10, 40, -40, -20, 20])
    assert calibration.project([70, -70, -30, 30, 10, 20, 10, -10]) == pytest.approx(
        [40, -40, -20, 20, -5, 5, 10, -10]
    )


def test_camera_teleop_holds_brief_tracking_loss_then_stops():
    from lerobot.teleoperators.so_amazing_hand_camera.so_amazing_hand_camera import (
        SOAmazingHandCamera,
    )

    teleop = object.__new__(SOAmazingHandCamera)
    teleop.config = SimpleNamespace(hand_camera=SimpleNamespace(hold_timeout_s=0.5, smoothing=1.0))
    teleop._last_flexion = None
    teleop._last_detection = None
    teleop.hand_calibration = HandCalibration(np.zeros(8), np.array([40.0, -40.0] * 4))
    teleop.leader = SimpleNamespace(
        is_connected=True, get_action=lambda: {"wrist_roll.pos": 9.0, "gripper.pos": 100.0}
    )
    detections = iter([(np.array([1.0, 0.0, 0.0, 0.0]), 1.0), (None, 1.3), (None, 1.6)])
    teleop.tracker = SimpleNamespace(
        is_connected=True,
        read=lambda: next(detections),
    )

    action = teleop.get_action()
    assert action["wrist_roll.pos"] == 9.0
    assert "gripper.pos" not in action
    assert list(action.values())[1:] == [40.0, -40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert teleop.get_action() == action
    with pytest.raises(RuntimeError, match="tracking lost"):
        teleop.get_action()


def test_tracker_only_uses_the_selected_hand(monkeypatch):
    from lerobot.teleoperators.so_amazing_hand_camera import tracker

    hand_tracker = object.__new__(tracker.CameraHandTracker)
    hand_tracker.config = SimpleNamespace(handedness="Right", hold_timeout_s=0.5)
    hand_tracker.camera = SimpleNamespace(async_read=lambda **_kwargs: np.zeros((4, 4, 3), dtype=np.uint8))
    hand_tracker.dashboard = None
    hand_tracker.vision_bounds = None
    hand_tracker._last_timestamp_ms = -1
    straight = [SimpleNamespace(x=0.0, y=float(i), z=0.0) for i in range(21)]
    hand_tracker._landmarker = SimpleNamespace(
        detect_for_video=lambda _image, _timestamp: SimpleNamespace(
            handedness=[[SimpleNamespace(category_name="Left")], [SimpleNamespace(category_name="Right")]],
            hand_world_landmarks=[[], straight],
        )
    )
    monkeypatch.setattr(
        tracker,
        "mp",
        SimpleNamespace(Image=lambda **kwargs: kwargs, ImageFormat=SimpleNamespace(SRGB=0)),
        raising=False,
    )
    assert hand_tracker.read()[0] == pytest.approx([0, 0, 0, 0])
    hand_tracker.config.handedness = "Left"
    assert hand_tracker.read()[0] is None


def test_servo_mode_exposes_measured_positions_and_applied_targets():
    from lerobot.robots.so_amazing_hand import SO101AmazingHandFollower
    from lerobot.utils.feature_utils import hw_to_dataset_features

    class Hand:
        def read_present_position(self, servo_id):
            assert 1 <= servo_id <= 8
            return np.deg2rad(10 + servo_id)

        def sync_write_goal_speed(self, ids, speeds):
            pass

        def sync_write_goal_position(self, ids, radians):
            self.degrees = np.rad2deg(radians)

    robot = object.__new__(SO101AmazingHandFollower)
    robot.id = "test_follower"
    arm_motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    robot.bus = SimpleNamespace(
        is_connected=True,
        motors=dict.fromkeys(arm_motors),
        sync_read=lambda *_args, **_kwargs: dict.fromkeys(arm_motors, 3.0),
        sync_write=lambda *_args: None,
        disconnect=lambda *_args: None,
    )
    robot.cameras = {}
    robot.config = SimpleNamespace(
        hand_control_mode="servo", hand_max_step_deg=6.0, num_read_retries=0, max_relative_target=None
    )
    robot.hand = Hand()
    robot._servo_calibration = HandCalibration(np.zeros(8), np.array([40.0, -40.0] * 4))
    robot._last_hand_target = np.zeros(8)
    robot.MiddlePos = [10.0] * 8
    robot.MaxSpeed = 5
    keys = [f"hand_servo_{i}.pos" for i in range(1, 9)]

    assert hw_to_dataset_features(robot.action_features, "action")["action"]["shape"] == (13,)
    obs = robot.get_observation()
    assert [obs[key] for key in keys] == pytest.approx([1, 2, 3, 4, 5, 6, 7, 8])
    action = dict.fromkeys(arm_motors, 9.0)
    action = {f"{key}.pos": value for key, value in action.items()}
    sent = robot.send_action(action | dict(zip(keys, [80, -80, 0, 0, 0, 0, 0, 0], strict=True)))
    assert list(sent.values()) == pytest.approx([9, 9, 9, 9, 9, 6, -6, 0, 0, 0, 0, 0, 0])
    assert robot.hand.degrees == pytest.approx([16, 4, 10, 10, 10, 10, 10, 10])
    robot.config.hand_control_mode = "servo_juxi"
    robot._last_hand_target = np.zeros(8)
    robot.MiddlePos = [0.0] * 8
    # Same-sign actuator targets carry side-to-side motion.
    sent = robot.send_action(action | dict(zip(keys, [-10, -10, 0, 0, 0, 0, 0, 0], strict=True)))
    assert [sent[key] for key in keys] == pytest.approx([-6, -6, 0, 0, 0, 0, 0, 0])
    assert robot.hand.degrees == pytest.approx([-6, -6, 0, 0, 0, 0, 0, 0])
    robot.config.hand_control_mode = "gripper"
    assert list(robot._motors_ft) == [f"{key}.pos" for key in arm_motors] + ["gripper.pos"]
    robot.bus.is_connected = False


def test_juxi_receives_logical_finger_order_as_numeric_servo_order():
    from lerobot.teleoperators.so_amazing_hand_camera.juxi import read_shaft_targets

    goals = [[5, -1], [6, -2], [3, -3], [4, -4], [1, -5], [2, -6], [7, -7], [8, -8]]
    assert read_shaft_targets(goals) == [-5, -6, -3, -4, -1, -2, -7, -8]
    goals[0] = [1, -1]
    with pytest.raises(ValueError, match="servo ID"):
        read_shaft_targets(goals)
    goals[0] = [5, float("nan")]
    with pytest.raises(ValueError, match="shaft angle"):
        read_shaft_targets(goals)


def test_recording_uses_applied_targets_for_the_servo_dataset(monkeypatch):
    pytest.importorskip("datasets")

    from lerobot.robots.so_amazing_hand import SO101AmazingHandFollower
    from lerobot.scripts.lerobot_record import record_loop
    from lerobot.teleoperators.so_amazing_hand_camera.so_amazing_hand_camera import SOAmazingHandCamera
    from lerobot.utils.feature_utils import hw_to_dataset_features

    robot = object.__new__(SO101AmazingHandFollower)
    robot.config = SimpleNamespace(hand_control_mode="servo")
    joint_names = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
    ]
    joint_names.extend(f"hand_servo_{i}.pos" for i in range(1, 9))
    robot.get_observation = lambda: dict.fromkeys(joint_names, 1.0)
    robot.send_action = lambda _action: dict.fromkeys(joint_names, 2.0)
    teleop = object.__new__(SOAmazingHandCamera)
    teleop.get_action = lambda: dict.fromkeys(joint_names, 3.0)
    features = dict.fromkeys(joint_names, float)
    frames = []
    events = {"exit_early": False}

    def add_frame(frame):
        frames.append(frame)
        events["exit_early"] = True

    dataset = SimpleNamespace(
        fps=30,
        features=hw_to_dataset_features(features, "observation") | hw_to_dataset_features(features, "action"),
        add_frame=add_frame,
    )
    monkeypatch.setattr("lerobot.scripts.lerobot_record.precise_sleep", lambda _seconds: None)
    record_loop(
        robot=robot,
        events=events,
        fps=30,
        teleop_action_processor=lambda value: value[0],
        robot_action_processor=lambda value: value[0],
        robot_observation_processor=lambda obs: obs,
        dataset=dataset,
        teleop=teleop,
        control_time_s=1,
        single_task="Pick up the cube",
    )
    assert len(frames) == 1
    assert frames[0]["observation.state"].tolist() == [1.0] * 13
    assert frames[0]["action"].tolist() == [2.0] * 13
