# AmazingHand Camera Teleoperation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement the dependent tasks in order.

**Goal:** Control AmazingHand with front-camera hand tracking and collect synchronized SO-101 and hand data for LeRobot.

**Architecture:** Share the hand calibration and actuator field names between the camera and the robot. Compose the camera with the existing SO-101 leader, and use the follower's sent action as the training label in the explicit servo mode.

**Tech Stack:** Python 3.12+, OpenCV, MediaPipe Tasks, rustypot, LeRobotDataset and ACT.

**Spec:** `docs/superpowers/specs/2026-11-20-amazinghand-camera-teleoperation-design.md`

## Global Constraints

- Four independently flexing fingers, each with two coordinated actuators. Hand angles use degrees relative to the existing center offsets.
- Preserve the default six-channel follower schema. The new schema uses five arm degrees followed by eight named hand servo positions.
- Use an explicit hand calibration for the servo mode and preserve real encoder feedback under failures.

## Review Focus

- Nonfinite or degenerate landmarks, left/right filtering, and tracking timeouts.
- Reversed servo endpoints, coupled target limits, and safe first writes.
- Optional dependency absence when running regular LeRobot commands.
- Recorded labels match the applied targets, including arm clipping.
- Existing datasets and the default gripper behavior preserve their schema.

## Tasks

- [ ] Add calibration validation and four-finger geometry in `lerobot/utils/amazinghand.py`. Write a test for distinct finger curls, mirrored or malformed landmarks, and reversed servo endpoints.
- [ ] Add `so101_amazing_hand_camera` in `lerobot/teleoperators/so_amazing_hand_camera/`. Use MediaPipe video timestamps, a bounded last-pose hold, and the existing SO-101 leader. Add a serial-free JSON preview entry point. Verify that CLI config discovery works without MediaPipe installed.
- [ ] Add `hand_control_mode=servo` to the existing follower. Expose `hand_servo_1.pos` through `hand_servo_8.pos`, use the SCS0009 position feedback and consistent zero offsets, and project each pair onto the calibrated flexion path before limiting motion. Exercise a fake controller and the default legacy schema.
- [ ] Integrate actual sent labels in `lerobot_record.py` for servo mode. Verify the regular LeRobot frame schema builds 13 values from the measured state and sent action. Document preview, teleoperation, recording, training, and rollout in the Chinese README.
- [ ] Run focused tests, static checks, CLI smoke tests, and report which hardware checks remain.
