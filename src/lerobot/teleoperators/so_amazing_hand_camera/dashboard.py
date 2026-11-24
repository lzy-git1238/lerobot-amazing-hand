"""Loopback-only monitor for the camera already owned by LeRobot."""

import base64
import contextlib
import json
import logging
import os
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from lerobot.utils.amazinghand import (
    CALIBRATION_AXIS_LIMITS,
    CALIBRATION_PRESETS,
    HandCalibration,
    calibration_axes,
    calibration_servo_goals,
)

logger = logging.getLogger(__name__)


def save_fields(path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(fields)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        try:
            json.dump(data, output, indent=2, allow_nan=False)
            output.flush()
            os.replace(output.name, path)
        finally:
            Path(output.name).unlink(missing_ok=True)


def vision_path(config):
    return config.calibration_path.with_name(f"vision_{config.handedness.lower()}.json")


def load_vision(config):
    path = vision_path(config)
    if not path.exists():
        return None
    fields = json.loads(path.read_text())
    bounds = np.array([fields["open"], fields["close"]], dtype=float)
    if bounds.shape != (2, 4) or not np.isfinite(bounds).all() or np.any(bounds[1] - bounds[0] < 0.05):
        raise ValueError(f"Invalid vision calibration in {path}")
    return bounds


class HandDashboard:
    def __init__(self, config, mode, hardware=False, servo_center=None):
        self.config = config
        self.lock = threading.RLock()
        self.state = {
            "mode": mode,
            "hardware": hardware,
            "hand": config.handedness,
            "phase": mode,
            "points": [],
            "frame": None,
            "flexion": None,
            "target": None,
            "measured": None,
            "samples": {},
            "finger_axes": None,
            "axis_limits": CALIBRATION_AXIS_LIMITS,
        }
        self.servo_center = servo_center
        self.pending_goals = {}
        self.samples = {}
        self.raw = None
        self.last_frame = self.last_motor = 0.0
        self.stop_requested = False
        self.bounds = load_vision(config)
        self.calibration = (
            HandCalibration.load(config.calibration_path) if config.calibration_path.exists() else None
        )
        self.server = None

    def start(self):
        dashboard = self
        page = Path(__file__).with_name("dashboard.html").read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, code, kind, data):
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(data)

            def do_GET(self):
                if self.headers.get("Host") != dashboard.address:
                    return self.send_error(403)
                if self.path == "/":
                    self.send(200, "text/html; charset=utf-8", page)
                elif self.path == "/state":
                    with dashboard.lock:
                        data = json.dumps(dashboard.state, allow_nan=False).encode()
                    self.send(200, "application/json", data)
                else:
                    self.send_error(404)

            def do_POST(self):
                if (
                    self.headers.get("Host") != dashboard.address
                    or self.headers.get("Origin") != dashboard.url
                ):
                    return self.send_error(403)
                try:
                    with dashboard.lock:
                        payload = None
                        if self.path == "/finger":
                            size = int(self.headers.get("Content-Length", "0"))
                            if not 0 < size <= 256:
                                raise ValueError("无效的手指控制请求。")
                            payload = json.loads(self.rfile.read(size))
                        message = dashboard.command(self.path, payload)
                except (ValueError, OSError, KeyError) as exc:
                    return self.send(400, "text/plain; charset=utf-8", str(exc).encode())
                self.send(200, "text/plain; charset=utf-8", message.encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", self.config.dashboard_port), Handler)
        self.address = f"127.0.0.1:{self.server.server_port}"
        self.url = f"http://{self.address}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"AmazingHand: {self.url}", flush=True)
        threading.Thread(target=webbrowser.open, args=(self.url,), daemon=True).start()

    def close(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    def command(self, path, payload=None):
        if path == "/stop" and self.state["mode"] == "实机跟随":
            self.stop_requested = True
            return "正在停止跟随"
        if self.state["mode"] != "标定预览":
            raise ValueError("请在标定预览中修改端点。")
        if path == "/stop":
            self.stop_requested = True
            return "正在释放设备"
        if path == "/finger" or path.startswith("/pose/"):
            return self.queue_calibration_move(path, payload)
        key = path.removeprefix("/")
        if key not in ("vision-open", "vision-close", "servo-open", "servo-close"):
            raise ValueError("未知操作")
        kind, pose = key.split("-")
        is_vision = kind == "vision"
        if not is_vision and self.pending_goals:
            raise ValueError("舵机正在调整，请稍等再采集。")
        values = self.raw if is_vision else self.state["measured"]
        sample_time = self.last_frame if is_vision else self.last_motor
        if values is None or time.monotonic() - sample_time > 0.6:
            raise ValueError("等待新的有效读数后再采集。")
        if not is_vision and self.servo_center is not None:
            target = self.state["target"]
            if target is None:
                raise ValueError("请先用滑块或参考姿态驱动灵巧手。")
            if np.max(np.abs(np.asarray(values) - target)) > 3:
                raise ValueError("实测角度还未到达目标。若已停止运动，请调整滑块直到目标可达。")
        self.samples[key] = list(values)
        self.state["samples"][key] = True
        other = f"{kind}-{'close' if pose == 'open' else 'open'}"
        if other not in self.samples:
            return "已采集，请采集另一端点"
        opened, closed = np.array([self.samples[kind + "-open"], self.samples[kind + "-close"]])
        if is_vision:
            if np.any(closed - opened < 0.05):
                raise ValueError("至少一指的屈伸幅度不足，请重新采集。")
            save_fields(vision_path(self.config), {"open": opened.tolist(), "close": closed.tolist()})
            self.bounds = np.stack([opened, closed])
        else:
            if np.any(np.max(np.abs(closed - opened).reshape(4, 2), axis=1) < 1):
                raise ValueError("至少一指的两个舵机端点过近，请重新采集。")
            save_fields(
                self.config.calibration_path,
                {"hand_open_angles": opened.tolist(), "hand_close_angles": closed.tolist()},
            )
            self.calibration = HandCalibration(opened, closed)
        for suffix in ("open", "close"):
            self.samples.pop(kind + "-" + suffix)
            self.state["samples"].pop(kind + "-" + suffix)
        return "两个端点已保存"

    def queue_calibration_move(self, path, payload):
        axes = self.state["finger_axes"]
        if axes is None:
            raise ValueError("等待灵巧手位置反馈。")
        if path == "/finger":
            if not isinstance(payload, dict):
                raise ValueError("无效的手指控制请求。")
            finger, axis, value = payload.get("finger"), payload.get("axis"), payload.get("value")
            if type(finger) is not int or not 0 <= finger < 4 or type(axis) is not int or axis not in (0, 1):
                raise ValueError("手指或控制轴无效。")
            lower, upper = CALIBRATION_AXIS_LIMITS[axis]
            if type(value) not in (int, float) or not lower <= value <= upper:
                raise ValueError("角度超出控制范围。")
            axes[finger][axis] = value
            self.pending_goals[finger] = calibration_servo_goals(*axes[finger])
        else:
            flexions = CALIBRATION_PRESETS[path.removeprefix("/pose/")]
            for finger, flexion in enumerate(flexions):
                axes[finger] = [flexion, 0]
                self.pending_goals[finger] = calibration_servo_goals(flexion, 0)
        self.last_motor = 0.0
        return "已接收调整指令"

    def take_calibration_goals(self):
        with self.lock:
            goals, self.pending_goals = self.pending_goals, {}
            return goals

    def frame(self, rgb, points, raw, flexion):
        now = time.monotonic()
        if now - self.last_frame < 0.09:
            return
        encoded = None
        if rgb is not None:
            ok, jpg = cv2.imencode(
                ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 72]
            )
            if ok:
                encoded = base64.b64encode(jpg).decode("ascii")
        with self.lock:
            self.last_frame = now
            self.raw = None if raw is None else raw.tolist()
            self.state.update(
                frame=encoded, points=points, flexion=None if flexion is None else flexion.tolist()
            )
            if self.state["mode"] == "标定预览" and self.servo_center is None:
                self.state["target"] = (
                    self.calibration.angles(flexion).tolist()
                    if self.calibration and flexion is not None
                    else None
                )

    def telemetry(self, measured, target=None, phase=None):
        with self.lock:
            self.state["measured"] = list(measured)
            self.last_motor = time.monotonic()
            if self.servo_center is not None and self.state["finger_axes"] is None:
                absolute = np.asarray(measured) + self.servo_center
                self.state["finger_axes"] = [
                    list(calibration_axes(odd, even)) for odd, even in absolute.reshape(4, 2)
                ]
            if target is not None:
                self.state["target"] = list(target)
            if phase:
                self.state["phase"] = phase
