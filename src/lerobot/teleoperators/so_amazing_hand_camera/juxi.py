"""Receive actuator targets from the unmodified AmazingHand browser tracker."""

import io
import json
import math
import secrets
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from lerobot.utils.amazinghand import HAND_SERVO_IDS, JUXI_SHAFT_LIMIT_DEG

from .tracker import HandTrackingLostError


def full_dof_tracking(source: str) -> str:
    """Enable four calibrated splay axes in the locally served Juxi tracker."""
    guard = 'if (key !== "thumb") {'
    decay = "    side *= (1 - curl);"
    if source.count(guard) != 1 or source.count("  const R = cal;") != 1 or source.count(decay) != 1:
        raise ValueError("Unsupported Juxi tracking.js version for eight-axis recording.")
    source = source.replace(guard, 'if (key !== "thumb" || state.camera?.full_dof) {')
    # Thumb opposition couples the flexion estimate to lateral motion. Applying
    # the long-finger curl suppression here was cancelling the requested splay.
    source = source.replace(
        decay,
        '    if (key !== "thumb" || !state.camera?.full_dof) side *= (1 - curl);',
    )
    return source.replace(
        "  const R = cal;",
        """  const R = cal;
  if (state.camera?.full_dof) {
    const missing = ["middle", "thumb"].filter(
      key => Math.abs(R.splayHi[key] - R.splayLo[key]) < 3
    );
    if (missing.length) {
      const status = el("cal-state");
      if (status) {
        status.textContent = "侧摆标定幅度不足：" + missing.map(fingerLabel).join("、") +
          "。请重采伸直张开与伸直合拢，让中指和拇指侧向移动。";
        status.className = "err";
      }
      return null;
    }
  }""",
    )


def read_shaft_targets(goals) -> list[float]:
    if not isinstance(goals, list) or len(goals) != len(HAND_SERVO_IDS):
        raise ValueError("Expected eight SCS0009 servo targets.")
    by_id = {}
    for entry in goals:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError("Expected [servo ID, absolute degrees].")
        servo_id, degrees = entry
        if (
            type(servo_id) is not int
            or servo_id not in HAND_SERVO_IDS
            or servo_id in by_id
            or type(degrees) not in (int, float)
            or not -JUXI_SHAFT_LIMIT_DEG <= degrees <= JUXI_SHAFT_LIMIT_DEG
        ):
            raise ValueError("Invalid SCS0009 servo ID or shaft angle.")
        by_id[servo_id] = float(degrees)
    return [by_id[servo_id] for servo_id in HAND_SERVO_IDS]


class JuxiHandBridge:
    def __init__(self, source: Path, port: int, timeout: float):
        self.source = source.expanduser().resolve()
        self.web = self.source / "Amazinghand-web-io"
        self.media = self.source / "Amazinghand-web" / "web" / "mediapipe"
        self.port = port
        self.timeout = timeout
        self.server = None
        self.token = secrets.token_urlsafe(24)
        self.condition = threading.Condition()
        self.goals = None
        self.received_at = -math.inf

    @property
    def is_connected(self) -> bool:
        return self.server is not None

    def connect(self):
        root, bridge = self.web, self
        for file in (root / "tracking.js", root / "common.js", self.media / "hand_landmarker.task"):
            if not file.is_file():
                raise FileNotFoundError(f"AmazingHand source asset missing: {file}")
        page = (root / "index.html").read_text(encoding="utf-8")
        if '<script src="ws.js"></script>' not in page:
            raise ValueError("Unsupported AmazingHand page: expected its ws.js transport entry.")
        page = page.replace('<script src="ws.js"></script>', '<script src="/lerobot.js"></script>')
        adapter = Path(__file__).with_name("juxi_bridge.js").read_bytes()
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        camera = config["camera"]
        camera["full_dof"] = True
        camera["side_together"] = {**camera["side_together"], "middle": -20, "thumb": 0}
        camera["side_spread"] = {**camera["side_spread"], "middle": 20, "thumb": -45}
        tracking = full_dof_tracking((root / "tracking.js").read_text(encoding="utf-8")).encode()
        hand = config["hands"]["right"]
        ids = [servo_id for finger in hand["fingers"] for servo_id in finger["ids"]]
        if sorted(ids) != list(HAND_SERVO_IDS) or hand["angle_offset_deg"] != 0:
            raise ValueError("Juxi integration requires the right SCS0009 hand, IDs 1–8, with zero offset.")
        state = json.dumps(
            {
                "token": self.token,
                "camera": camera,
                "axes": config["axes"],
                "limits": config["limits"],
                "middle": config["middle"],
                "hand": hand,
                "ids": ids,
            }
        ).encode()

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(root), **kwargs)

            def log_message(self, *args):
                pass

            def translate_path(self, path):
                path = unquote(urlsplit(path).path)
                if path.startswith("/mediapipe/"):
                    directory, suffix = bridge.media, path.removeprefix("/mediapipe/")
                else:
                    directory, suffix = root, path.lstrip("/")
                resolved = (directory / suffix).resolve()
                if not resolved.is_relative_to(directory):
                    return str(directory / "__forbidden__")
                return str(resolved)

            def list_directory(self, path):
                return self.send_error(404)

            def end_headers(self):
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Frame-Options", "DENY")
                super().end_headers()

            def send_head(self):
                if self.headers.get("Host") != bridge.address:
                    return self.send_error(403)
                path = urlsplit(self.path).path
                if path in ("/", "/index.html"):
                    data, kind = page.encode(), "text/html; charset=utf-8"
                elif path == "/lerobot.js":
                    data, kind = adapter, "application/javascript"
                elif path == "/api/config":
                    data, kind = state, "application/json"
                elif path == "/tracking.js":
                    data, kind = tracking, "application/javascript"
                else:
                    return super().send_head()
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return io.BytesIO(data)

            def do_POST(self):
                if (
                    self.headers.get("Host") != bridge.address
                    or self.headers.get("Origin") != bridge.url
                    or self.headers.get("X-LeRobot-Session") != bridge.token
                ):
                    return self.send_error(403)
                if self.path != "/api/targets":
                    return self.send_error(404)
                self.connection.settimeout(1)
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 1024:
                        raise ValueError("Expected a short JSON servo target message.")
                    goals = read_shaft_targets(json.loads(self.rfile.read(length)))
                except (ValueError, OSError, RecursionError) as exc:
                    return self.send_error(400, str(exc))
                with bridge.condition:
                    bridge.goals = goals
                    bridge.received_at = time.monotonic()
                    bridge.condition.notify_all()
                self.send_response(204)
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.address = f"127.0.0.1:{self.server.server_port}"
        self.url = f"http://{self.address}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"AmazingHand / LeRobot: {self.url}", flush=True)
        threading.Thread(target=webbrowser.open, args=(self.url,), daemon=True).start()

    def wait_for_targets(self):
        print("Start the camera and finish all three poses in the AmazingHand browser page.", flush=True)
        with self.condition:
            while time.monotonic() - self.received_at > self.timeout:
                self.condition.wait(timeout=0.2)

    def read(self) -> list[float]:
        with self.condition:
            if self.goals is None or time.monotonic() - self.received_at > self.timeout:
                raise HandTrackingLostError(
                    "AmazingHand browser tracking lost. Keep the calibrated hand in view."
                )
            return self.goals.copy()

    def disconnect(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        with self.condition:
            self.goals = None
            self.received_at = -math.inf
