/* Use the original page's transport interface with LeRobot in place of Web Serial. */
window.AH_BACKEND = "ws";
window.AH_MEDIAPIPE_BASE = "/mediapipe";

(() => {
  let config;
  let newest;
  let sending = false;
  let online = false;

  async function publish() {
    if (sending) return;
    sending = true;
    try {
      while (newest) {
        const goals = newest;
        newest = null;
        const reply = await fetch("/api/targets", {
          method: "POST",
          headers: {"X-LeRobot-Session": config.token},
          body: JSON.stringify(goals),
          signal: AbortSignal.timeout(1000),
        });
        if (!reply.ok) throw Error("LeRobot 拒绝了舵机目标");
      }
      online = true;
    } catch {
      online = false;
      newest = null;
    } finally {
      sending = false;
    }
  }

  function sendTrackingAngles(angles) {
    if (!config) return false;
    const positions = config.hand.fingers.flatMap(finger => angles[finger.key] || []);
    if (positions.length !== 8 || !positions.every(Number.isFinite)) return false;
    const safe = positions.map(angle => window.AH_COMMON.clampPose(config.limits, angle));
    newest = window.AH_COMMON.poseToServoGoals(config.hand, safe);
    void publish();
    return online;
  }

  window.AH_WS = {
    async init({onMessage}) {
      const response = await fetch("/api/config");
      if (!response.ok) return;
      config = await response.json();
      window.AH.sendAngles = sendTrackingAngles;
      onMessage({
        type: "state", enabled: ["right"], target: "right", driver: "scs0009",
        hands: {right: {...config.hand, servo_ids: config.ids, torque: false}},
        poses: {}, middle: config.middle, positions: {right: config.middle.positions},
        axes: config.axes, limits: config.limits, camera: config.camera,
        capabilities: {torque: false, speed: false},
        port: {current: null, connected: true, available: []},
      });
      const camera = document.getElementById("tab-camera");
      camera.click();
      document.querySelector(".port-bar").style.display = "none";
      document.querySelector("nav.tabs").style.display = "none";
      const note = document.createElement("p");
      note.className = "muted cam-hint";
      note.textContent = "LeRobot 八自由度采集 · 重采三个姿势。张开与合拢时，中指和拇指都要有侧向位移。中指侧摆 −20°～20°，拇指 0°～−30°。串口由 LeRobot 管理";
      document.getElementById("panel-camera").prepend(note);
      const heartbeat = setInterval(async () => {
        try {
          const reply = await fetch("/api/config", {signal: AbortSignal.timeout(1000)});
          if (reply.ok && (await reply.json()).token === config.token) return;
        } catch {
          // The LeRobot process has exited.
        }
        clearInterval(heartbeat);
        online = false;
        newest = null;
        config = null;
        document.getElementById("btn-cam-stop").click();
        document.getElementById("btn-cam-start").disabled = true;
        note.textContent = "LeRobot 已断开。重新启动程序后再打开本页。";
      }, 1500);
    },
    send() { return false; },
    isConnected() { return online; },
  };
})();
