# Bundled AmazingHand browser assets

Source is Juxi Technology's AmazingHand project at
https://github.com/Juxi-Technology/AmazingHand.

The browser files were copied from the local source checkout on 2026-11-24.
Their upstream paths are preserved here. SHA256SUMS.json records imported hashes.

The Juxi browser application, Pollen Robotics-derived servo configuration, and
Google MediaPipe runtime and model are distributed under Apache License 2.0.
See LICENSE and UPSTREAM_NOTICE.md for attribution and component details.
MediaPipe includes the Google JavaScript runtime, WebAssembly binaries and
hand landmarker model for local use without a model download.

LeRobot serves these files from its installed Python package. juxi_bridge.js
replaces the page transport with a loopback connection to the recorder.
juxi.py extends the served tracking code to enable all four splay axes and
calibration validation. The imported tracking.js is preserved for comparison.
