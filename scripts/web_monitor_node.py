#!/usr/bin/env python3
"""
Gaussian-LIC web monitor.

Endpoints:
  GET /            – dashboard HTML (Three.js 3-D view)
  GET /lidar.bin   – accumulated LiDAR point cloud (binary, polled on version change)
  GET /input.jpg   – latest undistorted input frame   (JPEG)
  GET /render.jpg  – latest Gaussian-splat render     (JPEG)
  GET /state.json  – trajectory + stats + lidar version (1 s)

Binary format for lidar.bin  (little-endian):
  uint32   N          – number of points
  float32  xyz[N*3]   – world-frame positions (x, y, z)
  uint8    rgb[N*3]   – colours (r, g, b)
"""

import json
import os
import struct

_STATIC_DIR = os.path.join(
    os.path.dirname(__file__),
    "../web/static",
)
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Empty

# ---------------------------------------------------------------------------
# Dashboard HTML
# Coordinate mapping  ROS → Three.js:  (x, y, z) → (x, z, -y)
# ---------------------------------------------------------------------------
HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
  <title>Gaussian-LIC Monitor</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{background:#0a0a0a;color:#e0e0e0;font-family:monospace;
         height:100dvh;display:flex;flex-direction:column;overflow:hidden}
    #statusbar{background:#141414;padding:7px 12px;font-size:13px;
               display:flex;justify-content:space-between;align-items:center;
               border-bottom:1px solid #2a2a2a;flex-shrink:0;gap:8px}
    .dot{display:inline-block;width:8px;height:8px;border-radius:50%;
         background:#4caf50;margin-right:6px;flex-shrink:0}
    .dot.dead{background:#f44336}
    #main{display:flex;flex:1;overflow:hidden;min-height:0}
    #view-wrap{flex:1;position:relative;min-width:0}
    #view-wrap canvas{touch-action:none;display:block}
    #hint{position:absolute;bottom:8px;right:8px;color:#333;font-size:10px;
          pointer-events:none;text-align:right}
    #sidebar{width:200px;display:flex;flex-direction:column;gap:6px;padding:8px;
             background:#111;border-left:1px solid #222;flex-shrink:0;overflow-y:auto}
    .img-wrap{position:relative;width:100%}
    .img-wrap img{width:100%;border-radius:4px;background:#1a1a1a;
                  display:block;min-height:60px;object-fit:cover}
    .img-label{position:absolute;bottom:4px;left:4px;
               background:rgba(0,0,0,.65);color:#aaa;font-size:9px;
               text-transform:uppercase;letter-spacing:.08em;
               padding:1px 5px;border-radius:3px;pointer-events:none}
    .img-label.live{color:#4caf50}
    .card{background:#1a1a1a;border-radius:4px;padding:8px 10px;
          font-size:12px;line-height:2}
    .lbl{color:#777;display:block;font-size:10px;text-transform:uppercase;
         letter-spacing:.05em}
    .val{color:#4caf50;font-size:16px;font-weight:bold}
    @media(max-width:480px){#sidebar{width:140px}.val{font-size:13px}}
    #rec-btn{background:#c62828;color:#fff;border:none;border-radius:4px;
             padding:5px 14px;font-family:monospace;font-size:12px;cursor:pointer;
             letter-spacing:.05em;animation:recpulse 1.4s ease-in-out infinite;flex-shrink:0}
    #rec-btn:disabled{background:#2e7d32;animation:none;cursor:default}
    @keyframes recpulse{0%,100%{opacity:1}50%{opacity:.55}}
    .hw-row{display:flex;justify-content:space-between;font-size:11px;line-height:1.9;color:#aaa}
    .hw-row span:last-child{color:#e0e0e0;font-weight:bold}
    .hw-bar{height:3px;background:#1e1e1e;border-radius:2px;margin-bottom:5px}
    .hw-fill{height:100%;border-radius:2px;width:0%;transition:width .6s,background .6s}
    .hw-temps{font-size:10px;color:#666;margin-top:2px;line-height:1.7}
  </style>
  <script type="importmap">
    {"imports":{
      "three":"/three.module.min.js"
    }}
  </script>
</head>
<body>
<div id="statusbar">
  <span><span id="dot" class="dot dead"></span><span id="stxt">Connecting…</span></span>
  <button id="rec-btn" onclick="startRecording()">&#9679; Start Recording</button>
  <button id="stop-btn" onclick="stopRecording()" disabled style="background:#555;color:#fff;border:none;border-radius:4px;padding:5px 14px;font-family:monospace;font-size:12px;cursor:default;flex-shrink:0">&#9632; Stop &amp; Save</button>
  <button id="discard-btn" onclick="discardSession()" disabled style="background:#555;color:#fff;border:none;border-radius:4px;padding:5px 14px;font-family:monospace;font-size:12px;cursor:default;flex-shrink:0">&#10005; Discard</button>
  <button id="new-session-btn" onclick="newSession()" style="display:none;background:#1565c0;color:#fff;border:none;border-radius:4px;padding:5px 14px;font-family:monospace;font-size:12px;cursor:pointer;flex-shrink:0">&#8635; New Session</button>
  <span id="elapsed" style="color:#666">--:--</span>
</div>
<div id="main">
  <div id="view-wrap">
    <canvas id="glcanvas"></canvas>
    <div id="hint">drag · scroll · two-finger</div>
  </div>
  <div id="sidebar">
    <div class="img-wrap">
      <img id="input-img" alt="Input frame">
      <span class="img-label live">▶ Live sensor</span>
    </div>
    <div class="img-wrap">
      <img id="render-img" alt="Rendered view">
      <span class="img-label">⬡ Reconstruction</span>
    </div>
    <div class="card">
      <span class="lbl">Odometry msgs</span>
      <span class="val" id="kf">—</span>
      <span class="lbl">Gaussians</span>
      <span class="val" id="gs">—</span>
    </div>
    <div class="card">
      <span class="lbl">Hardware</span>
      <div class="hw-row"><span>CPU</span><span id="hw-cpu-val">—</span></div>
      <div class="hw-bar"><div class="hw-fill" id="hw-cpu-fill"></div></div>
      <div class="hw-row"><span>RAM</span><span id="hw-ram-val">—</span></div>
      <div class="hw-bar"><div class="hw-fill" id="hw-ram-fill"></div></div>
      <div class="hw-row"><span>GPU</span><span id="hw-gpu-val">—</span></div>
      <div class="hw-bar"><div class="hw-fill" id="hw-gpu-fill"></div></div>
      <div class="hw-row" style="margin-top:2px"><span style="color:#555">GPU MHz</span><span id="hw-gpu-mhz" style="color:#666">—</span></div>
      <div class="hw-temps" id="hw-temps">—</div>
    </div>
  </div>
</div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from '/OrbitControls.module.js';

// ── Scene ──────────────────────────────────────────────────────────────────
const wrap     = document.getElementById('view-wrap');
const canvas   = document.getElementById('glcanvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);

const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 2000);
camera.position.set(0, 20, 30);
camera.lookAt(0, 0, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.screenSpacePanning = false;

const grid = new THREE.GridHelper(200, 100, 0x1a1a1a, 0x151515);
scene.add(grid);

// ── Gaussian point cloud ───────────────────────────────────────────────────
let cloudMesh    = null;
let cloudVersion = -1;

function loadCloud(serverVersion) {
  if (serverVersion <= cloudVersion) return;
  fetch('/cloud.bin')
    .then(r => { if (!r.ok) throw r; return r.arrayBuffer(); })
    .then(buf => {
      const n   = new DataView(buf).getUint32(0, true);
      if (n === 0) return;
      const xyz = new Float32Array(buf, 4, n * 3);
      const rgb = new Uint8Array(buf, 4 + n * 12, n * 3);
      const col = new Float32Array(n * 3);
      for (let i = 0; i < n * 3; i++) col[i] = rgb[i] / 255;
      const pos = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        pos[i*3]   =  xyz[i*3];
        pos[i*3+1] =  xyz[i*3+2];
        pos[i*3+2] = -xyz[i*3+1];
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('color',    new THREE.BufferAttribute(col, 3));
      const mat = new THREE.PointsMaterial({
        size: 0.05, vertexColors: true, sizeAttenuation: true
      });
      if (cloudMesh) scene.remove(cloudMesh);
      cloudMesh = new THREE.Points(geo, mat);
      scene.add(cloudMesh);
      cloudVersion = serverVersion;
    })
    .catch(() => {});
}

// ── New-keyframe point highlight ──────────────────────────────────────────
let newPtsMesh    = null;
let newPtsVersion = -1;
let newPtsTime    = -Infinity;
const NEW_PTS_FADE = 4.0;

function loadNewPoints(serverVersion) {
  if (serverVersion <= newPtsVersion) return;
  fetch('/newpts.bin')
    .then(r => { if (!r.ok) throw r; return r.arrayBuffer(); })
    .then(buf => {
      const n = new DataView(buf).getUint32(0, true);
      if (n === 0) return;
      const xyzSrc = new Float32Array(buf, 4, n * 3);
      const pos = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        pos[i*3]   =  xyzSrc[i*3];
        pos[i*3+1] =  xyzSrc[i*3+2];
        pos[i*3+2] = -xyzSrc[i*3+1];
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      const mat = new THREE.PointsMaterial({
        color: 0xffeb3b, size: 0.12, sizeAttenuation: true,
        transparent: true, opacity: 1.0, depthWrite: false,
      });
      if (newPtsMesh) scene.remove(newPtsMesh);
      newPtsMesh    = new THREE.Points(geo, mat);
      newPtsVersion = serverVersion;
      newPtsTime    = performance.now() / 1000;
      scene.add(newPtsMesh);
    })
    .catch(() => {});
}

function tickNewPoints(now) {
  if (!newPtsMesh) return;
  const age = now - newPtsTime;
  if (age >= NEW_PTS_FADE) {
    scene.remove(newPtsMesh);
    newPtsMesh = null;
  } else {
    newPtsMesh.material.opacity = 1.0 - age / NEW_PTS_FADE;
  }
}

// ── Trajectory + current-pose marker ──────────────────────────────────────
let trajLine  = null;
let posMarker = null;
let cameraFollowing = true;

function updateTrajectory(traj) {
  if (traj.length < 1) return;

  const pts = traj.map(p => new THREE.Vector3(p[0], p[2] ?? 0, -p[1]));

  if (pts.length >= 2) {
    const n   = pts.length;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos[i*3]   = pts[i].x;
      pos[i*3+1] = pts[i].y;
      pos[i*3+2] = pts[i].z;
      const t = i / (n - 1);
      col[i*3]   = 0.1 + 0.9*t;
      col[i*3+1] = 0.8 - 0.5*t;
      col[i*3+2] = 1.0 - 0.8*t;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(col, 3));
    const mat = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });
    if (trajLine) scene.remove(trajLine);
    trajLine = new THREE.Line(geo, mat);
    scene.add(trajLine);
  }

  const last = pts[pts.length - 1];
  if (!posMarker) {
    posMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.25, 12, 8),
      new THREE.MeshBasicMaterial({ color: 0xff5252 })
    );
    scene.add(posMarker);
  }
  posMarker.position.copy(last);

  if (cameraFollowing && traj.length > 5) {
    const first = pts[0];
    camera.position.set(first.x, first.y + 20, first.z + 28);
    controls.target.copy(first);
    controls.update();
    cameraFollowing = false;
  }
}

// ── Resize ─────────────────────────────────────────────────────────────────
function resize() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// ── Animation loop ─────────────────────────────────────────────────────────
(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  tickNewPoints(performance.now() / 1000);
  renderer.render(scene, camera);
})();

// ── Hardware monitor ──────────────────────────────────────────────────────
function hwBar(prefix, pct, label) {
  const val  = document.getElementById('hw-' + prefix + '-val');
  const fill = document.getElementById('hw-' + prefix + '-fill');
  if (val)  val.textContent = label;
  if (fill) {
    fill.style.width      = Math.min(pct, 100) + '%';
    fill.style.background = pct > 90 ? '#f44336' : pct > 70 ? '#ff9800' : '#4caf50';
  }
}

// ── Recording buttons ──────────────────────────────────────────────────────
window.startRecording = function() {
  const btn = document.getElementById('rec-btn');
  btn.disabled = true;
  btn.textContent = '⟳ Starting…';
  fetch('/start', { method: 'POST' })
    .then(r => r.json())
    .then(d => {
      if (d.drivers === 'starting') {
        btn.textContent = '⟳ Starting drivers…';
      } else {
        btn.textContent = '● Recording…';
        const stop = document.getElementById('stop-btn');
        stop.disabled = false;
        stop.style.background = '#e65100';
        stop.style.cursor = 'pointer';
      }
    })
    .catch(() => {
      btn.disabled = false;
      btn.textContent = '⬤ Start Recording';
    });
};

window.stopRecording = function() {
  const btn = document.getElementById('stop-btn');
  btn.disabled = true;
  btn.textContent = '■ Saving…';
  btn.style.background = '#555';
  btn.style.cursor = 'default';
  document.getElementById('discard-btn').disabled = true;
  fetch('/stop', { method: 'POST' }).catch(() => {});
};

window.discardSession = function() {
  const btn = document.getElementById('discard-btn');
  btn.disabled = true;
  btn.textContent = '✕ Discarding…';
  document.getElementById('stop-btn').disabled = true;
  fetch('/discard', { method: 'POST' })
    .then(() => window.location.reload())
    .catch(() => {
      btn.disabled = false;
      btn.textContent = '✕ Discard';
    });
};

window.newSession = function() {
  const btn = document.getElementById('new-session-btn');
  btn.disabled = true;
  btn.textContent = '⟳ Starting…';
  fetch('/new_session', { method: 'POST' })
    .then(r => r.json())
    .then(() => {
      btn.style.display = 'none';
      btn.disabled = false;
      btn.textContent = '↻ New Session';
      const rec  = document.getElementById('rec-btn');
      const stop = document.getElementById('stop-btn');
      rec.disabled = false;
      rec.textContent = '⬤ Start Recording';
      stop.disabled = true;
      stop.textContent = '■ Stop & Save';
      stop.style.background = '#555';
      stop.style.cursor = 'default';
      startTime = null;
    })
    .catch(() => {
      btn.disabled = false;
      btn.textContent = '↻ New Session';
    });
};

// ── Data polling ───────────────────────────────────────────────────────────
let startTime = null;

function fetchState() {
  fetch('/state.json')
    .then(r => r.json())
    .then(d => {
      updateTrajectory(d.trajectory);
      loadCloud(d.stats.cloud_version || 0);
      loadNewPoints(d.stats.newpts_version || 0);

      document.getElementById('kf').textContent = d.stats.keyframes || '—';
      const gs = document.getElementById('gs');
      if (gs) gs.textContent = d.stats.gaussians > 0
        ? (d.stats.gaussians / 1e6).toFixed(2) + 'M' : '—';

      document.getElementById('dot').className = 'dot';
      document.getElementById('stxt').textContent = 'LIVE';

      if (d.hw) {
        const hw = d.hw;
        if (hw.cpu_pct  != null) hwBar('cpu', hw.cpu_pct,  hw.cpu_pct.toFixed(0)  + '%');
        if (hw.gpu_pct  != null) hwBar('gpu', hw.gpu_pct,  hw.gpu_pct.toFixed(0)  + '%');
        if (hw.ram_pct  != null) hwBar('ram', hw.ram_pct,
          (hw.ram_used_mb/1024).toFixed(1) + ' / ' + (hw.ram_total_mb/1024).toFixed(1) + ' GB');
        const mhzEl = document.getElementById('hw-gpu-mhz');
        if (mhzEl && hw.gpu_mhz != null)
          mhzEl.textContent = hw.gpu_mhz + ' / ' + (hw.gpu_max_mhz ?? '?') + ' MHz';
        const te = document.getElementById('hw-temps');
        if (te && hw.temps && Object.keys(hw.temps).length)
          te.textContent = Object.entries(hw.temps).map(([k,v])=>k+': '+v+'°C').join('  ');
      }

      const btn  = document.getElementById('rec-btn');
      const stop = document.getElementById('stop-btn');
      const discard = document.getElementById('discard-btn');
      if (d.recording_started && !d.stop_requested && !d.session_done) {
        btn.disabled = true;
        btn.textContent = '● Recording…';
        if (stop.disabled) {
          stop.disabled = false;
          stop.style.background = '#e65100';
          stop.style.cursor = 'pointer';
          stop.textContent = '■ Stop & Save';
          discard.disabled = false;
          discard.style.background = '#6d1f1f';
          discard.style.cursor = 'pointer';
        }
      }
      if (d.stop_requested && !d.session_done) {
        stop.disabled = true;
        stop.textContent = '■ Saving…';
        stop.style.background = '#555';
        stop.style.cursor = 'default';
      }
      if (d.session_done) {
        stop.textContent = '✓ Saved';
        discard.style.display = 'none';
        const nb = document.getElementById('new-session-btn');
        if (nb) nb.style.display = 'inline-block';
      }

      if (!startTime) startTime = Date.now();
      const s = Math.floor((Date.now() - startTime) / 1000);
      document.getElementById('elapsed').textContent =
        String(Math.floor(s / 60)).padStart(2, '0') + ':' +
        String(s % 60).padStart(2, '0');
    })
    .catch(() => {
      document.getElementById('dot').className = 'dot dead';
      document.getElementById('stxt').textContent = 'NO DATA';
    });
}

function fetchImg(id, path) {
  const url = path + '?t=' + Date.now();
  const tmp = new Image();
  tmp.onload = () => { document.getElementById(id).src = url; };
  tmp.src = url;
}

fetchState();
fetchImg('input-img',  '/input.jpg');
fetchImg('render-img', '/render.jpg');
setInterval(fetchState, 200);
setInterval(() => fetchImg('input-img',  '/input.jpg'),  200);
setInterval(() => fetchImg('render-img', '/render.jpg'), 500);
</script>
</body>
</html>
"""

class WebMonitorNode(Node):
    def __init__(self):
        super().__init__("web_monitor")

        self._bridge = CvBridge()
        self._lock = threading.Lock()

        self._trajectory: list = []
        self._odom_count: int = 0
        self._recording_started: bool = False
        self._stop_requested: bool = False
        self._session_done: bool = False
        self._hw: dict = {}
        self._drivers_state: str = 'idle'
        self._driver_proc = None
        self._we_started_drivers: bool = False
        self._gs_proc = None
        self._latest_jpeg: bytes = b""
        self._input_jpeg: bytes = b""

        # Gaussian point cloud
        self._gaussian_count: int = 0
        self._cloud_bin: bytes = b""
        self._cloud_version: int = 0
        self._newpts_bin: bytes = b""
        self._newpts_version: int = 0

        self._config_path = self.declare_parameter(
            "config_path", "").get_parameter_value().string_value
        self._lpips_path = self.declare_parameter(
            "lpips_path", "").get_parameter_value().string_value

        be_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5,
        )

        self.create_subscription(Odometry,    "/odin1/odometry",            self._odom_cb,      be_qos)
        self.create_subscription(PointCloud2, "/gaussian_lic/gaussians",   self._gaussians_cb,  1)
        self.create_subscription(PointCloud2, "/gaussian_lic/new_points",  self._new_points_cb, 1)
        self.create_subscription(Image,       "/gaussian_lic/input",       self._input_cb,      1)
        self.create_subscription(Image,       "/gaussian_lic/render",      self._render_cb,     1)

        self._start_pub = self.create_publisher(Empty, "/gaussian_lic/start_recording", 1)
        self._stop_pub  = self.create_publisher(Empty, "/gaussian_lic/stop_recording",  1)

        threading.Thread(target=self._hw_worker,      daemon=True).start()
        threading.Thread(target=self._session_watcher, daemon=True).start()

        port = self.declare_parameter("port", 8765).get_parameter_value().integer_value
        server = ThreadingHTTPServer(("0.0.0.0", port), self._make_handler())
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.get_logger().info(f"Web monitor → http://0.0.0.0:{port}")

    # ------------------------------------------------------------------
    # ROS2 callbacks
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        with self._lock:
            self._odom_count += 1
            if self._odom_count % 5 == 1:
                self._trajectory.append([p.x, p.y, p.z])

    def _img_to_jpeg(self, msg: Image, quality: int = 82) -> bytes:
        bgr = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return bytes(buf) if ok else b""

    def _input_cb(self, msg: Image) -> None:
        try:
            jpeg = self._img_to_jpeg(msg, quality=80)
            if jpeg:
                with self._lock:
                    self._input_jpeg = jpeg
        except Exception as exc:
            self.get_logger().warn(f"input decode: {exc}", throttle_duration_sec=5.0)

    def _render_cb(self, msg: Image) -> None:
        try:
            jpeg = self._img_to_jpeg(msg, quality=82)
            if jpeg:
                with self._lock:
                    self._latest_jpeg = jpeg
        except Exception as exc:
            self.get_logger().warn(f"render decode: {exc}", throttle_duration_sec=5.0)

    def _gaussians_cb(self, msg: PointCloud2) -> None:
        N = msg.width
        if N == 0:
            return
        try:
            raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(N, 16)
            xyz = raw[:, :12].view(np.float32).reshape(N, 3).copy()
            rgb_u32 = raw[:, 12:16].view(np.uint32).reshape(N).copy()
            r = ((rgb_u32 >> 16) & 0xFF).astype(np.uint8)
            g = ((rgb_u32 >>  8) & 0xFF).astype(np.uint8)
            b = ( rgb_u32        & 0xFF).astype(np.uint8)
            step  = max(1, N // 40000)
            idx   = np.arange(0, N, step)
            xyz_d = xyz[idx].astype(np.float32)
            rgb_d = np.column_stack([r[idx], g[idx], b[idx]])
            n_d   = len(idx)
            cloud_bin = (
                struct.pack("<I", n_d)
                + xyz_d.tobytes()
                + rgb_d.tobytes()
            )
            with self._lock:
                self._cloud_bin      = cloud_bin
                self._gaussian_count = N
                self._cloud_version += 1
        except Exception as exc:
            self.get_logger().warn(f"cloud parse: {exc}", throttle_duration_sec=5.0)

    def _new_points_cb(self, msg: PointCloud2) -> None:
        N = msg.width
        if N == 0:
            return
        try:
            raw   = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(N, 16)
            xyz   = raw[:, :12].view(np.float32).reshape(N, 3).copy()
            rgb_u32 = raw[:, 12:16].view(np.uint32).reshape(N).copy()
            r = ((rgb_u32 >> 16) & 0xFF).astype(np.uint8)
            g = ((rgb_u32 >>  8) & 0xFF).astype(np.uint8)
            b = ( rgb_u32        & 0xFF).astype(np.uint8)
            rgb_d = np.column_stack([r, g, b])
            newpts_bin = (
                struct.pack("<I", N)
                + xyz.astype(np.float32).tobytes()
                + rgb_d.tobytes()
            )
            with self._lock:
                self._newpts_bin     = newpts_bin
                self._newpts_version += 1
        except Exception as exc:
            self.get_logger().warn(f"new_points parse: {exc}", throttle_duration_sec=5.0)

    # ------------------------------------------------------------------
    # Hardware monitor
    # ------------------------------------------------------------------

    def _hw_worker(self) -> None:
        prev_total = prev_idle = 0
        while True:
            time.sleep(1.5)
            hw: dict = {}

            try:
                with open('/proc/stat') as f:
                    cpu = f.readline().split()
                idle  = int(cpu[4]) + int(cpu[5])
                total = sum(int(x) for x in cpu[1:])
                if prev_total > 0:
                    d_total = total - prev_total
                    d_idle  = idle  - prev_idle
                    hw['cpu_pct'] = round(100.0 * (1.0 - d_idle / d_total), 1) if d_total else 0.0
                prev_total, prev_idle = total, idle
            except Exception:
                pass

            try:
                mem: dict = {}
                with open('/proc/meminfo') as f:
                    for line in f:
                        parts = line.split()
                        mem[parts[0].rstrip(':')] = int(parts[1])
                used_mb  = (mem['MemTotal'] - mem['MemAvailable']) // 1024
                total_mb = mem['MemTotal'] // 1024
                hw['ram_used_mb']  = used_mb
                hw['ram_total_mb'] = total_mb
                hw['ram_pct']      = round(100.0 * used_mb / total_mb, 1) if total_mb else 0.0
            except Exception:
                pass

            _gpu_devfreq = '/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu'
            for _gpu_load_path in ('/sys/devices/platform/bus@0/17000000.gpu/load',
                                   '/sys/devices/gpu.0/load'):
                try:
                    with open(_gpu_load_path) as f:
                        hw['gpu_pct'] = round(int(f.read().strip()) / 10.0, 1)
                    break
                except Exception:
                    pass
            try:
                with open(f'{_gpu_devfreq}/cur_freq') as f:
                    cur_hz = int(f.read().strip())
                with open(f'{_gpu_devfreq}/max_freq') as f:
                    max_hz = int(f.read().strip())
                hw['gpu_mhz']     = cur_hz // 1_000_000
                hw['gpu_max_mhz'] = max_hz // 1_000_000
            except Exception:
                pass

            temps: dict = {}
            try:
                base = '/sys/class/thermal'
                for zone in sorted(os.listdir(base)):
                    if not zone.startswith('thermal_zone'):
                        continue
                    try:
                        with open(f'{base}/{zone}/type') as f:
                            name = f.read().strip()
                        with open(f'{base}/{zone}/temp') as f:
                            temp_c = round(int(f.read().strip()) / 1000.0, 1)
                        if any(k in name for k in ('CPU', 'GPU', 'cpu', 'gpu')):
                            temps[name] = temp_c
                    except Exception:
                        pass
            except Exception:
                pass
            hw['temps'] = temps

            with self._lock:
                self._hw = hw

    # ------------------------------------------------------------------
    # Driver / session management
    # ------------------------------------------------------------------

    def _gs_mapping_running(self) -> bool:
        return self.count_subscribers('/gaussian_lic/stop_recording') > 0

    def _session_watcher(self) -> None:
        while True:
            time.sleep(1.0)
            with self._lock:
                if not self._stop_requested or self._session_done:
                    continue
            for _ in range(120):
                time.sleep(1.0)
                if not self._gs_mapping_running():
                    break
            with self._lock:
                self._session_done = True
            self.get_logger().info('Session saved — ready for new session.')

    def _launch_gs_mapping(self, result_path: str) -> None:
        import ament_index_python.packages as _ament
        pkg_share = _ament.get_package_share_directory('gaussian_lic')
        config  = self._config_path  or os.path.join(pkg_share, 'config', 'odin1.yaml')
        lpips   = self._lpips_path   or os.path.join(pkg_share, 'src', 'lpips')
        torch_lib = '/home/jetson/.local/lib/python3.10/site-packages/torch/lib'
        cmd = (
            'source /opt/ros/humble/setup.bash && '
            'source /home/jetson/ws/install/setup.bash && '
            f'export LD_LIBRARY_PATH={torch_lib}:${{LD_LIBRARY_PATH}} && '
            f'ros2 run gaussian_lic gs_mapping '
            f'--ros-args '
            f'-p config_path:={config} '
            f'-p result_path:={result_path} '
            f'-p lpips_path:={lpips} '
            f'-p wait_for_start:=true'
        )
        proc = subprocess.Popen(['bash', '-c', cmd])
        with self._lock:
            self._gs_proc = proc
        self.get_logger().info(f'gs_mapping launching → {result_path}')

    def _odin_drivers_running(self) -> bool:
        return self.count_publishers('/odin1/odometry') > 0

    def _launch_drivers(self) -> None:
        with self._lock:
            self._drivers_state = 'starting'

        def _worker():
            try:
                env = os.environ.copy()
                cmd = (
                    'source /opt/ros/humble/setup.bash && '
                    'source /home/jetson/catkin_ws/install/setup.bash && '
                    'source /home/jetson/ws/install/setup.bash && '
                    'ros2 launch gaussian_lic odin1_drivers.launch.py'
                )
                proc = subprocess.Popen(['bash', '-c', cmd], env=env)
                with self._lock:
                    self._driver_proc = proc
                    self._we_started_drivers = True

                self.get_logger().info('Odin1 drivers launching…')

                for _ in range(60):
                    time.sleep(0.5)
                    if self._odin_drivers_running():
                        break
                else:
                    self.get_logger().error('Odin1 drivers did not publish within 30 s.')
                    with self._lock:
                        self._drivers_state = 'idle'
                    return

                with self._lock:
                    self._drivers_state = 'running'
                    self._recording_started = True
                self._start_pub.publish(Empty())
                self.get_logger().info('Odin1 drivers up — recording started.')
            except Exception as e:
                self.get_logger().error(f'Driver launch failed: {e}')
                with self._lock:
                    self._drivers_state = 'idle'

        threading.Thread(target=_worker, daemon=True).start()

    def _reset_session_state(self):
        """Reset all per-session state. Must be called with _lock held."""
        self._trajectory.clear()
        self._odom_count          = 0
        self._gaussian_count      = 0
        self._cloud_bin           = b""
        self._cloud_version       = 0
        self._newpts_bin          = b""
        self._newpts_version      = 0
        self._latest_jpeg         = b""
        self._input_jpeg          = b""
        self._recording_started   = False
        self._stop_requested      = False
        self._session_done        = False
        self._we_started_drivers  = False
        self._driver_proc         = None
        self._gs_proc             = None
        self._drivers_state       = 'idle'

    # ------------------------------------------------------------------
    # HTTP handler
    # ------------------------------------------------------------------

    def _make_handler(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                p = self.path.split("?")[0]
                if p == "/":
                    self._send(HTML.encode(), "text/html; charset=utf-8")
                elif p in ("/three.module.min.js", "/OrbitControls.module.js"):
                    fname = os.path.join(_STATIC_DIR, p.lstrip("/"))
                    with open(fname, "rb") as f:
                        self._send(f.read(), "text/javascript; charset=utf-8")
                elif p == "/input.jpg":
                    with node._lock: data = node._input_jpeg
                    self._send_or_204(data, "image/jpeg")
                elif p == "/render.jpg":
                    with node._lock: data = node._latest_jpeg
                    self._send_or_204(data, "image/jpeg")
                elif p == "/cloud.bin":
                    with node._lock: data = node._cloud_bin
                    self._send_or_204(data, "application/octet-stream")
                elif p == "/newpts.bin":
                    with node._lock: data = node._newpts_bin
                    self._send_or_204(data, "application/octet-stream")
                elif p == "/state.json":
                    with node._lock:
                        traj    = node._trajectory[-2000:]
                        stats   = {
                            "keyframes":      node._odom_count,
                            "gaussians":      node._gaussian_count,
                            "cloud_version":  node._cloud_version,
                            "newpts_version": node._newpts_version,
                        }
                        rec     = node._recording_started
                        stop    = node._stop_requested
                        done    = node._session_done
                        hw      = node._hw.copy()
                        drivers = node._drivers_state
                    self._send(
                        json.dumps({"trajectory": traj, "stats": stats,
                                    "recording_started": rec, "stop_requested": stop,
                                    "session_done": done,
                                    "hw": hw,
                                    "drivers_state": drivers}).encode(),
                        "application/json",
                        no_cache=True,
                    )
                else:
                    self.send_response(404); self.end_headers()

            def do_POST(self):
                p = self.path.split("?")[0]
                if p == "/start":
                    if node._odin_drivers_running():
                        with node._lock:
                            node._recording_started = True
                            node._drivers_state = 'external'
                        node._start_pub.publish(Empty())
                        node.get_logger().info("Start Recording triggered (driver already running).")
                        self._send(b'{"ok":true,"drivers":"external"}', "application/json")
                    else:
                        node._launch_drivers()
                        self._send(b'{"ok":true,"drivers":"starting"}', "application/json")
                elif p == "/stop":
                    driver_proc = None
                    with node._lock:
                        node._stop_requested = True
                        if node._we_started_drivers:
                            driver_proc = node._driver_proc
                    node._stop_pub.publish(Empty())
                    if driver_proc:
                        try:
                            driver_proc.terminate()
                        except Exception:
                            pass
                    node.get_logger().info("Stop & Save triggered via web dashboard.")
                    self._send(b'{"ok":true}', "application/json")
                elif p == "/discard":
                    with node._lock:
                        gs_proc = node._gs_proc
                    if gs_proc:
                        try:
                            gs_proc.kill()
                        except Exception:
                            pass
                    subprocess.run(['pkill', '-KILL', '-f', 'gs_mapping'], timeout=5)
                    time.sleep(1.5)
                    from datetime import datetime as _dt
                    result_path = f'/home/jetson/results/{_dt.now().strftime("%Y-%m-%d_%H-%M-%S")}'
                    with node._lock:
                        node._reset_session_state()
                    node._launch_gs_mapping(result_path)
                    node.get_logger().info(f'Session discarded — new session → {result_path}')
                    self._send(b'{"ok":true}', "application/json")
                elif p == "/new_session":
                    from datetime import datetime
                    result_path = f'/home/jetson/results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
                    with node._lock:
                        node._reset_session_state()
                    node._launch_gs_mapping(result_path)
                    if not node._odin_drivers_running():
                        node._launch_drivers()
                    node.get_logger().info(f'New session started → {result_path}')
                    self._send(b'{"ok":true}', "application/json")
                else:
                    self.send_response(404); self.end_headers()

            def _send(self, body: bytes, ct: str, no_cache=False):
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(body)))
                if no_cache:
                    self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except BrokenPipeError:
                    pass

            def _send_or_204(self, body: bytes, ct: str):
                if body:
                    self._send(body, ct, no_cache=True)
                else:
                    self.send_response(204); self.end_headers()

            def log_message(self, fmt, *args):
                pass

        return Handler


def main():
    rclpy.init()
    node = WebMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
