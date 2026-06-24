#!/usr/bin/env python3
"""
Gaussian-LIC parameter sweep.

Runs gs_mapping with different parameter combinations over a ROS2 bag,
saving all results + statistics to timestamped sub-folders.

Usage (source your workspace first):
    python3 scripts/sweep.py /path/to/bag_dir
    python3 scripts/sweep.py /path/to/bag_dir --rate 0.5   # slow playback

Results layout:
    $RESULTS_ROOT/sweep_<timestamp>/
        1/   <- point_cloud.ply, render/, gt/, stats.txt
        2/
        ...
        summary.txt

Edit the SWEEP list below to define the parameter combinations.
Every key in a dict overrides the corresponding key in the base odin1.yaml;
all other keys stay at their defaults.

Override the output root via the RESULTS_ROOT environment variable.
On Jetson Orin the default is /home/jetson/results; set RESULTS_ROOT to
override for a different layout.
"""

import os
import re
import sys
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("Install pyyaml:  pip3 install pyyaml")

# ── Parameter combinations to sweep ──────────────────────────────────────────
# Each dict is one run. Unlisted keys keep their base-config values.
SWEEP = [
    # baseline (real-time settings tuned for Jetson Orin)
    dict(max_iters=15,  optimize_depth=False, prune_opacity_thresh=0.01,  prune_every_n_keyframes=5),
    # more iterations, same pruning
    dict(max_iters=30,  optimize_depth=False, prune_opacity_thresh=0.01,  prune_every_n_keyframes=5),
    # depth loss enabled
    dict(max_iters=15,  optimize_depth=True,  prune_opacity_thresh=0.01,  prune_every_n_keyframes=5),
    # depth loss + more iterations
    dict(max_iters=30,  optimize_depth=True,  prune_opacity_thresh=0.01,  prune_every_n_keyframes=5),
    # high quality (offline — slower, not suitable for live use on Orin NX 8 GB)
    dict(max_iters=100, optimize_depth=True,  prune_opacity_thresh=0.005, prune_every_n_keyframes=10),
]

RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", "/home/jetson/results"))

# Seconds to wait between launching gs_mapping and starting bag play.
# TRT engine loading on Jetson can take 10–20 s on first run; increase if needed.
INIT_WAIT_S = 20

# Bag playback rate (1.0 = real time). Use 0.5 on Orin NX 8 GB to avoid dropped frames.
BAG_RATE = "1.0"

# ─────────────────────────────────────────────────────────────────────────────

ANSI = re.compile(r'\x1B\[[0-9;]*[A-Za-z]|\x1B\([A-Za-z]')


def find_pkg_share() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("gaussian_lic"))
    except Exception:
        sys.exit(
            "Cannot find gaussian_lic package.\n"
            "Source your workspace:  source install/setup.bash"
        )


def make_config(base_path: Path, overrides: dict, out_path: Path) -> None:
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides)
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def stream_to(proc, lines: list, ready_event: threading.Event) -> None:
    """Read proc stdout line-by-line, print to terminal, append to list.
    Sets ready_event when the node signals it's subscribed and publishing."""
    for raw in proc.stdout:
        print(raw, end="", flush=True)
        lines.append(raw)
        if "Publishing:" in raw and not ready_event.is_set():
            ready_event.set()
    ready_event.set()


def run_single(
    bag_path: Path,
    config_path: Path,
    result_path: Path,
    lpips_path: Path,
) -> tuple[int, str]:
    """
    Start gs_mapping, wait for it to be ready, start bag play,
    then wait for gs_mapping to finish (it calls rclcpp::shutdown()
    automatically after evaluating quality).
    Returns (returncode, captured_stdout).
    """
    result_path.mkdir(parents=True, exist_ok=True)

    gs_cmd = [
        "ros2", "run", "gaussian_lic", "gs_mapping",
        "--ros-args",
        "-p", f"config_path:={config_path}",
        "-p", f"result_path:={result_path}",
        "-p", f"lpips_path:={lpips_path}",
    ]
    bag_cmd = [
        "ros2", "bag", "play", str(bag_path),
        "--clock", "-r", BAG_RATE,
    ]

    gs_proc = subprocess.Popen(
        gs_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    lines: list[str] = []
    ready_evt = threading.Event()
    reader = threading.Thread(
        target=stream_to, args=(gs_proc, lines, ready_evt), daemon=True
    )
    reader.start()

    print(f"\n[sweep] Waiting for gs_mapping to initialise (up to {INIT_WAIT_S}s)…",
          flush=True)
    ready_evt.wait(timeout=INIT_WAIT_S)

    if gs_proc.poll() is not None:
        reader.join()
        return gs_proc.returncode, "".join(lines)

    print(f"\n[sweep] Starting bag: {bag_path}\n", flush=True)
    bag_proc = subprocess.Popen(bag_cmd)

    gs_proc.wait()
    reader.join()

    if bag_proc.poll() is None:
        bag_proc.send_signal(signal.SIGINT)
        try:
            bag_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            bag_proc.kill()

    return gs_proc.returncode, "".join(lines)


def extract_stats(output: str) -> str:
    clean = ANSI.sub("", output)
    idx = clean.find("Runtime Statistics")
    return clean[idx:] if idx != -1 else "(stats block not found in output)"


def fmt_params(params: dict) -> str:
    return "  " + "\n  ".join(f"{k}: {v}" for k, v in params.items())


# ─────────────────────────────────────────────────────────────────────────────

import subprocess  # noqa: E402 — placed here so stream_to can reference it at module level


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} <bag_path>  [--rate <r>]")

    bag_path = Path(sys.argv[1])
    if not bag_path.exists():
        sys.exit(f"Bag not found: {bag_path}")

    global BAG_RATE
    if "--rate" in sys.argv:
        BAG_RATE = sys.argv[sys.argv.index("--rate") + 1]

    pkg_share  = find_pkg_share()
    base_cfg   = pkg_share / "config" / "odin1.yaml"
    lpips_path = pkg_share / "src" / "lpips"

    ts         = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_root = RESULTS_ROOT / f"sweep_{ts}"
    sweep_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Gaussian-LIC Parameter Sweep")
    print(f"  Bag:     {bag_path}")
    print(f"  Results: {sweep_root}")
    print(f"  Runs:    {len(SWEEP)}")
    print(f"{'='*60}\n")

    summary_rows: list[str] = []

    for i, params in enumerate(SWEEP, start=1):
        run_dir  = sweep_root / str(i)
        tmp_cfg  = sweep_root / f"_config_{i}.yaml"

        print(f"\n{'='*60}")
        print(f"  Run {i}/{len(SWEEP)}")
        print(fmt_params(params))
        print(f"  → {run_dir}")
        print(f"{'='*60}\n")

        make_config(base_cfg, params, tmp_cfg)

        t0 = time.time()
        rc, output = run_single(bag_path, tmp_cfg, run_dir, lpips_path)
        wall = time.time() - t0

        tmp_cfg.unlink(missing_ok=True)

        stats_block = extract_stats(output)

        stats_path = run_dir / "stats.txt"
        with open(stats_path, "w") as f:
            f.write(f"Run {i} / {len(SWEEP)}\n")
            f.write(f"{'='*60}\n\n")
            f.write("Parameters\n----------\n")
            f.write(fmt_params(params) + "\n")
            f.write(f"\nBag:       {bag_path}\n")
            f.write(f"Wall time: {wall:.1f} s\n")
            f.write(f"Exit code: {rc}\n\n")
            f.write("Runtime + Visual Quality Statistics\n")
            f.write("-----------------------------------\n")
            f.write(stats_block)

        row = (
            f"  {i:2d}  wall={wall:6.0f}s  rc={rc}  "
            + "  ".join(f"{k}={v}" for k, v in params.items())
        )
        summary_rows.append(row)
        print(f"\n[sweep] Run {i} done in {wall:.0f}s — {stats_path}\n")

    summary_path = sweep_root / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Sweep {ts}\n")
        f.write(f"Bag: {bag_path}\n")
        f.write(f"{'='*60}\n\n")
        f.write("\n".join(summary_rows) + "\n")

    print(f"\n{'='*60}")
    print(f"  All {len(SWEEP)} runs complete.")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
