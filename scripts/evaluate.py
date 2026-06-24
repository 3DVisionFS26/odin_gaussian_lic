#!/usr/bin/env python3
"""Offline evaluation of GaussianLIC render quality.

Compares every matched pair of ground-truth and rendered images in a result
directory and reports PSNR, SSIM, LPIPS, and MAE — per frame and as means.

Usage (inside or outside the container, requires torch + cv2):
    python3 evaluate.py <result_dir> [--lpips-model <path>]

    <result_dir> must contain:
        gt/        ground-truth frames   (test_NNNN.jpg)
        render/    rendered frames       (test_NNNN.jpg)

    --lpips-model  path to lpips_alex.pt
                   (default: <result_dir>/../../src/lpips/lpips_alex.pt,
                    then <script_dir>/../src/lpips/lpips_alex.pt)

Output:
    Prints a per-frame table and aggregate means.
    Writes  <result_dir>/metrics.csv   — per-frame numbers
    Writes  <result_dir>/metrics.json  — summary means
    Writes  <result_dir>/metrics.png   — per-frame plots for all four metrics
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Metrics — all operate on float32 tensors in [0, 1] of shape (C, H, W)
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt).item()
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> float:
    """SSIM matching the C++ loss_utils implementation (window-based, C1/C2 standard)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    p = pred.unsqueeze(0)   # (1, C, H, W)
    g = gt.unsqueeze(0)

    channels = p.size(1)
    half = window_size // 2

    # Build 2-D Gaussian kernel
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - half
    g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g1d /= g1d.sum()
    kernel = g1d.unsqueeze(1) @ g1d.unsqueeze(0)  # (ws, ws)
    kernel = kernel.unsqueeze(0).unsqueeze(0)       # (1, 1, ws, ws)
    kernel = kernel.expand(channels, 1, window_size, window_size).to(pred.device)

    opts = dict(padding=half, groups=channels)
    mu1 = F.conv2d(p, kernel, **opts)
    mu2 = F.conv2d(g, kernel, **opts)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(p * p, kernel, **opts) - mu1_sq
    s2 = F.conv2d(g * g, kernel, **opts) - mu2_sq
    s12 = F.conv2d(p * g, kernel, **opts) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean().item()


def mae(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return (pred - gt).abs().mean().item()


# ---------------------------------------------------------------------------
# LPIPS loader (uses the repo's own lpipsPyTorch + lpips_alex.pt)
# ---------------------------------------------------------------------------

def _find_lpips_model(result_dir: str, cli_path: str | None) -> str | None:
    if cli_path and os.path.isfile(cli_path):
        return cli_path
    candidates = [
        os.path.join(result_dir, "..", "..", "src", "lpips", "lpips_alex.pt"),
        os.path.join(os.path.dirname(__file__), "..", "src", "lpips", "lpips_alex.pt"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.realpath(c)
    return None


def load_lpips(model_path: str) -> "torch.jit.ScriptModule | None":
    try:
        m = torch.jit.load(model_path)
        m.eval()
        if torch.cuda.is_available():
            m.cuda()
        return m
    except Exception as exc:
        print(f"[WARN] Could not load LPIPS model ({model_path}): {exc}", file=sys.stderr)
        return None


def lpips_score(model, pred: torch.Tensor, gt: torch.Tensor) -> float:
    if model is None:
        return float("nan")
    p = pred.unsqueeze(0)
    g = gt.unsqueeze(0)
    if torch.cuda.is_available():
        p, g = p.cuda(), g.cuda()
    with torch.no_grad():
        return model(p, g).item()


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> torch.Tensor:
    """Load a JPEG/PNG as a float32 (C, H, W) tensor in [0, 1]."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# PNG plot
# ---------------------------------------------------------------------------

def _write_plots(rows: list, summary: dict, result_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[WARN] matplotlib not found — skipping metrics.png", file=sys.stderr)
        return

    # (label, row-key, summary-key, hex-color, higher-is-better)
    metric_defs = [
        ("PSNR (dB)", "psnr",  "mean_psnr",  "#4c9fd5", True),
        ("SSIM",      "ssim",  "mean_ssim",  "#5cb85c", True),
        ("LPIPS",     "lpips", "mean_lpips", "#e05c5c", False),
        ("MAE",       "mae",   "mean_mae",   "#f0a030", False),
    ]

    xs       = list(range(len(rows)))
    run_name = os.path.basename(result_dir.rstrip("/"))

    def _style_ax(ax):
        ax.set_facecolor("#1a1a1a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.tick_params(colors="#888888", labelsize=8)
        ax.xaxis.label.set_color("#888888")
        ax.yaxis.label.set_color("#888888")
        ax.title.set_color("#cccccc")
        ax.grid(True, color="#2a2a2a", linewidth=0.6)

    # ── Chart 1: per-frame line plots (2×2) ───────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig1.patch.set_facecolor("#0f0f0f")
    fig1.suptitle(
        f"GaussianLIC Evaluation — {run_name}  ({summary['n_frames']} frames)",
        color="#dddddd", fontsize=13, fontweight="bold", y=0.99,
    )

    for ax, (label, key, mean_key, color, higher_better) in zip(axes.flat, metric_defs):
        _style_ax(ax)

        vals     = [r[key] for r in rows if r[key] is not None]
        valid_xs = [x for x, r in zip(xs, rows) if r[key] is not None]

        if not vals:
            ax.set_title(f"{label}  (N/A)")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="#555555")
            continue

        mean_val = summary.get(mean_key)

        ax.fill_between(valid_xs, vals, alpha=0.15, color=color)
        ax.plot(valid_xs, vals, color=color, linewidth=1.2, zorder=3)
        ax.scatter(valid_xs, vals, color=color, s=14, zorder=4, edgecolors="none")

        if mean_val is not None:
            ax.axhline(mean_val, color=color, linewidth=1.0, linestyle="--", alpha=0.75)
            ax.text(
                0.99, mean_val, f" μ={mean_val:.4g}",
                transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color=color, alpha=0.9,
            )

        direction = "↑ better" if higher_better else "↓ better"
        ax.set_title(f"{label}   {direction}", fontsize=10, pad=5)
        ax.set_xlabel("Frame index", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
        ax.set_xlim(-0.5, len(xs) - 0.5)

        best_i  = int(np.argmax(vals) if higher_better else np.argmin(vals))
        worst_i = int(np.argmin(vals) if higher_better else np.argmax(vals))
        span = max(vals) - min(vals) + 1e-9
        for idx, tag, dy in [(best_i, "best", 0.04), (worst_i, "worst", -0.07)]:
            ax.annotate(
                f"{tag}\n{vals[idx]:.4g}",
                xy=(valid_xs[idx], vals[idx]),
                xytext=(valid_xs[idx], vals[idx] + dy * span),
                ha="center", fontsize=7, color="#aaaaaa",
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6),
            )

    fig1.tight_layout(rect=[0, 0, 1, 0.97])
    per_frame_path = os.path.join(result_dir, "metrics.png")
    fig1.savefig(per_frame_path, dpi=130, facecolor=fig1.get_facecolor(), bbox_inches="tight")
    plt.close(fig1)
    print(f"[OUT] {per_frame_path}")

    # ── Chart 2: summary bar chart (one bar per metric, mean ± std) ───────
    mean_vals  = [summary.get(m[2]) for m in metric_defs]
    std_vals   = [
        float(np.std([r[m[1]] for r in rows if r[m[1]] is not None]))
        for m in metric_defs
    ]
    bar_labels = [m[0] for m in metric_defs]
    bar_colors = [m[3] for m in metric_defs]
    directions = ["↑" if m[4] else "↓" for m in metric_defs]

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    fig2.patch.set_facecolor("#0f0f0f")
    _style_ax(ax2)
    ax2.grid(True, color="#2a2a2a", linewidth=0.6, axis="y")

    bar_positions = list(range(len(metric_defs)))
    bars = ax2.bar(
        bar_positions,
        [v if v is not None else 0.0 for v in mean_vals],
        yerr=[s for s in std_vals],
        color=bar_colors,
        alpha=0.82,
        width=0.55,
        error_kw=dict(ecolor="#555555", capsize=5, capthick=1.2, elinewidth=1.2),
        zorder=3,
    )

    # Value label above each bar
    for bar, val, std, direction in zip(bars, mean_vals, std_vals, directions):
        if val is None:
            continue
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.005 * (ax2.get_ylim()[1] - ax2.get_ylim()[0] + 1e-9),
            f"{val:.4g}\n±{std:.3g}  {direction}",
            ha="center", va="bottom", fontsize=9, color="#cccccc",
        )

    ax2.set_xticks(bar_positions)
    ax2.set_xticklabels(bar_labels, fontsize=10, color="#cccccc")
    ax2.set_ylabel("Mean value", fontsize=10)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
    ax2.set_title(
        f"Mean Metrics — {run_name}  ({summary['n_frames']} frames)",
        color="#dddddd", fontsize=12, pad=8,
    )

    fig2.tight_layout()
    summary_path = os.path.join(result_dir, "metrics_summary.png")
    fig2.savefig(summary_path, dpi=130, facecolor=fig2.get_facecolor(), bbox_inches="tight")
    plt.close(fig2)
    print(f"[OUT] {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate GaussianLIC render quality")
    ap.add_argument("result_dir", help="Path to a GaussianLIC result directory")
    ap.add_argument("--lpips-model", default=None, help="Override path to lpips_alex.pt")
    args = ap.parse_args()

    result_dir = os.path.realpath(args.result_dir)
    gt_dir     = os.path.join(result_dir, "gt")
    render_dir = os.path.join(result_dir, "render")

    for d in (gt_dir, render_dir):
        if not os.path.isdir(d):
            sys.exit(f"[ERROR] Missing directory: {d}")

    gt_files = sorted(
        f for f in os.listdir(gt_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    render_files = set(os.listdir(render_dir))

    paired = [(f, os.path.join(gt_dir, f), os.path.join(render_dir, f))
              for f in gt_files if f in render_files]

    if not paired:
        sys.exit("[ERROR] No matching filenames found between gt/ and render/")

    lpips_path = _find_lpips_model(result_dir, args.lpips_model)
    lpips_model = None
    if lpips_path:
        lpips_model = load_lpips(lpips_path)
        if lpips_model:
            print(f"[INFO] LPIPS model: {lpips_path}")
    else:
        print("[WARN] lpips_alex.pt not found — LPIPS will be NaN", file=sys.stderr)

    # Header
    col_w = [28, 8, 8, 8, 8]
    header = f"{'Frame':<{col_w[0]}} {'PSNR':>{col_w[1]}} {'SSIM':>{col_w[2]}} {'LPIPS':>{col_w[3]}} {'MAE':>{col_w[4]}}"
    print()
    print(header)
    print("-" * sum(col_w) + "----")

    rows = []
    total_psnr = total_ssim = total_lpips = total_mae = 0.0
    lpips_count = 0

    for fname, gt_path, render_path in paired:
        try:
            gt_img     = load_image(gt_path)
            render_img = load_image(render_path)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}", file=sys.stderr)
            continue

        # Resize render to gt size if they differ (shouldn't happen, but be safe)
        if render_img.shape != gt_img.shape:
            render_np = render_img.permute(1, 2, 0).numpy()
            h, w = gt_img.shape[1], gt_img.shape[2]
            render_np = cv2.resize(render_np, (w, h), interpolation=cv2.INTER_LINEAR)
            render_img = torch.from_numpy(render_np).permute(2, 0, 1)

        p  = psnr(render_img, gt_img)
        s  = ssim(render_img, gt_img)
        lp = lpips_score(lpips_model, render_img, gt_img)
        m  = mae(render_img, gt_img)

        total_psnr += p
        total_ssim += s
        total_mae  += m
        if not np.isnan(lp):
            total_lpips += lp
            lpips_count += 1

        lp_str = f"{lp:.4f}" if not np.isnan(lp) else "   N/A"
        print(f"{fname:<{col_w[0]}} {p:>{col_w[1]}.2f} {s:>{col_w[2]}.4f} {lp_str:>{col_w[3]}} {m:>{col_w[4]}.4f}")

        rows.append({
            "frame":  fname,
            "psnr":   round(p,  4),
            "ssim":   round(s,  4),
            "lpips":  round(lp, 4) if not np.isnan(lp) else None,
            "mae":    round(m,  4),
        })

    n = len(rows)
    if n == 0:
        sys.exit("[ERROR] No frames evaluated.")

    mean_psnr  = total_psnr / n
    mean_ssim  = total_ssim / n
    mean_mae   = total_mae  / n
    mean_lpips = total_lpips / lpips_count if lpips_count else float("nan")

    print("-" * sum(col_w) + "----")
    lp_mean_str = f"{mean_lpips:.4f}" if not np.isnan(mean_lpips) else "   N/A"
    print(f"{'MEAN  (' + str(n) + ' frames)':<{col_w[0]}} {mean_psnr:>{col_w[1]}.2f} {mean_ssim:>{col_w[2]}.4f} {lp_mean_str:>{col_w[3]}} {mean_mae:>{col_w[4]}.4f}")
    print()

    summary = {
        "n_frames":  n,
        "mean_psnr":  round(mean_psnr,  4),
        "mean_ssim":  round(mean_ssim,  4),
        "mean_lpips": round(mean_lpips, 4) if not np.isnan(mean_lpips) else None,
        "mean_mae":   round(mean_mae,   4),
    }

    # Write CSV
    csv_path = os.path.join(result_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "psnr", "ssim", "lpips", "mae"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OUT] {csv_path}")

    # Write JSON summary
    json_path = os.path.join(result_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OUT] {json_path}")

    # Write PNG plots
    _write_plots(rows, summary, result_dir)


if __name__ == "__main__":
    main()
