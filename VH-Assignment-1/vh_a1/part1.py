"""Part 1 support code (plotting + live widgets). Students do NOT implement anything in this file.

Notation (see the notation sheet in the notebook):
    R(theta)      2D rotation matrix            q_i   rest position of joint i
    T(q, theta)   rotation about the point q     e     arm end (grasping point)
    c^+ / c^-     claw tips
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .params import WINDOW, ZOOM, Q1

LINK_COLORS = ["tab:green", "tab:red", "tab:orange", "tab:purple"]


def new_axes(title: str | None = None, window=WINDOW, figsize=(5.2, 5.2), ax=None):
    """Square plot window with grid and the labels used in the notation sheet."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*window)
    ax.set_ylim(*window)
    ax._vh_span = window[1] - window[0]          # lets draw_chain scale labels / frames to the zoom level
    ax.set_aspect("equal")
    ax.grid(color="k", linestyle="--", linewidth=0.8, alpha=0.2)
    ax.set_xlabel(r"$x$ (spatial frame S)")
    ax.set_ylabel(r"$y$ (spatial frame S)")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def draw_chain(ax, joints, end, claw_up, claw_lo, *, thetas=None, alpha=1.0, label=True, frames=False):
    """Draw a planar arm.

    joints   (n,2) current joint positions   end  (2,) arm end e
    claw_up, claw_lo  (2,) absolute positions of the claw tips (not offsets)
    thetas   joint angles (rad), only needed when frames=True (draws the orientation of each link)
    """
    joints = np.asarray(joints, float)
    end = np.asarray(end, float)
    k = getattr(ax, "_vh_span", 6.0) / 6.0       # 1.0 for the full [-3,3] window, smaller when zoomed
    pts = np.vstack([joints, end])
    for i in range(len(pts) - 1):
        ax.plot(*zip(pts[i], pts[i + 1]), "-", lw=4, color=LINK_COLORS[i % len(LINK_COLORS)], alpha=alpha, solid_capstyle="round")
    for tip in (claw_up, claw_lo):
        ax.plot(*zip(end, np.asarray(tip, float)), "-", lw=1.8, color="k", alpha=alpha)
    ax.plot(*joints.T, "o", mfc="white", mec="tab:blue", mew=2, ms=11, alpha=alpha, zorder=3)
    ax.plot(*end, "s", color="k", ms=5, alpha=alpha, zorder=3)
    if label:
        for i, q in enumerate(joints, 1):
            ax.annotate(rf"$\mathbf{{q}}_{i}$", q + k * np.array([0.1, -0.3]), fontsize=11)
        ax.annotate(r"$\mathbf{e}$", end + k * np.array([0.1, 0.12]), fontsize=11)
    if frames and thetas is not None:
        cum = np.cumsum(thetas)
        for q, a in zip(joints, cum):
            for ang, col in ((a, "tab:red"), (a + np.pi / 2, "tab:green")):
                ax.annotate("", q + k * 0.4 * np.array([np.cos(ang), np.sin(ang)]), q,
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=0.8))
    return ax


def coverage(points, window=WINDOW, cell: float = 0.2) -> float | None:
    """Fraction of the window's cells (side `cell`) that contain at least one of the given points.

    Feed it the output of YOUR `workspace(links, n=200)`: it measures what your arm actually reaches. Returns None if no points yet.
    """
    if points is None:
        return None
    pts = np.asarray(points, float).reshape(-1, 2)
    n = int(round((window[1] - window[0]) / cell))
    ij = np.floor((pts - window[0]) / cell).astype(int)
    ij = ij[(ij >= 0).all(axis=1) & (ij < n).all(axis=1)]
    hit = np.zeros((n, n), bool)
    hit[ij[:, 0], ij[:, 1]] = True
    return float(hit.mean())


def scatter_workspace(ax, pts, s=2, color="tab:red", alpha=0.6):
    pts = np.asarray(pts, float).reshape(-1, 2)
    ax.scatter(pts[:, 0], pts[:, 1], s=s, marker="x", color=color, alpha=alpha, linewidths=0.6)
    return ax
