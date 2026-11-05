"""Part 3 support code: the camera of the assignment, the objects and the plotting helpers. Students implement
`project`, `cube_pose` and choose the camera parameters; nothing in this file solves those tasks.

Notation (Lecture 02.1, Collins):  p_W = (U, V, W)  world,  p_C = (X, Y, Z) = R (p_W - C)  camera,  (x, y) film,  (u, v) pixel (v DOWN)
    u = (f / s_x) X / Z + o_x ,   v = (f / s_y) Y / Z + o_y ,   K = [[f/s_x, 0, o_x], [0, f/s_y, o_y], [0, 0, 1]]
"""
from __future__ import annotations

import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from . import params as P
from .part2 import Rx, Ry, Rz  # noqa: F401  (re-exported: students use them to build rotations)

IMAGE_W, IMAGE_H = P.P3_IMAGE
F, (SX, SY) = P.P3_F, P.P3_S
O0 = np.array(P.P3_O, float)
C0 = np.array(P.P3_C, float)
R0 = np.array(P.P3_R, float)
L = P.P3_CUBE_EDGE
RECT1 = np.array(P.P3_RECT1, float)
RECT2 = np.array(P.P3_RECT2, float)

CUBE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def intrinsics(f, s_x, s_y, o_x, o_y):
    """K = [[f/s_x, 0, o_x], [0, f/s_y, o_y], [0, 0, 1]]  (slide 77)."""
    return np.array([[f / s_x, 0.0, o_x], [0.0, f / s_y, o_y], [0.0, 0.0, 1.0]])


K0 = intrinsics(F, SX, SY, *O0)


def cube_vertices():
    """The 8 cube vertices in their initial pose. x1..x4 = front face (nearer to the camera, in the order top-left, top-right,
    bottom-right, bottom-left of the cube), x5..x8 = the back face, x_{i+4} lies behind x_i."""
    h = L / 2
    loc = np.array([[-h, -h, h], [h, -h, h], [h, -h, -h], [-h, -h, -h], [-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h]])
    p = P.P3_CUBE_POSE
    Rc = Rz(np.deg2rad(p["yaw"])) @ Rx(np.deg2rad(p["pitch"])) @ Ry(np.deg2rad(p["roll"]))
    return loc @ Rc.T + np.array(p["centre"])


def rigid(V, R_obj, t_obj):
    """Apply the rigid motion p -> R_obj p + t_obj to every row of V."""
    return np.asarray(V, float) @ np.asarray(R_obj, float).T + np.asarray(t_obj, float)


def depth(R, C, p):
    """Z coordinate of p in the camera frame."""
    return float((np.asarray(R) @ (np.asarray(p, float) - np.asarray(C)))[2])


# --------------------------------------------------------------------------------------- plotting
def draw_image(objects: dict, project_fn, K, R, C, *, cube=None, title="", ax=None, highlight_front=True):
    """Image plane (u right, v DOWN) with the objects projected by YOUR `project(K, R, C, p_W)`."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlim(0, IMAGE_W)
    ax.set_ylim(IMAGE_H, 0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$u$ [px]")
    ax.set_ylabel(r"$v$ [px]  (points down)")
    ax.plot([K[0, 2]], [K[1, 2]], "+", color="gray", ms=12, mew=1.5)
    ax.annotate(r"$(o_x,o_y)$", (K[0, 2], K[1, 2]), xytext=(8, 14), textcoords="offset points", fontsize=9, color="0.4",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    todo = False
    colors = ["tab:green", "tab:purple", "tab:orange", "tab:red"]
    for i, (name, pts) in enumerate(objects.items()):
        if pts is None:
            continue
        uv = [project_fn(K, R, C, np.asarray(p, float)) for p in np.asarray(pts, float)]
        if any(x is None for x in uv):
            todo = True
            continue
        uv = np.array([np.asarray(x, float) for x in uv])
        loop = np.vstack([uv, uv[:1]])
        ax.fill(loop[:, 0], loop[:, 1], color=colors[i % 4], alpha=0.2)
        ax.plot(loop[:, 0], loop[:, 1], "-o", color=colors[i % 4], ms=4, label=name)
    if cube is not None:
        uv = [project_fn(K, R, C, np.asarray(p, float)) for p in np.asarray(cube, float)]
        if any(x is None for x in uv):
            todo = True
        else:
            uv = np.array([np.asarray(x, float) for x in uv])
            for a, b in CUBE_EDGES:
                front = a < 4 and b < 4
                ax.plot(*zip(uv[a], uv[b]), "-", color="tab:red" if (front and highlight_front) else "0.35", lw=2.2 if front else 1.2)
            mid = uv[:4].mean(axis=0)
            for i, p in enumerate(uv[:4], 1):  # label each front corner outside the face
                d = p - mid
                d = d / (np.linalg.norm(d) or 1.0)
                ax.annotate(rf"$x_{i}$", p, xytext=(11 * d[0], -11 * d[1]), textcoords="offset points", fontsize=9, color="tab:red",
                            ha="center", va="center", path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    if todo:
        ax.text(IMAGE_W / 2, IMAGE_H / 2, "implement project(...) first", ha="center", va="center", fontsize=12)
    if objects and any(v is not None for v in objects.values()):
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title, fontsize=10)
    return ax


def draw_topview(objects, R, C, K=None, *, cube=None, ax=None, title=""):
    """View of the scene from above (the U-V plane, W points towards you): camera centre, optical axis, horizontal field of view,
    and the footprints of the objects."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.8))
    R, C = np.asarray(R, float), np.asarray(C, float)
    K = K0 if K is None else K
    ax.set_xlim(-4, 9)
    ax.set_ylim(-8, 13)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$U$")
    ax.set_ylabel(r"$V$")
    for u, ls in ((0, "--"), (IMAGE_W, "--")):                            # horizontal field of view
        d = R.T @ np.array([(u - K[0, 2]) / K[0, 0], 0.0, 1.0])
        ax.plot([C[0], C[0] + 25 * d[0]], [C[1], C[1] + 25 * d[1]], ls, color="0.55", lw=1)
    ax.annotate("", C[:2] + 3.0 * R[2][:2] / max(np.linalg.norm(R[2][:2]), 1e-9), C[:2], arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.8))
    ax.plot(*C[:2], "o", color="k", ms=6)
    ax.annotate(r"$\mathbf{C}$", C[:2] + np.array([0.25, -0.9]), fontsize=11)
    colors = ["tab:green", "tab:purple", "tab:orange", "tab:red"]
    for i, (name, pts) in enumerate((objects or {}).items()):
        if pts is None:
            continue
        q = np.asarray(pts, float)[:, :2]
        ax.fill(q[:, 0], q[:, 1], color=colors[i % 4], alpha=0.25)
        ax.plot(*np.vstack([q, q[:1]]).T, "-", color=colors[i % 4], lw=1.5, label=name)
    if cube is not None:
        c = np.asarray(cube, float)[:, :2]
        for a, b in CUBE_EDGES:
            front = a < 4 and b < 4
            ax.plot(*zip(c[a], c[b]), "-", color="tab:red" if front else "0.35", lw=2.0 if front else 1.0)
    if objects:
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title or "top view", fontsize=10)
    return ax


def draw_scene(objects: dict, project_fn, K, R, C, *, cube=None, title=""):
    """Top view of the scene (left) next to the image produced by YOUR `project` (right)."""
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw=dict(width_ratios=[1, 1.35]))
    draw_topview(objects, R, C, K, cube=cube, ax=axs[0])
    draw_image(objects, project_fn, K, R, C, cube=cube, ax=axs[1], title=title or "image")
    fig.tight_layout()
    plt.show()
