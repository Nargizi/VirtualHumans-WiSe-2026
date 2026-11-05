"""Part 2 support code: the GIVEN conversions, sequence helper and diagnostics. Students implement the interpolations.

Notation (lecture 02, part 2):
    R(alpha, beta, gamma) = R_x(alpha) R_y(beta) R_z(gamma)          (active, counter-clockwise; slides 14, 17)
    q = (q_w, q_x, q_y, q_z) = (q_w, v), scalar first;  q1 o q2 = (w1 w2 - v1.v2,  w1 v2 + w2 v1 + v1 x v2)
    R(q) q-sandwich form: for a UNIT quaternion R(q) is a rotation, for a non-unit q it equals ||q||^2 times a rotation.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .params import SEQUENCES_DEG, N_FRAMES
from matplotlib.patches import Polygon


# ---------------------------------------------------------------------------- Euler angles
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_to_matrix(angles):
    """R(alpha, beta, gamma) = R_x(alpha) R_y(beta) R_z(gamma); angles in radians."""
    a, b, g = angles
    return Rx(a) @ Ry(b) @ Rz(g)


# ---------------------------------------------------------------------------- quaternions (given part)
def euler_to_quat(angles):
    """q_x(alpha) o q_y(beta) o q_z(gamma) with q_a(theta) = (cos(theta/2), sin(theta/2) * axis); angles in radians.

    Written out in closed form (c = cos of a half angle, s = sin of a half angle).  The result is NOT sign-normalised: it is
    the raw product, so q and -q both occur (see the double-cover question).
    """
    cx, cy, cz = np.cos(np.asarray(angles, float) / 2)
    sx, sy, sz = np.sin(np.asarray(angles, float) / 2)
    return np.array([cx * cy * cz - sx * sy * sz,
                     sx * cy * cz + cx * sy * sz,
                     cx * sy * cz - sx * cy * sz,
                     cx * cy * sz + sx * sy * cz])


def quat_to_matrix(q):
    """Matrix of x -> q o x o conj(q).  For a unit quaternion this is the rotation R(q); for a non-unit quaternion
    it is ||q||^2 * R(q/||q||)  (the quaternion is deliberately NOT normalised here)."""
    w, x, y, z = q
    return np.array([[w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_norm(q):
    return float(np.sqrt(np.dot(q, q)))


def quat_inv(q):
    """q^-1 = conj(q) / ||q||^2"""
    return quat_conj(q) / np.dot(q, q)


# ---------------------------------------------------------------------------- sequences
def interpolate_sequence(waypoints_deg, func, n=N_FRAMES, kind="euler"):
    """Apply `func` to every consecutive pair of waypoints and stack the results as an array of 3x3 matrices (M,3,3).

    kind="euler": func(angles0, angles1, n) -> (n,3) Euler angles (radians)      [section 1]
    kind="quat":  func(q0, q1, n)            -> (n,4) quaternions (scalar first)   [sections 2-4]
    The last frame of a segment is the first of the next one and is not repeated.
    Returns None while `func` is not implemented.
    """
    pts = np.deg2rad(np.asarray(waypoints_deg, float))
    out = []
    for i in range(len(pts) - 1):
        if kind == "euler":
            seg = func(pts[i], pts[i + 1], n)
            if seg is None:
                return None
            mats = [euler_to_matrix(a) for a in np.asarray(seg, float)]
        else:
            seg = func(euler_to_quat(pts[i]), euler_to_quat(pts[i + 1]), n)
            if seg is None:
                return None
            mats = [quat_to_matrix(q) for q in np.asarray(seg, float)]
        out.extend(mats if i == 0 else mats[1:])
    return np.array(out)


# ---------------------------------------------------------------------------- diagnostics
def object_scale(M):
    """Linear size factor of the object drawn with M.  A quaternion q gives M = ||q||^2 R, so this is ||q||^2."""
    return float(np.cbrt(np.linalg.det(M)))


def angular_step_deg(Rs):
    """Rotation angle (degrees) between consecutive frames, ignoring the scale of the matrices."""
    Rs = np.asarray(Rs, float)
    Rn = Rs / np.cbrt(np.linalg.det(Rs))[:, None, None]
    c = (np.einsum("nij,nij->n", Rn[:-1], Rn[1:]) - 1) / 2             # trace(R_k^T R_{k+1}) = <R_k, R_{k+1}>
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def swept_angle_deg(Rs):
    """Total rotation angle travelled along the frames (sum of the steps)."""
    return float(np.sum(angular_step_deg(Rs)))


def plot_metrics(frames: dict, title: str = ""):
    """Static figure for the PDF export: object scale ||q||^2 and angular speed along the frames of each method."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    for name, Rs in frames.items():
        if Rs is None:
            continue
        Rs = np.asarray(Rs, float)
        axs[0].plot([object_scale(M) for M in Rs], label=name)
        axs[1].plot(angular_step_deg(Rs), label=name)
    lo, hi = [], []
    for Rs in frames.values():
        if Rs is not None:
            sc = [object_scale(M) for M in np.asarray(Rs, float)]
            lo.append(min(sc)); hi.append(max(sc))
    if lo:                                           # a constant scale 1.0 must not be autoscaled to rounding noise
        axs[0].set_ylim(min(min(lo), 1.0) - 0.05, max(max(hi), 1.0) + 0.05)
    axs[0].set_title(r"object scale $\|\mathbf{q}\|^2$ (1 = unchanged)")
    axs[1].set_title("angular speed [degrees per frame]")
    for ax in axs:
        ax.set_xlabel("frame $k$")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    plt.show()


# ---- a static picture of the motion: the letter F drawn at equally spaced frames (painter's algorithm, orthographic view)
_F_CELLS = [(0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (1, 0)]      # unit cubes of the letter F (x, y)
_QUADS = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
_VIEW = Rx(np.deg2rad(-22)) @ Ry(np.deg2rad(28))


def _f_faces():
    half = 0.47 * 0.4
    base = half * np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    return [(base[list(q)] + c, c) for (cx, cy) in _F_CELLS for c in [np.array([(cx - 0.8) * 0.4, cy * 0.4, 0.0])] for q in _QUADS]


def draw_F(ax, M, view=_VIEW):
    """Draw the letter F transformed by the 3x3 matrix M (a rotation, or ||q||^2 times a rotation) into the axes `ax`."""
    light = np.array([0.3, 0.5, 0.8]); light /= np.linalg.norm(light)
    polys = []
    for verts, centre in _f_faces():
        v = (view @ (M @ verts.T)).T
        c = view @ (M @ centre)
        n = np.cross(v[1] - v[0], v[2] - v[0])
        if n @ (v.mean(axis=0) - c) < 0:
            n = -n
        if n[2] <= 0:                                                # back face
            continue
        n = n / (np.linalg.norm(n) + 1e-12)
        polys.append((v[:, 2].mean(), v[:, :2], 0.45 + 0.5 * max(0.0, n @ light)))
    for _, xy, shade in sorted(polys, key=lambda t: t[0]):
        ax.add_patch(Polygon(xy, closed=True, fc=(shade * 0.72, shade * 0.76, shade * 0.80), ec="0.15", lw=0.4))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")


def filmstrip(frames: dict, n_cols: int = 7, title: str = ""):
    """One row per method, `n_cols` equally spaced frames per row: the static counterpart of an animation (also in a PDF export)."""
    frames = {k: np.asarray(v, float) for k, v in frames.items() if v is not None}
    if not frames:
        print("Nothing to draw yet: implement the interpolation first.")
        return
    fig, axs = plt.subplots(len(frames), n_cols, figsize=(1.55 * n_cols, 1.55 * len(frames) + 0.5), squeeze=False)
    for r, (name, Rs) in enumerate(frames.items()):
        for c, k in enumerate(np.linspace(0, len(Rs) - 1, n_cols).round().astype(int)):
            draw_F(axs[r, c], Rs[k])
            if r == 0:
                axs[r, c].set_title(f"$k={k}$", fontsize=9)
        axs[r, 0].text(-0.08, 0.5, name, transform=axs[r, 0].transAxes, ha="right", va="center", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.subplots_adjust(left=0.14, right=0.995, wspace=0.0, hspace=0.0, top=0.9 if title else 0.93, bottom=0.01)
    plt.show()


SEQ = {k: v for k, v in SEQUENCES_DEG.items()}
