"""Numbers used by Assignment 1 (single place to change them; see DESIGN.md §5).

All lengths are in the arbitrary world units of the plots; all angles in these tables are
given in DEGREES for readability and are converted with np.deg2rad where they are used.
"""
import numpy as np

# ------------------------------------------------------------------ Part 1: planar chains
WINDOW = (-3.0, 3.0)                     # plotted window [-3, 3]^2 (workspace plots, section 4)
ZOOM = (-1.6, 1.6)                       # zoomed window for the pose plots (sections 1-3, 5)

Q1 = np.array([-0.4, 0.3])               # rest position q_1 of joint 1
LINKS_2 = np.array([[0.70, 0.0],         # a_1  (rest link vectors, joint i -> joint i+1)
                    [0.25, 0.0]])        # a_2  (joint 2 -> arm end)
LINKS_3 = np.array([[0.35, 0.0],
                    [0.35, 0.0],
                    [0.35, 0.0]])
CLAW_UP = np.array([0.15, 0.15])         # c^+  (rest offsets of the claw tips from the arm end)
CLAW_LO = np.array([0.15, -0.15])        # c^-

THETA_SEC1_DEG = 50.0                    # section 1: single joint
N_SEC2 = 50                              # section 2: number of angles
THETA_SEC3A_DEG = (70.0, 0.0)            # section 3 (a)
THETA_SEC3B_DEG = (70.0, 40.0)           # section 3 (b)
N_SEC4 = 100                             # section 4: angles per joint (grid N x N)
THETA_SEC5_DEG = (40.0, -80.0, 40.0)     # section 5: three joints


def rest_chain(links, q1=Q1):
    """Rest joint positions Q (n,2) and rest arm end e (2,) for link vectors `links` (n,2).

    Q[i] = q_{i+1} = q_1 + a_1 + ... + a_i ; e = q_1 + a_1 + ... + a_n
    """
    links = np.asarray(links, float)
    pos = np.vstack([q1, q1 + np.cumsum(links, axis=0)])
    return pos[:-1].copy(), pos[-1].copy()


# ------------------------------------------------------------------ Part 2: rotation interpolation
# Waypoints are Euler angles (alpha, beta, gamma) in DEGREES in the lecture convention R = R_x(alpha) R_y(beta) R_z(gamma).
SEQUENCES_DEG = {
    "seq0": [(0, 0, -125), (0, 0, 125)],                                        # one axis, big wrap-around
    "seq1": [(40, 40, 0), (40, 90, 0), (-140, 50, 180), (40, 40, 0)],           # passes through beta = 90 deg (gimbal lock)
    "seq2": [(0, 0, 0), (0, 0, 60), (180, 0, 60), (180, 90, 0), (0, 0, 0)],     # contains two 180-degree segments
    "seq3": [(-170, 0, 140), (-160, -40, -140), (-170, 0, 140)],                # orientations only ~85 deg apart, but gamma written 280 deg apart
}
N_FRAMES = 40                            # frames per segment in the animations (t_k = k/(n-1))


# ------------------------------------------------------------------ Part 3: camera model (Lecture 02.1 notation)
# World coordinates p_W = (U, V, W) with W up.  Camera coordinates p_C = (X, Y, Z): X right, Y down, Z forward.  Pixel (u, v): v points DOWN.
P3_IMAGE = (640, 480)                    # image width x height in pixels
P3_F = 1.0                               # focal length f  [world length units]
P3_S = (0.0025, 0.0025)                  # pixel size (s_x, s_y) [length units per pixel]  ->  f / s = 400 px
P3_O = (220.0, 160.0)                    # principal point (o_x, o_y) in pixels (deliberately NOT the image centre)
P3_C = (2.0, -6.0, 1.5)                  # camera centre C in world coordinates
# Rows of R are the camera axes expressed in world coordinates (lecture slides 66-68): X = +U, Y = -W (down), Z = +V (forward)
P3_R = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
P3_CUBE_EDGE = 3.0                       # edge length L of the cube
P3_TARGET_PX = 120.0                     # section 2: side of the front face in the image [px]
P3_RECT1 = [[-1.0, 3.0, 1.5], [-1.0, 3.0, 3.0], [1.0, 3.0, 3.0], [1.0, 3.0, 1.5]]       # V = 3: parallel to the image plane
P3_RECT2 = [[-1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 10.0, 0.0], [-1.0, 10.0, 0.0]]     # W = 0: the ground, perpendicular to the image plane
# cube in its initial (tilted) pose: front face x1..x4 (nearer to the camera), back face x5..x8 behind it (x_{i+4} behind x_i)
P3_CUBE_POSE = {"yaw": 28.0, "pitch": -18.0, "roll": 22.0, "centre": (4.2, 3.0, 2.2)}
P3_RECT3_DEMO = [[3.0, 2.0, 0.5], [3.0, 4.0, 0.5], [3.0, 4.0, 2.0], [3.0, 2.0, 2.0]]   # the wall shown by default in the Lab (NOT the solution of section 1b)
