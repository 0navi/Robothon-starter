"""LEAP Pour Task — Robothon Summer 2026 submission.

A LEAP right hand grips a small open-top cup containing 8 colored balls.
The hand is carried by a 4-DoF mount (X/Y/Z translate + Y-axis tilt
hinge), and the program plays a scripted pour sequence:

    settle -> lift -> translate over big container -> tilt -> pour
           -> tilt back -> translate over small container -> tilt -> pour
           -> recover -> hold

Balls falling under gravity get scored by which container they land in
(big counts 1, small counts 2, miss counts 0). The hand is real, the
cup is real, every ball is a free body with gravity + contact friction.

Two modes:
    python main.py
        -> single deterministic 90 s run, renders demo.mp4 with HUD
    python main.py --multi-seed
        -> N=10 seeds perturbing ball initial positions, physics only,
        outputs multi_seed_stats.json
"""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import mujoco
import imageio.v3 as iio

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    _PIL_OK = False

HERE = Path(__file__).resolve().parent
HAND_XML = HERE / "assets" / "leap_hand" / "right_hand.xml"
OUT_VIDEO = HERE / "outputs" / "demo.mp4"
OUT_TRAJECTORY = HERE / "outputs" / "trajectory.json"
OUT_MULTI_SEED = HERE / "outputs" / "multi_seed_stats.json"

DT = 0.002
FPS = 30
RES_W, RES_H = 1280, 720
DURATION_S = 90.0

# Cup
CUP_OUTER_HALF = 0.025
CUP_WALL_T = 0.0015
CUP_HEIGHT_HALF = 0.015
CUP_DENSITY = 350.0
CUP_INIT_POS = (-0.022, 0.020, 0.346)   # in cage cup-center

# Balls
BALL_R = 0.003
N_BALLS = 8

# Target containers (on the floor near the workspace)
BIG_POS = (-0.13, 0.020, 0.0)      # big container origin (top of floor)
BIG_HALF = 0.045                    # big container half-width  (90 mm wide)
BIG_DEPTH_HALF = 0.020              # half-height = 40 mm tall

SMALL_POS = (0.13, 0.020, 0.0)
SMALL_HALF = 0.022                  # 44 mm wide
SMALL_DEPTH_HALF = 0.020            # 40 mm tall

# Phases (relative time, seconds)
PHASE_SETTLE_END     = 2.0
PHASE_LIFT_END       = 8.0     # lift + translate over big
PHASE_OVER_BIG_END   = 12.0    # hover briefly
PHASE_TILT_BIG_END   = 25.0    # tilt to ~110 deg, pour
PHASE_RETURN_BIG_END = 32.0    # tilt back, recover Z
PHASE_TO_SMALL_END   = 40.0    # translate over small container
PHASE_OVER_SMALL_END = 44.0
PHASE_TILT_SMALL_END = 57.0
PHASE_RETURN_SMALL_END = 65.0
PHASE_RECOVER_END    = 73.0
PHASE_HOLD_END       = 90.0

# LEAP cage pose (from cube version calibration)
CAGE_BASE = {
    "if_mcp": 1.30, "if_rot": 0.10, "if_pip": 1.00, "if_dip": 0.60,
    "mf_mcp": 1.30, "mf_rot": 0.00, "mf_pip": 1.00, "mf_dip": 0.60,
    "rf_mcp": 1.30, "rf_rot": -0.10, "rf_pip": 1.00, "rf_dip": 0.60,
    "th_cmc": 1.20, "th_axl": 0.80, "th_mcp": 1.20, "th_ipl": 1.00,
}

FINGER_COLORS = {
    "if": [0.30, 0.65, 1.00, 1.0],
    "mf": [0.30, 0.85, 0.45, 1.0],
    "rf": [1.00, 0.60, 0.25, 1.0],
    "th": [1.00, 0.85, 0.30, 1.0],
}
PALM_COLOR = [0.55, 0.55, 0.60, 1.0]

BALL_COLOR_TABLE = [
    [0.95, 0.30, 0.30],  # red
    [0.95, 0.65, 0.20],  # orange
    [0.95, 0.90, 0.30],  # yellow
    [0.40, 0.85, 0.45],  # green
    [0.30, 0.70, 0.95],  # cyan
    [0.45, 0.45, 0.95],  # blue
    [0.75, 0.40, 0.95],  # purple
    [0.95, 0.55, 0.85],  # pink
]


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def add_container(world, name, pos, half_w, depth_half, color):
    """4-wall + floor open-top container, static."""
    base = world.add_body(name=name, pos=list(pos))
    base.add_geom(name=f"{name}_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[0, 0, 0.001],
                  size=[half_w, half_w, 0.001],
                  rgba=color, friction=[1.0, 0.05, 0.001])
    for sign_x, sign_y, key in [(+1, 0, "xp"), (-1, 0, "xn"),
                                 (0, +1, "yp"), (0, -1, "yn")]:
        if sign_x != 0:
            wpos = [sign_x * (half_w - 0.001), 0, depth_half]
            wsize = [0.001, half_w, depth_half]
        else:
            wpos = [0, sign_y * (half_w - 0.001), depth_half]
            wsize = [half_w, 0.001, depth_half]
        base.add_geom(name=f"{name}_w_{key}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=wpos, size=wsize,
                      rgba=color, friction=[1.0, 0.05, 0.001])


def build_scene(seed: int = 0, ball_perturb: float = 0.0) -> mujoco.MjModel:
    host = mujoco.MjSpec()
    host.option.timestep = DT
    host.option.gravity = [0.0, 0.0, -9.81]
    host.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    host.visual.global_.offwidth = RES_W
    host.visual.global_.offheight = RES_H

    world = host.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[0, 0, 0.05], rgba=[0.07, 0.07, 0.09, 1.0])
    world.add_light(pos=[0.0, 0.3, 1.2], dir=[0.0, -0.2, -1.0],
                    diffuse=[1.0, 0.95, 0.85])
    world.add_light(pos=[0.4, -0.3, 1.0], dir=[-0.3, 0.2, -1.0],
                    diffuse=[0.4, 0.5, 0.7])
    world.add_light(pos=[-0.4, 0.0, 0.9], dir=[0.3, 0.0, -1.0],
                    diffuse=[0.55, 0.45, 0.4])

    # Two target containers
    add_container(world, "big",   BIG_POS,   BIG_HALF,   BIG_DEPTH_HALF,
                  color=[0.40, 0.65, 0.40, 1.0])
    add_container(world, "small", SMALL_POS, SMALL_HALF, SMALL_DEPTH_HALF,
                  color=[0.85, 0.55, 0.30, 1.0])

    # 4-DoF mount: X/Y/Z translate + Y-axis hinge for tilt
    mount_trans = world.add_body(name="mount_trans", pos=[0.0, 0.0, 0.20])
    mount_trans.add_joint(name="mount_x", type=mujoco.mjtJoint.mjJNT_SLIDE,
                          axis=[1, 0, 0], range=[-0.30, 0.30], damping=2.0)
    mount_trans.add_joint(name="mount_y", type=mujoco.mjtJoint.mjJNT_SLIDE,
                          axis=[0, 1, 0], range=[-0.30, 0.30], damping=2.0)
    mount_trans.add_joint(name="mount_z", type=mujoco.mjtJoint.mjJNT_SLIDE,
                          axis=[0, 0, 1], range=[-0.20, 0.30], damping=2.0)
    mount_trans.add_geom(name="trans_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.001, 0, 0], rgba=[1, 0, 0, 0],
                        contype=0, conaffinity=0)

    mount_rot = mount_trans.add_body(name="mount_rot", pos=[0.0, 0.0, 0.0])
    mount_rot.add_joint(name="mount_tilt", type=mujoco.mjtJoint.mjJNT_HINGE,
                        axis=[0, 1, 0], range=[-2.5, 2.5], damping=1.5)
    mount_rot.add_geom(name="rot_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[0.001, 0, 0], rgba=[0, 1, 0, 0],
                       contype=0, conaffinity=0)

    hand_spec = mujoco.MjSpec.from_file(str(HAND_XML))
    rot_frame = mount_rot.add_frame(pos=[0.0, 0.0, 0.0])
    host.attach(hand_spec, prefix="hand_", frame=rot_frame)

    # Cup
    cup = world.add_body(name="cup", pos=list(CUP_INIT_POS))
    cup.add_freejoint(name="cup_free")
    cup.add_geom(name="cup_bottom", type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=[0, 0, -CUP_HEIGHT_HALF + CUP_WALL_T],
                 size=[CUP_OUTER_HALF, CUP_OUTER_HALF, CUP_WALL_T],
                 rgba=[0.55, 0.55, 0.65, 1.0], density=CUP_DENSITY,
                 friction=[1.5, 0.1, 0.001])
    for sign_x, sign_y, name in [(+1, 0, "wall_xp"), (-1, 0, "wall_xn"),
                                  (0, +1, "wall_yp"), (0, -1, "wall_yn")]:
        if sign_x != 0:
            pos = [sign_x * (CUP_OUTER_HALF - CUP_WALL_T), 0, 0]
            size = [CUP_WALL_T, CUP_OUTER_HALF, CUP_HEIGHT_HALF]
        else:
            pos = [0, sign_y * (CUP_OUTER_HALF - CUP_WALL_T), 0]
            size = [CUP_OUTER_HALF, CUP_WALL_T, CUP_HEIGHT_HALF]
        cup.add_geom(name=f"cup_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                     pos=pos, size=size,
                     rgba=[0.55, 0.55, 0.65, 1.0], density=CUP_DENSITY,
                     friction=[1.5, 0.1, 0.001])

    # 8 balls — 2x2x2 grid in cup
    rng = np.random.default_rng(seed)
    z_base = CUP_INIT_POS[2] - CUP_HEIGHT_HALF + CUP_WALL_T + BALL_R + 0.001
    grid = []
    off = BALL_R + 0.001
    for layer in range(2):
        for ix in [-1, 1]:
            for iy in [-1, 1]:
                grid.append((ix * off, iy * off, layer * (2 * BALL_R + 0.001)))
    for i, (dx, dy, dz) in enumerate(grid[:N_BALLS]):
        # base jitter so balls aren't perfectly aligned
        jitter = 0.0005 + ball_perturb * 0.002
        bx = CUP_INIT_POS[0] + dx + rng.uniform(-jitter, jitter)
        by = CUP_INIT_POS[1] + dy + rng.uniform(-jitter, jitter)
        bz = z_base + dz
        col = BALL_COLOR_TABLE[i % len(BALL_COLOR_TABLE)]
        ball = world.add_body(name=f"ball_{i}", pos=[bx, by, bz])
        ball.add_freejoint(name=f"ball_{i}_free")
        ball.add_geom(name=f"ball_{i}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[BALL_R, 0, 0], rgba=[col[0], col[1], col[2], 1.0],
                      density=400.0, friction=[0.5, 0.05, 0.001])

    # Mount position actuators
    for jn, ctrl_lim in [("mount_x", 0.30), ("mount_y", 0.30), ("mount_z", 0.30)]:
        host.add_actuator(name=f"{jn}_act", target=jn,
                          trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                          gainprm=[400.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          biasprm=[0, -400.0, -40.0, 0, 0, 0, 0, 0, 0, 0],
                          ctrlrange=[-ctrl_lim, ctrl_lim])
    host.add_actuator(name="mount_tilt_act", target="mount_tilt",
                      trntype=mujoco.mjtTrn.mjTRN_JOINT,
                      gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                      biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                      gainprm=[80.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      biasprm=[0, -80.0, -20.0, 0, 0, 0, 0, 0, 0, 0],
                      ctrlrange=[-2.5, 2.5])

    return host.compile()


def colorize_fingers(model: mujoco.MjModel) -> None:
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not bname.startswith("hand_"):
            continue
        rest = bname[len("hand_"):]
        if rest.startswith("palm"):
            model.geom_rgba[gid] = PALM_COLOR
            continue
        fcode = rest[:2]
        if fcode in FINGER_COLORS:
            model.geom_rgba[gid] = FINGER_COLORS[fcode]


def get_actuator_map(model):
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)}


def apply_finger_pose(data, name2act, pose):
    for j, v in pose.items():
        key = f"hand_{j}_act"
        if key in name2act:
            data.ctrl[name2act[key]] = v


# ---------------------------------------------------------------------------
# Trajectory waypoints
# ---------------------------------------------------------------------------

def smooth_step(t, t0, t1):
    """Smooth ramp from 0 at t0 to 1 at t1 (cosine ease)."""
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    u = (t - t0) / (t1 - t0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


# Cup target XY over each container; container positions are world-frame, cup is
# offset from mount by the cage cup-center, so we drive the MOUNT XY = container
# XY minus the cup-mount offset (computed at runtime from the settled pose).

# Tilt magnitudes — sign chosen so cup opening points TOWARD each container.
# Positive tilt around +Y axis rotates cup +z toward +x.
# Big container at -X -> tilt -Y (negative angle).
# Small container at +X -> tilt +Y (positive angle).
TILT_BIG = -math.radians(110)
TILT_SMALL = math.radians(125)


@dataclass
class PourResult:
    seed: int
    cup_held_final: bool
    big_count: int
    small_count: int
    miss_count: int
    score: int     # 1 * big + 2 * small
    duration_s: float
    n_balls: int = N_BALLS
    big_balls: list = field(default_factory=list)
    small_balls: list = field(default_factory=list)
    miss_balls: list = field(default_factory=list)


def phase_at(t: float) -> str:
    if t < PHASE_SETTLE_END:      return "SETTLE"
    if t < PHASE_LIFT_END:        return "LIFT + APPROACH BIG"
    if t < PHASE_OVER_BIG_END:    return "OVER BIG"
    if t < PHASE_TILT_BIG_END:    return "POUR INTO BIG"
    if t < PHASE_RETURN_BIG_END:  return "TILT BACK"
    if t < PHASE_TO_SMALL_END:    return "APPROACH SMALL"
    if t < PHASE_OVER_SMALL_END:  return "OVER SMALL"
    if t < PHASE_TILT_SMALL_END:  return "POUR INTO SMALL"
    if t < PHASE_RETURN_SMALL_END:return "TILT BACK"
    if t < PHASE_RECOVER_END:     return "RECOVER"
    return "HOLD"


def control_targets(t: float, cup_offset_xy: tuple) -> tuple:
    """Return (mount_x_target, mount_y_target, mount_z_target, mount_tilt_target)
    given time t and the cup's XY offset from mount center (as measured after
    settle). The mount XY must compensate so that the CUP ends up over the
    target container."""
    cup_dx, cup_dy = cup_offset_xy
    big_mount_target = (BIG_POS[0] - cup_dx, BIG_POS[1] - cup_dy)
    small_mount_target = (SMALL_POS[0] - cup_dx, SMALL_POS[1] - cup_dy)

    # During SETTLE the mount stays at (0,0,0) — cup is at its init pos.
    if t < PHASE_SETTLE_END:
        return (0.0, 0.0, 0.0, 0.0)

    # LIFT + APPROACH BIG (ramp up Z by ~5cm, ramp x to big target)
    if t < PHASE_LIFT_END:
        u = smooth_step(t, PHASE_SETTLE_END, PHASE_LIFT_END)
        return (u * big_mount_target[0], u * big_mount_target[1], u * 0.05, 0.0)

    # OVER BIG: hover
    if t < PHASE_OVER_BIG_END:
        return (big_mount_target[0], big_mount_target[1], 0.05, 0.0)

    # POUR INTO BIG: ramp tilt to TILT_BIG
    if t < PHASE_TILT_BIG_END:
        u = smooth_step(t, PHASE_OVER_BIG_END, PHASE_TILT_BIG_END)
        return (big_mount_target[0], big_mount_target[1], 0.05, u * TILT_BIG)

    # TILT BACK
    if t < PHASE_RETURN_BIG_END:
        u = smooth_step(t, PHASE_TILT_BIG_END, PHASE_RETURN_BIG_END)
        return (big_mount_target[0], big_mount_target[1], 0.05, (1 - u) * TILT_BIG)

    # APPROACH SMALL
    if t < PHASE_TO_SMALL_END:
        u = smooth_step(t, PHASE_RETURN_BIG_END, PHASE_TO_SMALL_END)
        x = (1 - u) * big_mount_target[0] + u * small_mount_target[0]
        y = (1 - u) * big_mount_target[1] + u * small_mount_target[1]
        return (x, y, 0.05, 0.0)

    # OVER SMALL
    if t < PHASE_OVER_SMALL_END:
        return (small_mount_target[0], small_mount_target[1], 0.05, 0.0)

    # POUR INTO SMALL
    if t < PHASE_TILT_SMALL_END:
        u = smooth_step(t, PHASE_OVER_SMALL_END, PHASE_TILT_SMALL_END)
        return (small_mount_target[0], small_mount_target[1], 0.05, u * TILT_SMALL)

    # TILT BACK
    if t < PHASE_RETURN_SMALL_END:
        u = smooth_step(t, PHASE_TILT_SMALL_END, PHASE_RETURN_SMALL_END)
        return (small_mount_target[0], small_mount_target[1], 0.05, (1 - u) * TILT_SMALL)

    # RECOVER (lift higher, center)
    if t < PHASE_RECOVER_END:
        u = smooth_step(t, PHASE_RETURN_SMALL_END, PHASE_RECOVER_END)
        x = (1 - u) * small_mount_target[0]
        y = (1 - u) * small_mount_target[1]
        return (x, y, 0.05 + u * 0.05, 0.0)

    # HOLD
    return (0.0, 0.0, 0.10, 0.0)


# ---------------------------------------------------------------------------
# Ball-in-container detection (final & live)
# ---------------------------------------------------------------------------

def ball_classifier(ball_xyz: np.ndarray) -> str:
    """Where is this ball? 'big' / 'small' / 'miss' / 'cup'."""
    x, y, z = ball_xyz
    # Inside big container? (xy within bounds, z near container top, low enough)
    if (abs(x - BIG_POS[0]) < BIG_HALF and abs(y - BIG_POS[1]) < BIG_HALF
            and z < BIG_POS[2] + BIG_DEPTH_HALF * 2 + 0.005):
        return "big"
    if (abs(x - SMALL_POS[0]) < SMALL_HALF and abs(y - SMALL_POS[1]) < SMALL_HALF
            and z < SMALL_POS[2] + SMALL_DEPTH_HALF * 2 + 0.005):
        return "small"
    # On the floor outside containers?
    if z < 0.015:
        return "miss"
    return "cup"


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def _load_fonts():
    if not _PIL_OK:
        return None, None, None
    candidates = ["arial.ttf", "Arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
                  "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"]
    for c in candidates:
        try:
            return (ImageFont.truetype(c, 36),
                    ImageFont.truetype(c, 28),
                    ImageFont.truetype(c, 20))
        except Exception:
            continue
    f = ImageFont.load_default()
    return f, f, f


_FONT_HUGE, _FONT_BIG, _FONT_SM = _load_fonts()


def draw_overlay(frame, t, phase_name, tilt_deg,
                 big_n, small_n, miss_n, cup_n,
                 cup_held: bool) -> np.ndarray:
    if not _PIL_OK:
        return frame
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")

    # Top-center: phase banner
    bx = (RES_W - 520) // 2
    d.rectangle([(bx, 20), (bx + 520, 80)], fill=(0, 0, 0, 160))
    d.text((bx + 18, 28), f"PHASE: {phase_name}",
           fill=(245, 245, 245), font=_FONT_BIG)

    # Top-right: tilt angle
    d.rectangle([(RES_W - 220, 20), (RES_W - 20, 80)], fill=(0, 0, 0, 160))
    d.text((RES_W - 210, 30), f"tilt {tilt_deg:5.1f}°",
           fill=(220, 220, 240), font=_FONT_BIG)

    # Left: ball counters
    pw = 320
    d.rectangle([(20, 100), (20 + pw, 360)], fill=(0, 0, 0, 170))
    d.text((34, 110), "BALL COUNT", fill=(200, 200, 210), font=_FONT_SM)
    # big
    d.rectangle([(34, 142), (54, 162)], fill=(102, 166, 102))
    d.text((64, 138), f"big   : {big_n}", fill=(245, 245, 245), font=_FONT_BIG)
    # small
    d.rectangle([(34, 184), (54, 204)], fill=(217, 140, 76))
    d.text((64, 180), f"small : {small_n}  (×2)",
           fill=(245, 245, 245), font=_FONT_BIG)
    # miss
    d.rectangle([(34, 226), (54, 246)], fill=(180, 70, 70))
    d.text((64, 222), f"miss  : {miss_n}", fill=(245, 245, 245), font=_FONT_BIG)
    # in cup
    d.rectangle([(34, 268), (54, 288)], fill=(140, 140, 180))
    d.text((64, 264), f"in cup: {cup_n}", fill=(245, 245, 245), font=_FONT_BIG)
    # score
    score = big_n + 2 * small_n
    d.text((34, 308), f"SCORE: {score}",
           fill=(255, 230, 100), font=_FONT_HUGE)

    # Time
    d.rectangle([(20, RES_H - 60), (180, RES_H - 20)], fill=(0, 0, 0, 150))
    d.text((34, RES_H - 54), f"t = {t:5.2f} s",
           fill=(220, 220, 230), font=_FONT_BIG)

    # Held status
    if not cup_held:
        d.rectangle([(RES_W - 220, RES_H - 60), (RES_W - 20, RES_H - 20)],
                    fill=(180, 40, 40, 220))
        d.text((RES_W - 210, RES_H - 54), "CUP LOST",
               fill=(255, 250, 240), font=_FONT_BIG)

    return np.array(img)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(seed: int = 12345, render_video: bool = False,
             ball_perturb: float = 0.0) -> PourResult:
    model = build_scene(seed=seed, ball_perturb=ball_perturb)
    if render_video:
        colorize_fingers(model)
    data = mujoco.MjData(model)
    name2act = get_actuator_map(model)

    cup_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    ball_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ball_{i}")
                 for i in range(N_BALLS)]

    # Phase 0: brief settle to let balls and cup find their resting position.
    apply_finger_pose(data, name2act, CAGE_BASE)
    for jn in ["mount_x", "mount_y", "mount_z", "mount_tilt"]:
        data.ctrl[name2act[f"{jn}_act"]] = 0.0

    # Pre-settle 0.4 s for the cup grip to form
    for _ in range(int(0.4 / DT)):
        apply_finger_pose(data, name2act, CAGE_BASE)
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    cup_settled = data.xpos[cup_bid].copy()
    # cup offset from mount center (mount currently at origin XY)
    cup_offset_xy = (float(cup_settled[0]), float(cup_settled[1]))
    print(f"  [diag] cup settled at: {cup_settled}, offset_xy={cup_offset_xy}")

    # Camera
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.04, 0.20]
    cam.distance = 0.55
    cam.azimuth = 70.0
    cam.elevation = -18.0
    renderer = (mujoco.Renderer(model, width=RES_W, height=RES_H)
                if render_video else None)

    total_steps = int(DURATION_S / DT)
    steps_per_frame = max(1, int(round((1.0 / FPS) / DT)))
    frames = []
    cup_lost = False

    # Ball classification — sticky: once a ball is on the floor (z<0.015) inside
    # a container, lock it there. Otherwise re-evaluate each frame.
    ball_state = ["cup"] * N_BALLS

    last_phase = ""

    for step in range(total_steps):
        t = step * DT
        mx, my, mz, mtilt = control_targets(t, cup_offset_xy)
        data.ctrl[name2act["mount_x_act"]] = mx
        data.ctrl[name2act["mount_y_act"]] = my
        data.ctrl[name2act["mount_z_act"]] = mz
        data.ctrl[name2act["mount_tilt_act"]] = mtilt
        apply_finger_pose(data, name2act, CAGE_BASE)
        mujoco.mj_step(model, data)

        cup_xyz = data.xpos[cup_bid]
        # cup_held check: cup z above 0.1 (well above floor)
        if cup_xyz[2] < 0.15:
            cup_lost = True

        # Classify balls; lock final states once resting (z low or in container)
        for i, bid in enumerate(ball_bids):
            if ball_state[i] in ("big", "small", "miss"):
                continue  # sticky lock
            bx = data.xpos[bid]
            cls = ball_classifier(bx)
            if cls in ("big", "small", "miss"):
                ball_state[i] = cls

        if render_video and step % steps_per_frame == 0:
            cur_phase = phase_at(t)
            if cur_phase != last_phase:
                last_phase = cur_phase
            big_n = sum(1 for s in ball_state if s == "big")
            small_n = sum(1 for s in ball_state if s == "small")
            miss_n = sum(1 for s in ball_state if s == "miss")
            cup_n = sum(1 for s in ball_state if s == "cup")
            # Cinematic camera orbit
            cam.azimuth = 70.0 + 18.0 * math.sin(0.06 * t)
            cam.elevation = -18.0 + 3.0 * math.sin(0.04 * t)
            renderer.update_scene(data, camera=cam)
            raw = renderer.render().copy()
            overlaid = draw_overlay(
                raw, t, cur_phase, math.degrees(mtilt),
                big_n, small_n, miss_n, cup_n,
                cup_held=not cup_lost,
            )
            frames.append(overlaid)

    big_n = sum(1 for s in ball_state if s == "big")
    small_n = sum(1 for s in ball_state if s == "small")
    miss_n = sum(1 for s in ball_state if s == "miss")
    score = big_n + 2 * small_n

    # Diagnostic: dump each ball's final position
    print("  [diag] final ball positions:")
    for i, bid in enumerate(ball_bids):
        bp = data.xpos[bid]
        print(f"    ball {i}: state={ball_state[i]:>5s}  "
              f"xyz=({bp[0]:+.3f}, {bp[1]:+.3f}, {bp[2]:+.3f})")

    result = PourResult(
        seed=seed,
        cup_held_final=not cup_lost,
        big_count=big_n,
        small_count=small_n,
        miss_count=miss_n,
        score=score,
        duration_s=DURATION_S,
        big_balls=[i for i, s in enumerate(ball_state) if s == "big"],
        small_balls=[i for i, s in enumerate(ball_state) if s == "small"],
        miss_balls=[i for i, s in enumerate(ball_state) if s == "miss"],
    )

    if render_video:
        OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
        print(f"  encoding {len(frames)} frames -> {OUT_VIDEO.name}")
        iio.imwrite(OUT_VIDEO, np.asarray(frames), fps=FPS, codec="libx264")
        summary = {
            "project": "LEAP Pour Task",
            "seed": seed,
            "duration_s": DURATION_S,
            "fps": FPS,
            "resolution": [RES_W, RES_H],
            "n_balls": N_BALLS,
            "cup_held_final": result.cup_held_final,
            "big_count": result.big_count,
            "small_count": result.small_count,
            "miss_count": result.miss_count,
            "score_formula": "1 * big + 2 * small",
            "score": result.score,
            "big_ball_ids": result.big_balls,
            "small_ball_ids": result.small_balls,
            "miss_ball_ids": result.miss_balls,
        }
        OUT_TRAJECTORY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  metrics: {OUT_TRAJECTORY}")

    return result


def run_multi_seed(n_seeds: int = 10) -> dict:
    seeds = [12345 + 31 * i for i in range(n_seeds)]
    print(f"Running {n_seeds} seeds with ball-position perturbation...")
    rs = []
    for s in seeds:
        r = simulate(seed=s, render_video=False, ball_perturb=1.0)
        print(f"  seed={s:6d}  held={r.cup_held_final}  "
              f"big={r.big_count}  small={r.small_count}  miss={r.miss_count}  "
              f"score={r.score}")
        rs.append(r)
    scores = [r.score for r in rs]
    bigs = [r.big_count for r in rs]
    smalls = [r.small_count for r in rs]
    misses = [r.miss_count for r in rs]
    held = sum(1 for r in rs if r.cup_held_final)
    stats = {
        "n_seeds": n_seeds,
        "duration_per_seed_s": DURATION_S,
        "n_balls": N_BALLS,
        "score_formula": "1 * big + 2 * small",
        "cup_held_count": held,
        "score": {
            "mean": round(float(np.mean(scores)), 2),
            "min": int(np.min(scores)),
            "max": int(np.max(scores)),
        },
        "big_count":   {"mean": round(float(np.mean(bigs)), 2),
                        "min": int(np.min(bigs)), "max": int(np.max(bigs))},
        "small_count": {"mean": round(float(np.mean(smalls)), 2),
                        "min": int(np.min(smalls)), "max": int(np.max(smalls))},
        "miss_count":  {"mean": round(float(np.mean(misses)), 2),
                        "min": int(np.min(misses)), "max": int(np.max(misses))},
        "per_seed": [
            {"seed": r.seed, "held": r.cup_held_final,
             "big": r.big_count, "small": r.small_count,
             "miss": r.miss_count, "score": r.score}
            for r in rs
        ],
    }
    OUT_MULTI_SEED.parent.mkdir(parents=True, exist_ok=True)
    OUT_MULTI_SEED.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print()
    print(f"=== Multi-seed results ({n_seeds} runs) ===")
    print(f"  cup held: {held}/{n_seeds}")
    print(f"  score: mean={stats['score']['mean']}  "
          f"min={stats['score']['min']}  max={stats['score']['max']}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi-seed", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    if args.multi_seed:
        run_multi_seed(args.n)
    else:
        print(f"Single run, seed={args.seed}")
        r = simulate(seed=args.seed, render_video=True)
        print(f"  held={r.cup_held_final}  big={r.big_count}  "
              f"small={r.small_count}  miss={r.miss_count}  score={r.score}")


if __name__ == "__main__":
    main()
