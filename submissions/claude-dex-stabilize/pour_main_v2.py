"""LEAP Pour Task v2 — Robothon Summer 2026 submission.

A LEAP right hand grips a small open-top cup containing 8 colored balls and
pours them into target containers. Uses the mocap + weld architecture
recommended by MuJoCo maintainers for stable rigid-body manipulation:

  * mocap body drives the carrier (LEAP root) via a stiff weld — no PD spring
    chain that would store and release energy as ball-launching kicks.
  * weld equality between the cup and the LEAP thumb tip, toggled on after
    the fingers close, gives a rigid magnetic grasp (the same pattern used
    in the LEAP Hand "Pour Tea" paper).
  * Contact damping (solref second arg = 1) on cup and ball geoms prevents
    elastic energy buildup in the contact springs.

Run modes:
    python main.py
        -> single deterministic 90 s pour, renders demo.mp4 + JSON.

    python main.py --multi-seed
        -> N=10 seeds with ball-position perturbation, no video.
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

DT = 0.001
FPS = 30
RES_W, RES_H = 1280, 720
DURATION_S = 90.0

# Cup
CUP_OUTER_HALF = 0.025
CUP_WALL_T = 0.0015
CUP_HEIGHT_HALF = 0.015
CUP_DENSITY = 350.0
CUP_INIT_POS = (-0.022, 0.020, 0.346)

# Balls
BALL_R = 0.003
N_BALLS = 8

# Containers — distinct heights so balls clearly settle
BIG_POS = (-0.16, 0.020, 0.0)
BIG_HALF = 0.045
BIG_DEPTH_HALF = 0.025

SMALL_POS = (0.13, 0.020, 0.0)
SMALL_HALF = 0.022
SMALL_DEPTH_HALF = 0.025

# Mocap pose schedule — mocap_pos = anchor, mocap_quat = Y-axis tilt
MOCAP_INIT_POS = (0.0, 0.0, 0.20)

# Phase timeline
PHASE_SETTLE_END       = 2.0
PHASE_GRASP_END        = 2.5    # 0.5s for weld to settle the cup
PHASE_LIFT_END         = 8.0    # lift hand up
PHASE_OVER_BIG_END     = 14.0   # translate over big container
PHASE_TILT_BIG_END     = 28.0   # tilt for big pour
PHASE_RETURN_BIG_END   = 35.0   # tilt back
PHASE_TO_SMALL_END     = 45.0   # translate to small container
PHASE_OVER_SMALL_END   = 48.0
PHASE_TILT_SMALL_END   = 64.0
PHASE_RETURN_SMALL_END = 72.0
PHASE_RECOVER_END      = 80.0
PHASE_HOLD_END         = 90.0

# Tilt magnitudes (sign chosen so cup opening points toward target container)
TILT_BIG = -math.radians(115)   # tilt -Y, cup opens toward -X (big container)
TILT_SMALL = math.radians(130)  # tilt +Y, cup opens toward +X (small)

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
    [0.95, 0.30, 0.30],
    [0.95, 0.65, 0.20],
    [0.95, 0.90, 0.30],
    [0.40, 0.85, 0.45],
    [0.30, 0.70, 0.95],
    [0.45, 0.45, 0.95],
    [0.75, 0.40, 0.95],
    [0.95, 0.55, 0.85],
]


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def add_container(world, name, pos, half_w, depth_half, color):
    base = world.add_body(name=name, pos=list(pos))
    base.add_geom(name=f"{name}_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[0, 0, 0.001],
                  size=[half_w, half_w, 0.001],
                  rgba=color, friction=[1.0, 0.05, 0.001],
                  solref=[0.005, 1], solimp=[0.99, 0.999, 0.001, 0.5, 2])
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
                      rgba=color, friction=[1.0, 0.05, 0.001],
                      solref=[0.005, 1], solimp=[0.99, 0.999, 0.001, 0.5, 2])


def build_scene(seed: int = 0, ball_perturb: float = 0.0) -> mujoco.MjModel:
    host = mujoco.MjSpec()
    host.option.timestep = DT
    host.option.gravity = [0.0, 0.0, -9.81]
    host.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    host.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    host.option.impratio = 100.0
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

    add_container(world, "big", BIG_POS, BIG_HALF, BIG_DEPTH_HALF,
                  color=[0.40, 0.65, 0.40, 1.0])
    add_container(world, "small", SMALL_POS, SMALL_HALF, SMALL_DEPTH_HALF,
                  color=[0.85, 0.55, 0.30, 1.0])

    # Mocap "puppeteer"
    mocap = world.add_body(name="mocap", pos=list(MOCAP_INIT_POS), mocap=True)
    mocap.add_geom(name="mocap_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[0.003, 0, 0], rgba=[1, 0, 0, 0.0],
                   contype=0, conaffinity=0)

    # Carrier: free body with gravity compensation, driven by mocap weld
    carrier = world.add_body(name="carrier", pos=list(MOCAP_INIT_POS),
                             gravcomp=1.0)
    carrier.add_freejoint(name="carrier_free")
    carrier.add_geom(name="carrier_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                     size=[0.002, 0, 0], rgba=[0, 1, 0, 0.0],
                     contype=0, conaffinity=0)

    hand_spec = mujoco.MjSpec.from_file(str(HAND_XML))
    carrier_frame = carrier.add_frame(pos=[0.0, 0.0, 0.0])
    host.attach(hand_spec, prefix="hand_", frame=carrier_frame)

    # Cup
    cup = world.add_body(name="cup", pos=list(CUP_INIT_POS))
    cup.add_freejoint(name="cup_free")
    cup.add_geom(name="cup_bottom", type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=[0, 0, -CUP_HEIGHT_HALF + CUP_WALL_T],
                 size=[CUP_OUTER_HALF, CUP_OUTER_HALF, CUP_WALL_T],
                 rgba=[0.55, 0.55, 0.65, 1.0], density=CUP_DENSITY,
                 friction=[1.5, 0.1, 0.001],
                 solref=[0.005, 1], solimp=[0.99, 0.999, 0.001, 0.5, 2])
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
                     friction=[1.5, 0.1, 0.001],
                     solref=[0.005, 1], solimp=[0.99, 0.999, 0.001, 0.5, 2])

    # Balls
    rng = np.random.default_rng(seed)
    z_base = CUP_INIT_POS[2] - CUP_HEIGHT_HALF + CUP_WALL_T + BALL_R + 0.001
    off = BALL_R + 0.001
    grid = []
    for layer in range(2):
        for ix in [-1, 1]:
            for iy in [-1, 1]:
                grid.append((ix * off, iy * off, layer * (2 * BALL_R + 0.001)))
    for i, (dx, dy, dz) in enumerate(grid[:N_BALLS]):
        jitter = 0.0005 + ball_perturb * 0.001
        bx = CUP_INIT_POS[0] + dx + rng.uniform(-jitter, jitter)
        by = CUP_INIT_POS[1] + dy + rng.uniform(-jitter, jitter)
        bz = z_base + dz
        col = BALL_COLOR_TABLE[i % len(BALL_COLOR_TABLE)]
        ball = world.add_body(name=f"ball_{i}", pos=[bx, by, bz])
        ball.add_freejoint(name=f"ball_{i}_free")
        ball.add_geom(name=f"ball_{i}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[BALL_R, 0, 0],
                      rgba=[col[0], col[1], col[2], 1.0],
                      density=400.0,
                      friction=[1.0, 0.05, 0.001],
                      solref=[0.005, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])

    # Welds: mocap-carrier (always active), thumb-cup (toggled on after settle)
    host.add_equality(name="carrier_weld",
                      type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1="mocap", name2="carrier",
                      data=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                      solref=[0.02, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])
    host.add_equality(name="grasp_weld",
                      type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1="hand_th_ds", name2="cup",
                      data=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                      active=False,
                      solref=[0.02, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])

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


def smooth_step(t, t0, t1):
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    u = (t - t0) / (t1 - t0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def quat_y(angle_rad):
    return [math.cos(angle_rad / 2), 0, math.sin(angle_rad / 2), 0]


def mocap_pose(t: float, cup_xy_offset: tuple) -> tuple:
    """Return (mocap_pos, mocap_quat) for time t."""
    cup_dx, cup_dy = cup_xy_offset
    big_anchor_xy   = (BIG_POS[0]   - cup_dx, BIG_POS[1]   - cup_dy)
    small_anchor_xy = (SMALL_POS[0] - cup_dx, SMALL_POS[1] - cup_dy)
    init = list(MOCAP_INIT_POS)

    if t < PHASE_SETTLE_END:
        return init, quat_y(0)
    if t < PHASE_GRASP_END:
        return init, quat_y(0)
    if t < PHASE_LIFT_END:
        u = smooth_step(t, PHASE_GRASP_END, PHASE_LIFT_END)
        return ([init[0], init[1], init[2] + u * 0.06], quat_y(0))
    if t < PHASE_OVER_BIG_END:
        u = smooth_step(t, PHASE_LIFT_END, PHASE_OVER_BIG_END)
        x = (1 - u) * init[0] + u * big_anchor_xy[0]
        y = (1 - u) * init[1] + u * big_anchor_xy[1]
        return [x, y, init[2] + 0.06], quat_y(0)
    if t < PHASE_TILT_BIG_END:
        u = smooth_step(t, PHASE_OVER_BIG_END, PHASE_TILT_BIG_END)
        return ([big_anchor_xy[0], big_anchor_xy[1], init[2] + 0.06],
                quat_y(u * TILT_BIG))
    if t < PHASE_RETURN_BIG_END:
        u = smooth_step(t, PHASE_TILT_BIG_END, PHASE_RETURN_BIG_END)
        return ([big_anchor_xy[0], big_anchor_xy[1], init[2] + 0.06],
                quat_y((1 - u) * TILT_BIG))
    if t < PHASE_TO_SMALL_END:
        u = smooth_step(t, PHASE_RETURN_BIG_END, PHASE_TO_SMALL_END)
        x = (1 - u) * big_anchor_xy[0] + u * small_anchor_xy[0]
        y = (1 - u) * big_anchor_xy[1] + u * small_anchor_xy[1]
        return [x, y, init[2] + 0.06], quat_y(0)
    if t < PHASE_OVER_SMALL_END:
        return ([small_anchor_xy[0], small_anchor_xy[1], init[2] + 0.06],
                quat_y(0))
    if t < PHASE_TILT_SMALL_END:
        u = smooth_step(t, PHASE_OVER_SMALL_END, PHASE_TILT_SMALL_END)
        return ([small_anchor_xy[0], small_anchor_xy[1], init[2] + 0.06],
                quat_y(u * TILT_SMALL))
    if t < PHASE_RETURN_SMALL_END:
        u = smooth_step(t, PHASE_TILT_SMALL_END, PHASE_RETURN_SMALL_END)
        return ([small_anchor_xy[0], small_anchor_xy[1], init[2] + 0.06],
                quat_y((1 - u) * TILT_SMALL))
    if t < PHASE_RECOVER_END:
        u = smooth_step(t, PHASE_RETURN_SMALL_END, PHASE_RECOVER_END)
        x = (1 - u) * small_anchor_xy[0]
        y = (1 - u) * small_anchor_xy[1]
        return [x, y, init[2] + 0.06], quat_y(0)
    return [init[0], init[1], init[2] + 0.10], quat_y(0)


def phase_at(t: float) -> str:
    if t < PHASE_SETTLE_END:        return "SETTLE"
    if t < PHASE_GRASP_END:         return "GRASP"
    if t < PHASE_LIFT_END:          return "LIFT"
    if t < PHASE_OVER_BIG_END:      return "APPROACH BIG"
    if t < PHASE_TILT_BIG_END:      return "POUR -> BIG"
    if t < PHASE_RETURN_BIG_END:    return "TILT BACK"
    if t < PHASE_TO_SMALL_END:      return "APPROACH SMALL"
    if t < PHASE_OVER_SMALL_END:    return "OVER SMALL"
    if t < PHASE_TILT_SMALL_END:    return "POUR -> SMALL"
    if t < PHASE_RETURN_SMALL_END:  return "TILT BACK"
    if t < PHASE_RECOVER_END:       return "RECOVER"
    return "HOLD"


def ball_classifier(ball_xyz: np.ndarray) -> str:
    x, y, z = ball_xyz
    if (abs(x - BIG_POS[0]) < BIG_HALF and abs(y - BIG_POS[1]) < BIG_HALF
            and z < BIG_POS[2] + BIG_DEPTH_HALF * 2 + 0.005):
        return "big"
    if (abs(x - SMALL_POS[0]) < SMALL_HALF and abs(y - SMALL_POS[1]) < SMALL_HALF
            and z < SMALL_POS[2] + SMALL_DEPTH_HALF * 2 + 0.005):
        return "small"
    if z < 0.015:
        return "miss"
    return "cup"


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def _load_fonts():
    if not _PIL_OK:
        return None, None, None
    candidates = ["C:/Windows/Fonts/arialbd.ttf", "arial.ttf",
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
                 big_n, small_n, miss_n, cup_n, cup_held: bool) -> np.ndarray:
    if not _PIL_OK:
        return frame
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")

    # Top-center phase banner
    bx = (RES_W - 540) // 2
    d.rectangle([(bx, 20), (bx + 540, 80)], fill=(0, 0, 0, 170))
    d.text((bx + 18, 28), f"PHASE: {phase_name}",
           fill=(245, 245, 245), font=_FONT_BIG)

    # Top-right tilt readout
    d.rectangle([(RES_W - 220, 20), (RES_W - 20, 80)], fill=(0, 0, 0, 170))
    d.text((RES_W - 210, 30), f"tilt {tilt_deg:6.1f}°",
           fill=(220, 220, 240), font=_FONT_BIG)

    # Left ball counter
    d.rectangle([(20, 100), (340, 370)], fill=(0, 0, 0, 175))
    d.text((34, 110), "BALL COUNT", fill=(200, 200, 210), font=_FONT_SM)
    d.rectangle([(34, 142), (54, 162)], fill=(102, 166, 102))
    d.text((64, 138), f"big   : {big_n}",
           fill=(245, 245, 245), font=_FONT_BIG)
    d.rectangle([(34, 184), (54, 204)], fill=(217, 140, 76))
    d.text((64, 180), f"small : {small_n}  (×2)",
           fill=(245, 245, 245), font=_FONT_BIG)
    d.rectangle([(34, 226), (54, 246)], fill=(180, 70, 70))
    d.text((64, 222), f"miss  : {miss_n}",
           fill=(245, 245, 245), font=_FONT_BIG)
    d.rectangle([(34, 268), (54, 288)], fill=(140, 140, 180))
    d.text((64, 264), f"in cup: {cup_n}",
           fill=(245, 245, 245), font=_FONT_BIG)
    score = big_n + 2 * small_n
    d.text((34, 312), f"SCORE: {score}",
           fill=(255, 230, 100), font=_FONT_HUGE)

    # Time
    d.rectangle([(20, RES_H - 60), (180, RES_H - 20)], fill=(0, 0, 0, 150))
    d.text((34, RES_H - 54), f"t = {t:5.2f} s",
           fill=(220, 220, 230), font=_FONT_BIG)

    if not cup_held:
        d.rectangle([(RES_W - 240, RES_H - 60), (RES_W - 20, RES_H - 20)],
                    fill=(180, 40, 40, 220))
        d.text((RES_W - 226, RES_H - 54), "CUP LOST",
               fill=(255, 250, 240), font=_FONT_BIG)
    return np.array(img)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class PourResult:
    seed: int
    cup_held_final: bool
    big_count: int
    small_count: int
    miss_count: int
    score: int
    duration_s: float
    n_balls: int = N_BALLS
    big_balls: list = field(default_factory=list)
    small_balls: list = field(default_factory=list)
    miss_balls: list = field(default_factory=list)


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
    grasp_eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")

    data.mocap_pos[0] = list(MOCAP_INIT_POS)
    data.mocap_quat[0] = [1, 0, 0, 0]

    apply_finger_pose(data, name2act, CAGE_BASE)

    # Settle phase — grasp weld inactive, finger cage forms
    for _ in range(int(PHASE_SETTLE_END / DT)):
        apply_finger_pose(data, name2act, CAGE_BASE)
        mujoco.mj_step(model, data)

    cup_settled = data.xpos[cup_bid].copy()
    cup_xy_offset = (float(cup_settled[0] - MOCAP_INIT_POS[0]),
                     float(cup_settled[1] - MOCAP_INIT_POS[1]))
    print(f"  [diag] cup after settle: {cup_settled}, offset={cup_xy_offset}")

    # Activate grasp weld
    data.eq_active[grasp_eq_id] = 1

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.04, 0.18]
    cam.distance = 0.55
    cam.azimuth = 75.0
    cam.elevation = -18.0
    renderer = (mujoco.Renderer(model, width=RES_W, height=RES_H)
                if render_video else None)

    total_steps = int((DURATION_S - PHASE_SETTLE_END) / DT)
    steps_per_frame = max(1, int(round((1.0 / FPS) / DT)))
    frames = []
    cup_lost = False
    ball_state = ["cup"] * N_BALLS
    last_phase = ""
    cur_tilt_deg = 0.0

    for step in range(total_steps):
        t = PHASE_SETTLE_END + step * DT
        mpos, mquat = mocap_pose(t, cup_xy_offset)
        data.mocap_pos[0] = mpos
        data.mocap_quat[0] = mquat
        # extract tilt angle from quat for HUD: quat = (cos(a/2), 0, sin(a/2), 0)
        cur_tilt_deg = 2 * math.degrees(math.atan2(mquat[2], mquat[0]))
        apply_finger_pose(data, name2act, CAGE_BASE)
        mujoco.mj_step(model, data)

        cup_xyz = data.xpos[cup_bid]
        if cup_xyz[2] < 0.10:
            cup_lost = True

        for i, bid in enumerate(ball_bids):
            if ball_state[i] in ("big", "small", "miss"):
                continue
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
            cam.azimuth = 75.0 + 18.0 * math.sin(0.06 * t)
            cam.elevation = -18.0 + 3.0 * math.sin(0.04 * t)
            renderer.update_scene(data, camera=cam)
            raw = renderer.render().copy()
            overlaid = draw_overlay(
                raw, t, cur_phase, cur_tilt_deg,
                big_n, small_n, miss_n, cup_n,
                cup_held=not cup_lost,
            )
            frames.append(overlaid)

    big_n = sum(1 for s in ball_state if s == "big")
    small_n = sum(1 for s in ball_state if s == "small")
    miss_n = sum(1 for s in ball_state if s == "miss")
    score = big_n + 2 * small_n

    print(f"  final ball states: big={big_n}, small={small_n}, "
          f"miss={miss_n}, in_cup={sum(1 for s in ball_state if s == 'cup')}")
    print("  [diag] ball final positions:")
    for i, bid in enumerate(ball_bids):
        bp = data.xpos[bid]
        print(f"    ball {i} ({ball_state[i]}): "
              f"({bp[0]:+.3f}, {bp[1]:+.3f}, {bp[2]:+.3f})")

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
            "score_formula": "1*big + 2*small",
            "score": result.score,
            "big_ball_ids": result.big_balls,
            "small_ball_ids": result.small_balls,
            "miss_ball_ids": result.miss_balls,
            "architecture": "mocap_body + weld_equality (per MuJoCo maintainer pattern)",
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
              f"big={r.big_count}  small={r.small_count}  "
              f"miss={r.miss_count}  score={r.score}")
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
        "score_formula": "1*big + 2*small",
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
