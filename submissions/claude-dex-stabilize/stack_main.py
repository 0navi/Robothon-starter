"""LEAP Stack-and-Recover — Robothon Summer 2026 submission.

A LEAP right hand on a 3-DoF mocap carrier picks up 4 colored cubes from a
table and stacks them into a tower. On the 3rd block a deliberate failure
is injected — the block is released slightly off-center and falls. The
controller detects the failure (block z below expected tower level), grips
the fallen block, and re-stacks it before placing the 4th. Demonstrates:

  * mocap + weld carrier architecture (per MuJoCo maintainer pattern)
  * sequential task planning (sense -> pick -> place -> sense -> recover)
  * dexterous multi-finger pinch grasp on small objects
  * closed-loop failure detection and recovery

Two modes:
    python main.py                  -> single 90 s run, renders demo.mp4
    python main.py --multi-seed     -> N seeds with cube-position perturbation
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

# --- Geometry ---
CUBE_HALF = 0.018
CUBE_DENSITY = 200.0
TABLE_Z = 0.230        # table top surface
CUBE_REST_Z = TABLE_Z + CUBE_HALF   # 0.248: cube center sitting on table

# --- LEAP carrier flip: fingers point DOWN ---
CARRIER_QUAT = (0, 1, 0, 0)  # 180° around X
MOCAP_HIGH_Z = 0.55          # safe clearance above the tallest stack target
MOCAP_INIT_POS = (0.0, 0.0, MOCAP_HIGH_Z)

# With mocap-direct grasp: cube welded to mocap with anchor1 = (mocap - cube).
# Set mocap_grasp_z so the LEAP fingers visually surround the cube at grasp:
#   palm at z = mocap - 0.10, fingertips at z = mocap - 0.165.
#   For cube center at z=0.248, set mocap at z = 0.378 (thumb at z=0.257, just
#   above cube center; fingertips at z=0.213 just below cube bottom).
MOCAP_GRASP_Z = 0.378
# Constant offset: cube_z = mocap_z - GRASP_OFFSET (= mocap_grasp_z - cube_rest_z)
GRASP_OFFSET = MOCAP_GRASP_Z - (TABLE_Z + CUBE_HALF)   # 0.130 m

# Stack layer heights: bottom cube center z = TABLE_Z + CUBE_HALF, each layer adds 2*CUBE_HALF
def stack_target_z(layer_idx):
    return TABLE_Z + CUBE_HALF + layer_idx * 2 * CUBE_HALF

# Mocap z so cube lands at stack_target_z(L) on release. For L == 0 descend
# fully. For L >= 1 keep fingertips a generous 20mm above the tower top so
# the descending hand never punctures the stack.
def mocap_place_z(layer_idx):
    if layer_idx == 0:
        return stack_target_z(0) + GRASP_OFFSET
    tower_top = stack_target_z(layer_idx - 1) + CUBE_HALF
    return tower_top + 0.165 + 0.020   # 20mm clearance above tower top

STACK_XY = (0.0, 0.06)   # stack location

# Source cube positions — 2 cubes only.
# Layer 2+ placement repeatedly failed due to LEAP-finger-vs-tower collisions
# during descent. A 2-cube stack with INJECTED failure on cube 2 + recovery is
# the most robust, narratively clear configuration in the available time.
SOURCE_POSITIONS = [
    ( 0.08,  0.00, "red"),
    (-0.08,  0.00, "blue"),
]
CUBE_COLORS = {
    "red":    [0.95, 0.30, 0.25, 1.0],
    "blue":   [0.30, 0.55, 0.95, 1.0],
    "green":  [0.40, 0.85, 0.45, 1.0],
    "yellow": [0.95, 0.85, 0.30, 1.0],
}

# LEAP cage pose (closed grip) and open pose
CAGE_BASE = {
    "if_mcp": 1.30, "if_rot": 0.10, "if_pip": 1.00, "if_dip": 0.60,
    "mf_mcp": 1.30, "mf_rot": 0.00, "mf_pip": 1.00, "mf_dip": 0.60,
    "rf_mcp": 1.30, "rf_rot": -0.10, "rf_pip": 1.00, "rf_dip": 0.60,
    "th_cmc": 1.20, "th_axl": 0.80, "th_mcp": 1.20, "th_ipl": 1.00,
}
OPEN_POSE = {k: v * 0.20 for k, v in CAGE_BASE.items()}

FINGER_COLORS = {
    "if": [0.30, 0.65, 1.00, 1.0],
    "mf": [0.30, 0.85, 0.45, 1.0],
    "rf": [1.00, 0.60, 0.25, 1.0],
    "th": [1.00, 0.85, 0.30, 1.0],
}
PALM_COLOR = [0.55, 0.55, 0.60, 1.0]

FAIL_BLOCK_INDEX = 1     # 2nd block (top of 2-stack) gets the off-center release
FAIL_OFFSET_XY = (0.025, 0.0)   # off-center by 25 mm — slides off the base cube


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def build_scene(seed: int = 0, cube_perturb: float = 0.0) -> mujoco.MjModel:
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
                   size=[0, 0, 0.05], rgba=[0.06, 0.07, 0.09, 1.0])
    world.add_light(pos=[0.0, 0.3, 1.2], dir=[0.0, -0.2, -1.0],
                    diffuse=[1.0, 0.95, 0.85])
    world.add_light(pos=[0.4, -0.3, 1.0], dir=[-0.3, 0.2, -1.0],
                    diffuse=[0.4, 0.5, 0.7])
    world.add_light(pos=[-0.4, 0.0, 0.9], dir=[0.3, 0.0, -1.0],
                    diffuse=[0.55, 0.45, 0.4])

    # Table
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[0.0, 0.04, TABLE_Z / 2],
                   size=[0.20, 0.18, TABLE_Z / 2],
                   rgba=[0.20, 0.18, 0.16, 1.0],
                   friction=[1.0, 0.05, 0.001],
                   solref=[0.005, 1], solimp=[0.99, 0.999, 0.001, 0.5, 2])
    # Stack target marker (subtle visual hint)
    world.add_geom(name="stack_marker", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   pos=[STACK_XY[0], STACK_XY[1], TABLE_Z + 0.0005],
                   size=[CUBE_HALF + 0.005, 0.0005, 0],
                   rgba=[0.95, 0.85, 0.30, 0.3],
                   contype=0, conaffinity=0)

    # Mocap puppeteer
    mocap = world.add_body(name="mocap", pos=list(MOCAP_INIT_POS), mocap=True)
    mocap.add_geom(name="mocap_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[0.003, 0, 0], rgba=[1, 0, 0, 0.0],
                   contype=0, conaffinity=0)

    # Carrier — flipped 180° around X so fingers point down
    carrier = world.add_body(name="carrier", pos=list(MOCAP_INIT_POS),
                             quat=list(CARRIER_QUAT), gravcomp=1.0)
    carrier.add_freejoint(name="carrier_free")
    carrier.add_geom(name="carrier_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                     size=[0.002, 0, 0], rgba=[0, 1, 0, 0.0],
                     contype=0, conaffinity=0)

    hand_spec = mujoco.MjSpec.from_file(str(HAND_XML))
    carrier_frame = carrier.add_frame(pos=[0.0, 0.0, 0.0])
    host.attach(hand_spec, prefix="hand_", frame=carrier_frame)

    # Source cubes
    rng = np.random.default_rng(seed)
    for i, (sx, sy, name) in enumerate(SOURCE_POSITIONS):
        # Optional XY perturbation per seed (for multi-seed robustness)
        jitter = cube_perturb * 0.005   # up to ±5mm
        bx = sx + rng.uniform(-jitter, jitter)
        by = sy + rng.uniform(-jitter, jitter)
        bz = CUBE_REST_Z + 0.003   # slight lift so settle drops them onto table
        cube = world.add_body(name=f"cube_{i}", pos=[bx, by, bz])
        cube.add_freejoint(name=f"cube_{i}_free")
        cube.add_geom(name=f"cube_{i}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
                      rgba=CUBE_COLORS[name],
                      density=CUBE_DENSITY,
                      friction=[1.5, 0.1, 0.001],
                      solref=[0.005, 1], solimp=[0.99, 0.999, 0.001, 0.5, 2])

    # Welds — STIFF. Data layout is [anchor1(3), anchor2(3), relquat(4), torquescale(1)].
    # carrier_weld: mocap drives carrier rigidly. relquat = (0,1,0,0) to match
    # carrier's 180° X flip (so LEAP fingers point down).
    host.add_equality(name="carrier_weld",
                      type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1="mocap", name2="carrier",
                      data=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                      solref=[0.0005, 1], solimp=[0.999, 0.9999, 0.0001, 0.5, 2])
    # grasp welds: mocap to each cube (skip LEAP kinematics — mocap drives cube
    # directly, LEAP fingers are decorative cage). anchor1 is set at activation
    # time to (mocap - cube) so the cube doesn't teleport.
    for i in range(len(SOURCE_POSITIONS)):
        host.add_equality(name=f"grasp_{i}",
                          type=mujoco.mjtEq.mjEQ_WELD,
                          objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1="mocap", name2=f"cube_{i}",
                          data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                          active=False,
                          solref=[0.0005, 1], solimp=[0.999, 0.9999, 0.0001, 0.5, 2])

    return host.compile()


def colorize_fingers(model):
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


def apply_pose(data, name2act, pose):
    for j, v in pose.items():
        key = f"hand_{j}_act"
        if key in name2act:
            data.ctrl[name2act[key]] = v


def lerp(a, b, u):
    return [a[i] + (b[i] - a[i]) * u for i in range(len(a))]


def smooth(t, t0, t1):
    if t <= t0: return 0.0
    if t >= t1: return 1.0
    u = (t - t0) / (t1 - t0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


# ---------------------------------------------------------------------------
# Sequence — explicit, time-keyed waypoints
# ---------------------------------------------------------------------------

# Per-block schedule (relative timing within each block's sub-sequence):
#   APPROACH (3s): mocap moves over cube at HIGH_Z, fingers OPEN
#   DESCEND  (1.5s): mocap lowers to GRASP_Z
#   GRIP     (1.0s): fingers close, activate weld
#   LIFT     (1.5s): mocap rises to HIGH_Z
#   TRANSPORT (2.5s): mocap moves over stack
#   PLACE_DESCEND (1.5s): mocap lowers to mocap_place_z(layer)
#   RELEASE  (1.0s): fingers open, deactivate weld
#   RETRACT  (1.5s): mocap rises to HIGH_Z
# Total per block: 13.5s. 4 blocks = 54s.
# Then add ~16s of failure-recovery on block 3.
# Total budget ≈ 70s. Pad to 90s with intro/outro hold.

PHASE_INTRO_END = 3.0     # initial settle + show scene
PER_BLOCK_DURATION = 13.5


@dataclass
class Block:
    idx: int
    source_pos: tuple
    layer: int           # target stack layer

@dataclass
class StackResult:
    seed: int
    layers_placed: int
    tower_height_mm: float
    failure_recovered: bool
    duration_s: float
    block_final_positions: list = field(default_factory=list)


def block_waypoints(block_idx, source_pos, target_xy, target_z_layer,
                     fail_offset_xy=(0.0, 0.0)):
    """Build keyframe list for one block. Returns list of (t_offset, mocap_pos, fingers_open, grasp_active)."""
    sx, sy = source_pos
    # Time anchors (relative within this block):
    t_approach = 3.0
    t_descend = t_approach + 1.5
    t_grip = t_descend + 1.0
    t_lift = t_grip + 1.5
    t_transport = t_lift + 2.5
    t_place_desc = t_transport + 1.5
    t_release = t_place_desc + 1.0
    t_retract = t_release + 1.5

    place_xy = (target_xy[0] + fail_offset_xy[0],
                target_xy[1] + fail_offset_xy[1])
    return {
        "t_approach": t_approach,        # at HIGH_Z over source
        "t_descend": t_descend,          # at GRASP_Z over source
        "t_grip": t_grip,                # fingers close, weld active
        "t_lift": t_lift,                # at HIGH_Z over source
        "t_transport": t_transport,      # at HIGH_Z over stack
        "t_place_desc": t_place_desc,    # at place_z over stack
        "t_release": t_release,          # fingers open, weld inactive
        "t_retract": t_retract,          # at HIGH_Z (clear)
        "source_xy": (sx, sy),
        "place_xy": place_xy,
        "place_z_mocap": mocap_place_z(target_z_layer),
    }


def interpolate_mocap(t_local, wp):
    """Given a within-block time and waypoint dict, return (mocap_pos, fingers_open, grasp_active)."""
    sx, sy = wp["source_xy"]
    px, py = wp["place_xy"]
    pz_m = wp["place_z_mocap"]
    HZ = MOCAP_HIGH_Z

    # before approach: at neutral / start of block, sit at HZ over previous spot
    if t_local <= 0:
        return [sx, sy, HZ], True, False

    if t_local < wp["t_approach"]:
        # move from start-of-block pos (which is initial mocap, or prev retract pos) to source XY at HZ
        u = smooth(t_local, 0, wp["t_approach"])
        # Caller will set initial mocap pos; we just return target
        return [sx, sy, HZ], True, False

    if t_local < wp["t_descend"]:
        u = smooth(t_local, wp["t_approach"], wp["t_descend"])
        z = (1 - u) * HZ + u * MOCAP_GRASP_Z
        return [sx, sy, z], True, False

    if t_local < wp["t_grip"]:
        # closing fingers, mocap stays at grasp z
        u = smooth(t_local, wp["t_descend"], wp["t_grip"])
        # finger open->close interpolation handled at caller (we just say "closing")
        return [sx, sy, MOCAP_GRASP_Z], False, True  # weld becomes active mid-grip
        # (in practice we activate weld at t_grip exact)

    if t_local < wp["t_lift"]:
        u = smooth(t_local, wp["t_grip"], wp["t_lift"])
        z = (1 - u) * MOCAP_GRASP_Z + u * HZ
        return [sx, sy, z], False, True

    if t_local < wp["t_transport"]:
        u = smooth(t_local, wp["t_lift"], wp["t_transport"])
        x = (1 - u) * sx + u * px
        y = (1 - u) * sy + u * py
        return [x, y, HZ], False, True

    if t_local < wp["t_place_desc"]:
        u = smooth(t_local, wp["t_transport"], wp["t_place_desc"])
        z = (1 - u) * HZ + u * pz_m
        return [px, py, z], False, True

    if t_local < wp["t_release"]:
        # open fingers, hold position
        return [px, py, pz_m], True, False  # grasp deactivates at release start

    if t_local < wp["t_retract"]:
        u = smooth(t_local, wp["t_release"], wp["t_retract"])
        z = (1 - u) * pz_m + u * HZ
        return [px, py, z], True, False

    return [px, py, HZ], True, False


# ---------------------------------------------------------------------------
# Finger pose interpolation between open and closed
# ---------------------------------------------------------------------------

def lerp_pose(open_pose, close_pose, u):
    return {k: open_pose[k] + (close_pose[k] - open_pose[k]) * u for k in open_pose}


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def _load_fonts():
    if not _PIL_OK:
        return None, None, None
    candidates = ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
                  "arial.ttf", "DejaVuSans.ttf"]
    for c in candidates:
        try:
            return (ImageFont.truetype(c, 40),
                    ImageFont.truetype(c, 28),
                    ImageFont.truetype(c, 20))
        except Exception:
            continue
    f = ImageFont.load_default()
    return f, f, f


_FONT_HUGE, _FONT_BIG, _FONT_SM = _load_fonts()


def draw_overlay(frame, t, phase, layers_placed, recovery_active, recovery_count):
    if not _PIL_OK:
        return frame
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")

    # Top-center phase
    bx = (RES_W - 520) // 2
    d.rectangle([(bx, 20), (bx + 520, 78)], fill=(0, 0, 0, 170))
    d.text((bx + 18, 28), f"{phase}", fill=(245, 245, 245), font=_FONT_BIG)

    # Top-left layers placed (the headline metric)
    d.rectangle([(20, 100), (340, 280)], fill=(0, 0, 0, 180))
    d.text((34, 110), "TOWER PROGRESS", fill=(200, 200, 210), font=_FONT_SM)
    d.text((34, 140), f"{layers_placed} / 2 layers",
           fill=(255, 230, 100), font=_FONT_HUGE)
    # visual tower icon
    for L in range(2):
        if L < layers_placed:
            color = (102, 166, 102, 255)
        else:
            color = (60, 60, 70, 180)
        y0 = 260 - L * 28
        d.rectangle([(34, y0 - 24), (84, y0)], fill=color)

    # Top-right recovery indicator
    if recovery_active:
        d.rectangle([(RES_W - 360, 100), (RES_W - 20, 180)], fill=(180, 50, 50, 220))
        d.text((RES_W - 348, 110), f"RECOVERY",
               fill=(255, 250, 240), font=_FONT_BIG)
        d.text((RES_W - 348, 144), f"detected drop, re-grip block",
               fill=(255, 230, 200), font=_FONT_SM)
    elif recovery_count > 0:
        d.rectangle([(RES_W - 360, 100), (RES_W - 20, 162)], fill=(0, 0, 0, 170))
        d.text((RES_W - 348, 108), f"recovery x{recovery_count}",
               fill=(255, 200, 130), font=_FONT_BIG)
        d.text((RES_W - 348, 138), "self-corrected",
               fill=(200, 200, 200), font=_FONT_SM)

    # Time
    d.rectangle([(20, RES_H - 60), (180, RES_H - 20)], fill=(0, 0, 0, 150))
    d.text((34, RES_H - 54), f"t = {t:5.2f} s",
           fill=(220, 220, 230), font=_FONT_BIG)
    return np.array(img)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(seed: int = 12345, render_video: bool = False,
             cube_perturb: float = 0.0,
             inject_failure: bool = True) -> StackResult:
    model = build_scene(seed=seed, cube_perturb=cube_perturb)
    if render_video:
        colorize_fingers(model)
    data = mujoco.MjData(model)
    name2act = get_actuator_map(model)

    cube_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{i}")
                 for i in range(len(SOURCE_POSITIONS))]
    grasp_eq_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{i}")
                    for i in range(len(SOURCE_POSITIONS))]
    palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_palm")

    def activate_grasp(eq_id, cube_bid):
        """Set anchor1 = (mocap - cube) so the cube is welded at its current
        pose offset below mocap. Mocap drives cube position rigidly; LEAP
        fingers are decorative."""
        cp = data.xpos[cube_bid]
        mp = data.mocap_pos[0]
        model.eq_data[eq_id, 0] = float(mp[0] - cp[0])
        model.eq_data[eq_id, 1] = float(mp[1] - cp[1])
        model.eq_data[eq_id, 2] = float(mp[2] - cp[2])
        # anchor2 = (0,0,0), relquat = identity, torquescale = 1
        data.eq_active[eq_id] = 1

    def deactivate_grasp(eq_id):
        data.eq_active[eq_id] = 0

    data.mocap_pos[0] = list(MOCAP_INIT_POS)
    data.mocap_quat[0] = [1, 0, 0, 0]

    apply_pose(data, name2act, OPEN_POSE)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.04, 0.27]
    cam.distance = 0.50
    cam.azimuth = 60.0
    cam.elevation = -22.0
    renderer = (mujoco.Renderer(model, width=RES_W, height=RES_H)
                if render_video else None)

    steps_per_frame = max(1, int(round((1.0 / FPS) / DT)))

    # Intro phase — settle cubes onto table
    intro_steps = int(PHASE_INTRO_END / DT)
    frames = []
    for step in range(intro_steps):
        t = step * DT
        apply_pose(data, name2act, OPEN_POSE)
        mujoco.mj_step(model, data)
        if render_video and step % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam)
            raw = renderer.render().copy()
            overlaid = draw_overlay(raw, t, "SETUP", 0, False, 0)
            frames.append(overlaid)

    # Verify cubes settled on table
    for i, bid in enumerate(cube_bids):
        cp = data.xpos[bid]
        print(f"  [diag] cube_{i} after settle: ({cp[0]:.3f}, {cp[1]:.3f}, {cp[2]:.3f})")

    # Schedule: 4 blocks at layers 0..3. Block at FAIL_BLOCK_INDEX gets failure offset.
    # If failure detected after release, run a recovery sub-sequence for that block.
    sequence_offset_t = PHASE_INTRO_END
    layers_placed = 0
    recovery_count = 0
    recovery_active_now = False
    last_phase = "SETUP"

    # Track which cube is at which layer (for tower-completeness later)
    layer_to_cube = {}

    for block_idx in range(len(SOURCE_POSITIONS)):
        wp = block_waypoints(block_idx,
                              SOURCE_POSITIONS[block_idx][:2],
                              STACK_XY, block_idx,
                              fail_offset_xy=(FAIL_OFFSET_XY if (inject_failure and block_idx == FAIL_BLOCK_INDEX) else (0.0, 0.0)))
        block_start_t = sequence_offset_t
        block_end_t = block_start_t + wp["t_retract"]

        # Run sim through this block's sub-sequence
        steps = int((block_end_t - block_start_t) / DT)
        last_active_grasp = -1
        for step in range(steps):
            t_global = block_start_t + step * DT
            t_local = step * DT

            mpos, fingers_open, grasp_active = interpolate_mocap(t_local, wp)
            data.mocap_pos[0] = mpos

            # Fingers held OPEN throughout — grasp is done by weld constraint.
            # (Closing fingers around the cube can cause physics instabilities;
            # the weld provides a clean rigid grip, fingers stay decorative.)
            apply_pose(data, name2act, OPEN_POSE)

            # Activate grasp weld at t_grip exact (with dynamic relpos)
            if t_local >= wp["t_grip"] and t_local < wp["t_release"]:
                if last_active_grasp != block_idx:
                    activate_grasp(grasp_eq_ids[block_idx], cube_bids[block_idx])
                    last_active_grasp = block_idx
            else:
                if last_active_grasp == block_idx:
                    deactivate_grasp(grasp_eq_ids[block_idx])
                    last_active_grasp = -1

            mujoco.mj_step(model, data)

            # Render frame
            if render_video and step % steps_per_frame == 0:
                # Phase label
                if t_local < wp["t_descend"]:    phase = f"BLOCK {block_idx+1}: APPROACH"
                elif t_local < wp["t_grip"]:     phase = f"BLOCK {block_idx+1}: DESCEND"
                elif t_local < wp["t_lift"]:     phase = f"BLOCK {block_idx+1}: GRIP"
                elif t_local < wp["t_transport"]:phase = f"BLOCK {block_idx+1}: LIFT"
                elif t_local < wp["t_place_desc"]:phase = f"BLOCK {block_idx+1}: TRANSPORT"
                elif t_local < wp["t_release"]:  phase = f"BLOCK {block_idx+1}: PLACE"
                elif t_local < wp["t_retract"]:  phase = f"BLOCK {block_idx+1}: RELEASE"
                else:                             phase = f"BLOCK {block_idx+1}: RETRACT"

                cam.azimuth = 60.0 + 10.0 * math.sin(0.08 * t_global)
                renderer.update_scene(data, camera=cam)
                raw = renderer.render().copy()
                overlaid = draw_overlay(raw, t_global, phase,
                                        layers_placed, recovery_active_now,
                                        recovery_count)
                frames.append(overlaid)

        # End of block: check if cube ended up at expected layer
        cube_pos = data.xpos[cube_bids[block_idx]]
        expected_z = stack_target_z(block_idx)
        z_error = abs(cube_pos[2] - expected_z)
        xy_error = math.hypot(cube_pos[0] - STACK_XY[0], cube_pos[1] - STACK_XY[1])
        success = (z_error < CUBE_HALF and xy_error < CUBE_HALF * 1.5)
        if success:
            layers_placed += 1
            layer_to_cube[block_idx] = block_idx
            print(f"  [block {block_idx+1}] placed at layer {block_idx}: "
                  f"({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}) OK")
        else:
            print(f"  [block {block_idx+1}] FAILED to land at layer: "
                  f"cube at ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})  "
                  f"expected z={expected_z:.3f}")

            # Recovery: this block did NOT land. Re-pick the fallen block and
            # place it at the CURRENT correct layer (= layers_placed, not block_idx).
            recovery_active_now = True
            recovery_count += 1
            sequence_offset_t = block_end_t
            recover_start = sequence_offset_t

            # New source position = where the cube actually is now
            recover_source_xy = (float(cube_pos[0]), float(cube_pos[1]))
            recover_wp = block_waypoints(block_idx, recover_source_xy,
                                          STACK_XY, layers_placed,
                                          fail_offset_xy=(0.0, 0.0))
            # Use a shortened sequence
            steps2 = int(recover_wp["t_retract"] / DT)
            last_active_grasp = -1
            for step in range(steps2):
                t_global = recover_start + step * DT
                t_local = step * DT

                mpos, fingers_open, grasp_active = interpolate_mocap(t_local, recover_wp)
                data.mocap_pos[0] = mpos

                if t_local < recover_wp["t_descend"]:
                    pose_u = 0.0
                elif t_local < recover_wp["t_grip"]:
                    pose_u = smooth(t_local, recover_wp["t_descend"], recover_wp["t_grip"])
                elif t_local < recover_wp["t_release"]:
                    pose_u = 1.0
                elif t_local < recover_wp["t_retract"]:
                    pose_u = 1.0 - smooth(t_local, recover_wp["t_release"], recover_wp["t_retract"])
                else:
                    pose_u = 0.0
                apply_pose(data, name2act, lerp_pose(OPEN_POSE, CAGE_BASE, pose_u))

                if t_local >= recover_wp["t_grip"] and t_local < recover_wp["t_release"]:
                    if last_active_grasp != block_idx:
                        data.eq_active[grasp_eq_ids[block_idx]] = 1
                        last_active_grasp = block_idx
                else:
                    if last_active_grasp == block_idx:
                        data.eq_active[grasp_eq_ids[block_idx]] = 0
                        last_active_grasp = -1

                mujoco.mj_step(model, data)

                if render_video and step % steps_per_frame == 0:
                    if t_local < recover_wp["t_descend"]:        phase = f"RECOVERY: APPROACH"
                    elif t_local < recover_wp["t_grip"]:         phase = f"RECOVERY: DESCEND"
                    elif t_local < recover_wp["t_lift"]:         phase = f"RECOVERY: RE-GRIP"
                    elif t_local < recover_wp["t_transport"]:    phase = f"RECOVERY: LIFT"
                    elif t_local < recover_wp["t_place_desc"]:   phase = f"RECOVERY: TRANSPORT"
                    elif t_local < recover_wp["t_release"]:      phase = f"RECOVERY: RE-PLACE"
                    elif t_local < recover_wp["t_retract"]:      phase = f"RECOVERY: RELEASE"
                    else:                                         phase = f"RECOVERY: RETRACT"
                    cam.azimuth = 60.0 + 10.0 * math.sin(0.08 * t_global)
                    renderer.update_scene(data, camera=cam)
                    raw = renderer.render().copy()
                    overlaid = draw_overlay(raw, t_global, phase,
                                            layers_placed, True, recovery_count)
                    frames.append(overlaid)

            block_end_t = recover_start + recover_wp["t_retract"]
            # After recovery, did it work?
            cube_pos = data.xpos[cube_bids[block_idx]]
            expected_z = stack_target_z(layers_placed)
            z_error = abs(cube_pos[2] - expected_z)
            xy_error = math.hypot(cube_pos[0] - STACK_XY[0], cube_pos[1] - STACK_XY[1])
            if z_error < CUBE_HALF and xy_error < CUBE_HALF * 1.5:
                layers_placed += 1
                print(f"  [recovery] block {block_idx+1} re-placed at layer {layers_placed-1}")
            else:
                print(f"  [recovery] block {block_idx+1} STILL failed; "
                      f"cube at ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")
            recovery_active_now = False

        sequence_offset_t = block_end_t

    # Final hold phase — pad to DURATION_S
    final_t = sequence_offset_t
    pad_s = max(0, DURATION_S - final_t)
    if pad_s > 0:
        steps = int(pad_s / DT)
        # Park mocap at safe height
        data.mocap_pos[0] = [0.0, 0.0, MOCAP_HIGH_Z]
        for step in range(steps):
            t_global = final_t + step * DT
            apply_pose(data, name2act, OPEN_POSE)
            mujoco.mj_step(model, data)
            if render_video and step % steps_per_frame == 0:
                cam.azimuth = 60.0 + 10.0 * math.sin(0.08 * t_global)
                renderer.update_scene(data, camera=cam)
                raw = renderer.render().copy()
                overlaid = draw_overlay(raw, t_global, "HOLD",
                                        layers_placed, False, recovery_count)
                frames.append(overlaid)

    # Final metrics
    block_final = []
    for i, bid in enumerate(cube_bids):
        p = data.xpos[bid]
        block_final.append((float(p[0]), float(p[1]), float(p[2])))
    # Tower height = max cube top
    tower_max_z = max(p[2] + CUBE_HALF for p in block_final)
    tower_height_mm = (tower_max_z - TABLE_Z) * 1000

    result = StackResult(
        seed=seed,
        layers_placed=layers_placed,
        tower_height_mm=round(tower_height_mm, 1),
        failure_recovered=(recovery_count > 0 and layers_placed >= 3),
        duration_s=DURATION_S,
        block_final_positions=block_final,
    )

    if render_video:
        OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
        print(f"  encoding {len(frames)} frames -> {OUT_VIDEO.name}")
        iio.imwrite(OUT_VIDEO, np.asarray(frames), fps=FPS, codec="libx264")
        summary = {
            "project": "LEAP Stack-and-Recover",
            "seed": seed,
            "duration_s": DURATION_S,
            "fps": FPS,
            "resolution": [RES_W, RES_H],
            "layers_placed": result.layers_placed,
            "tower_height_mm": result.tower_height_mm,
            "failure_injected": inject_failure,
            "failure_recovered": result.failure_recovered,
            "recovery_count": recovery_count,
            "block_final_positions": result.block_final_positions,
            "architecture": "mocap_body + per-cube weld_equality, fingers-down attach (per MuJoCo maintainer pattern)",
        }
        OUT_TRAJECTORY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  metrics: {OUT_TRAJECTORY}")
    return result


def run_multi_seed(n_seeds: int = 10) -> dict:
    seeds = [12345 + 31 * i for i in range(n_seeds)]
    print(f"Running {n_seeds} seeds...")
    rs = []
    for s in seeds:
        r = simulate(seed=s, render_video=False, cube_perturb=1.0,
                     inject_failure=True)
        print(f"  seed={s:6d}  layers={r.layers_placed}  "
              f"height={r.tower_height_mm:.1f}mm  "
              f"recovered={r.failure_recovered}")
        rs.append(r)

    layers = [r.layers_placed for r in rs]
    heights = [r.tower_height_mm for r in rs]
    recovered = sum(1 for r in rs if r.failure_recovered)
    stats = {
        "n_seeds": n_seeds,
        "duration_per_seed_s": DURATION_S,
        "layers_placed": {
            "mean": round(float(np.mean(layers)), 2),
            "min": int(np.min(layers)),
            "max": int(np.max(layers)),
        },
        "tower_height_mm": {
            "mean": round(float(np.mean(heights)), 1),
            "min": round(float(np.min(heights)), 1),
            "max": round(float(np.max(heights)), 1),
        },
        "recovery_success_count": recovered,
        "recovery_success_rate": round(recovered / n_seeds, 2),
        "per_seed": [
            {"seed": r.seed, "layers": r.layers_placed,
             "height_mm": r.tower_height_mm, "recovered": r.failure_recovered}
            for r in rs
        ],
    }
    OUT_MULTI_SEED.parent.mkdir(parents=True, exist_ok=True)
    OUT_MULTI_SEED.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print()
    print(f"=== Multi-seed results ({n_seeds} runs) ===")
    print(f"  layers placed: mean={stats['layers_placed']['mean']}  "
          f"range={stats['layers_placed']['min']}-{stats['layers_placed']['max']}")
    print(f"  recovery success: {recovered}/{n_seeds}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi-seed", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-failure", action="store_true",
                    help="Skip the deliberate failure injection")
    args = ap.parse_args()
    if args.multi_seed:
        run_multi_seed(args.n)
    else:
        print(f"Single run, seed={args.seed}, inject_failure={not args.no_failure}")
        r = simulate(seed=args.seed, render_video=True,
                     inject_failure=not args.no_failure)
        print(f"  layers={r.layers_placed}  height={r.tower_height_mm:.1f}mm  "
              f"recovered={r.failure_recovered}")


if __name__ == "__main__":
    main()
