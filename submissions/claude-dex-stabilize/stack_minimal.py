"""Minimal pick-and-place test — ONE cube, ONE simple cycle.

Sequence:
  0.0-2.0s   settle (cube falls onto table)
  2.0-4.0s   mocap descends from HIGH_Z to GRASP_Z over cube
  4.0-4.2s   activate weld (no finger close, just weld grasp)
  4.2-6.0s   mocap rises back to HIGH_Z (cube should lift)
  6.0-8.0s   translate mocap to a target XY
  8.0-10.0s  mocap descends to place height
  10.0-10.2s deactivate weld
  10.2-12.0s mocap rises away (cube should remain at target)

Verifies: can weld grip a cube on a table, lift it, place it elsewhere,
release it?
"""
from __future__ import annotations
import math
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
OUT_VIDEO = HERE / "outputs" / "stack_minimal.mp4"

DT = 0.001
FPS = 30
RES_W, RES_H = 1280, 720

CUBE_HALF = 0.018
CUBE_DENSITY = 200.0
TABLE_Z = 0.230

# Flipped LEAP: palm at z = mocap-0.10, fingertips at z = mocap-0.165
CARRIER_QUAT = (0, 1, 0, 0)
MOCAP_HIGH_Z = 0.45
# For thumb-tip welded to cube on table: thumb_tip at z = mocap-0.14
# Cube center at z = 0.248. mocap_z so thumb_tip at z=0.248 -> mocap = 0.388
MOCAP_GRASP_Z = 0.388

CUBE_SOURCE = (0.05, 0.02, TABLE_Z + CUBE_HALF + 0.001)
CUBE_TARGET = (-0.05, 0.02)  # place position

CAGE_BASE = {
    "if_mcp": 1.30, "if_rot": 0.10, "if_pip": 1.00, "if_dip": 0.60,
    "mf_mcp": 1.30, "mf_rot": 0.00, "mf_pip": 1.00, "mf_dip": 0.60,
    "rf_mcp": 1.30, "rf_rot": -0.10, "rf_pip": 1.00, "rf_dip": 0.60,
    "th_cmc": 1.20, "th_axl": 0.80, "th_mcp": 1.20, "th_ipl": 1.00,
}
OPEN_POSE = {k: v * 0.20 for k, v in CAGE_BASE.items()}


def smooth(t, t0, t1):
    if t <= t0: return 0.0
    if t >= t1: return 1.0
    u = (t - t0) / (t1 - t0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def lerp(a, b, u):
    return [a[i] + (b[i] - a[i]) * u for i in range(len(a))]


def build_scene():
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

    # Table
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[0.0, 0.04, TABLE_Z / 2],
                   size=[0.18, 0.16, TABLE_Z / 2],
                   rgba=[0.20, 0.18, 0.16, 1.0],
                   friction=[1.0, 0.05, 0.001],
                   solref=[0.005, 1], solimp=[0.99, 0.999, 0.001, 0.5, 2])
    # Source marker (green)
    world.add_geom(name="src_marker", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   pos=[CUBE_SOURCE[0], CUBE_SOURCE[1], TABLE_Z + 0.0005],
                   size=[CUBE_HALF + 0.005, 0.0005, 0],
                   rgba=[0.4, 0.85, 0.4, 0.5],
                   contype=0, conaffinity=0)
    # Target marker (gold)
    world.add_geom(name="tgt_marker", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   pos=[CUBE_TARGET[0], CUBE_TARGET[1], TABLE_Z + 0.0005],
                   size=[CUBE_HALF + 0.005, 0.0005, 0],
                   rgba=[0.95, 0.85, 0.30, 0.6],
                   contype=0, conaffinity=0)

    # Mocap
    mocap = world.add_body(name="mocap", pos=[0.0, 0.0, MOCAP_HIGH_Z], mocap=True)
    mocap.add_geom(name="mocap_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[0.001, 0, 0], rgba=[1, 0, 0, 0],
                   contype=0, conaffinity=0)

    # Carrier (flipped)
    carrier = world.add_body(name="carrier", pos=[0.0, 0.0, MOCAP_HIGH_Z],
                             quat=list(CARRIER_QUAT), gravcomp=1.0)
    carrier.add_freejoint(name="carrier_free")
    carrier.add_geom(name="carrier_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                     size=[0.001, 0, 0], rgba=[0, 1, 0, 0],
                     contype=0, conaffinity=0)

    hand_spec = mujoco.MjSpec.from_file(str(HAND_XML))
    cf = carrier.add_frame(pos=[0.0, 0.0, 0.0])
    host.attach(hand_spec, prefix="hand_", frame=cf)

    # Single cube
    cube = world.add_body(name="cube", pos=list(CUBE_SOURCE))
    cube.add_freejoint(name="cube_free")
    cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
                  rgba=[0.95, 0.30, 0.25, 1.0],
                  density=CUBE_DENSITY,
                  friction=[1.5, 0.1, 0.001],
                  solref=[0.005, 1], solimp=[0.99, 0.999, 0.001, 0.5, 2])

    # Welds — stiff carrier (no rotation change so this is just position lock),
    # stiff grasp to thumb tip
    # carrier_weld: relquat = (0,1,0,0) since carrier body has 180° X flip
    host.add_equality(name="carrier_weld",
                      type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1="mocap", name2="carrier",
                      data=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                      solref=[0.0005, 1], solimp=[0.999, 0.9999, 0.0001, 0.5, 2])
    # Weld cube directly to mocap (skip LEAP kinematics for grip) — cleaner
    # and avoids carrier/thumb-frame rotation complications. LEAP fingers
    # remain as a visual cage; mocap drives the cube position rigidly.
    host.add_equality(name="grasp",
                      type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1="mocap", name2="cube",
                      data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                      active=False,
                      solref=[0.0005, 1], solimp=[0.999, 0.9999, 0.0001, 0.5, 2])

    return host.compile()


def get_actuator_map(model):
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)}


def apply_pose(data, name2act, pose):
    for j, v in pose.items():
        key = f"hand_{j}_act"
        if key in name2act:
            data.ctrl[name2act[key]] = v


def main():
    print("MINIMAL PICK-AND-PLACE TEST")
    model = build_scene()
    data = mujoco.MjData(model)
    name2act = get_actuator_map(model)
    cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_palm")
    thumb_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_th_ds")
    grasp_eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp")

    data.mocap_pos[0] = [0.0, 0.0, MOCAP_HIGH_Z]
    data.mocap_quat[0] = [1, 0, 0, 0]
    apply_pose(data, name2act, OPEN_POSE)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.04, 0.27]
    cam.distance = 0.40
    cam.azimuth = 60.0
    cam.elevation = -22.0
    renderer = mujoco.Renderer(model, width=RES_W, height=RES_H)
    frames = []

    # Schedule
    T_SETTLE   = 1.5
    T_DESCEND  = T_SETTLE + 2.0    # 1.5-3.5
    T_GRASP    = T_DESCEND + 0.4   # 3.5-3.9 weld engage
    T_LIFT     = T_GRASP + 2.5     # 3.9-6.4
    T_TRANSPORT= T_LIFT + 2.5      # 6.4-8.9
    T_PLACE    = T_TRANSPORT + 2.0 # 8.9-10.9
    T_RELEASE  = T_PLACE + 0.4     # 10.9-11.3 weld release
    T_RETRACT  = T_RELEASE + 2.0   # 11.3-13.3
    TOTAL = T_RETRACT + 1.0        # +1s hold

    steps_per_frame = max(1, int(round((1.0 / FPS) / DT)))
    total_steps = int(TOTAL / DT)
    grasp_activated = False

    def diag(t):
        cp = data.xpos[cube_bid]
        thp = data.xpos[thumb_bid]
        mp = data.mocap_pos[0]
        active = data.eq_active[grasp_eq_id]
        print(f"  t={t:5.2f}  mocap=({mp[0]:+.3f},{mp[1]:+.3f},{mp[2]:+.3f})  "
              f"thumb=({thp[0]:+.3f},{thp[1]:+.3f},{thp[2]:+.3f})  "
              f"cube=({cp[0]:+.3f},{cp[1]:+.3f},{cp[2]:+.3f})  weld={active}")

    last_diag_t = -1
    for step in range(total_steps):
        t = step * DT

        # Mocap trajectory
        if t < T_SETTLE:
            mpos = [0.0, 0.0, MOCAP_HIGH_Z]
        elif t < T_DESCEND:
            u = smooth(t, T_SETTLE, T_DESCEND)
            mpos = [CUBE_SOURCE[0], CUBE_SOURCE[1],
                    (1-u) * MOCAP_HIGH_Z + u * MOCAP_GRASP_Z]
        elif t < T_GRASP:
            mpos = [CUBE_SOURCE[0], CUBE_SOURCE[1], MOCAP_GRASP_Z]
        elif t < T_LIFT:
            u = smooth(t, T_GRASP, T_LIFT)
            mpos = [CUBE_SOURCE[0], CUBE_SOURCE[1],
                    (1-u) * MOCAP_GRASP_Z + u * MOCAP_HIGH_Z]
        elif t < T_TRANSPORT:
            u = smooth(t, T_LIFT, T_TRANSPORT)
            mpos = [(1-u)*CUBE_SOURCE[0] + u*CUBE_TARGET[0],
                    (1-u)*CUBE_SOURCE[1] + u*CUBE_TARGET[1],
                    MOCAP_HIGH_Z]
        elif t < T_PLACE:
            u = smooth(t, T_TRANSPORT, T_PLACE)
            mpos = [CUBE_TARGET[0], CUBE_TARGET[1],
                    (1-u) * MOCAP_HIGH_Z + u * MOCAP_GRASP_Z]
        elif t < T_RELEASE:
            mpos = [CUBE_TARGET[0], CUBE_TARGET[1], MOCAP_GRASP_Z]
        elif t < T_RETRACT:
            u = smooth(t, T_RELEASE, T_RETRACT)
            mpos = [CUBE_TARGET[0], CUBE_TARGET[1],
                    (1-u) * MOCAP_GRASP_Z + u * MOCAP_HIGH_Z]
        else:
            mpos = [CUBE_TARGET[0], CUBE_TARGET[1], MOCAP_HIGH_Z]
        data.mocap_pos[0] = mpos

        # Weld toggle (cube to mocap)
        if t >= T_GRASP and t < T_RELEASE:
            if not grasp_activated:
                # Set weld anchor1 to where the cube currently is, in mocap's frame.
                # Mocap is at identity orientation, so world relative = local.
                cp = data.xpos[cube_bid]
                mp = data.mocap_pos[0]
                # anchor1 points FROM mocap origin to where cube should be.
                # MuJoCo's weld constraint convention requires anchor1 = mocap - cube
                # for the cube to be held at its CURRENT pose offset below mocap.
                model.eq_data[grasp_eq_id, 0] = float(mp[0] - cp[0])
                model.eq_data[grasp_eq_id, 1] = float(mp[1] - cp[1])
                model.eq_data[grasp_eq_id, 2] = float(mp[2] - cp[2])
                data.eq_active[grasp_eq_id] = 1
                grasp_activated = True
                print(f"  [t={t:.2f}] WELD ACTIVATED: cube relative to mocap = "
                      f"({cp[0]-mp[0]:+.4f},{cp[1]-mp[1]:+.4f},{cp[2]-mp[2]:+.4f})")
        else:
            if grasp_activated:
                data.eq_active[grasp_eq_id] = 0
                grasp_activated = False
                print(f"  [t={t:.2f}] WELD DEACTIVATED")

        apply_pose(data, name2act, OPEN_POSE)
        mujoco.mj_step(model, data)

        # Diagnostic every 0.5s
        if int(t * 2) != int(last_diag_t * 2):
            diag(t)
            last_diag_t = t

        if step % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam)
            raw = renderer.render().copy()
            frames.append(raw)

    print(f"\n  Final cube position: {data.xpos[cube_bid]}")
    print(f"  Expected target: ({CUBE_TARGET[0]:.3f}, {CUBE_TARGET[1]:.3f}, {TABLE_Z + CUBE_HALF:.3f})")
    cp = data.xpos[cube_bid]
    err = math.sqrt((cp[0]-CUBE_TARGET[0])**2 + (cp[1]-CUBE_TARGET[1])**2)
    print(f"  XY placement error: {err*1000:.1f} mm")
    print(f"  Cube on table (z>{TABLE_Z+CUBE_HALF-0.01:.3f}): {cp[2] > TABLE_Z+CUBE_HALF-0.01}")

    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    print(f"  encoding {len(frames)} frames -> {OUT_VIDEO.name}")
    iio.imwrite(OUT_VIDEO, np.asarray(frames), fps=FPS, codec="libx264")


if __name__ == "__main__":
    main()
