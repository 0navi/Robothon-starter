"""Pour spike — validate the hardest unknowns BEFORE committing to the full build:
  1) Can LEAP grip a small open-top cup (4 walls + bottom)?
  2) Do 8 small balls stay inside the cup during the grip+settle?
  3) Can a Y-axis HINGE on the mount tilt the LEAP without breaking physics?
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
HAND_XML = HERE / "assets" / "leap_hand" / "right_hand.xml"

DT = 0.002
SETTLE_S = 1.5
TILT_S = 2.0
TILT_DEG = 90.0

# Cup geometry: open-top square cup, outer 50x50mm, walls 3mm thick, height 30mm.
# Shorter than the cage's fingertip span so the LEAP closes around the cup walls
# without sweeping the open top.
CUP_OUTER_HALF = 0.025
CUP_WALL_T = 0.0015
CUP_HEIGHT_HALF = 0.015
CUP_DENSITY = 350.0

# Cup placed where the cage cup-center sits (from cube version calibration).
CUP_POS = (-0.022, 0.020, 0.346)

BALL_R = 0.003
N_BALLS = 8

# CAGE_BASE from cube version — already calibrated to cage around a 36mm cube
# at the same position. We re-use it; the cup is similar size and the cage
# closes around its outer walls.
CAGE_BASE = {
    "if_mcp": 1.30, "if_rot": 0.10, "if_pip": 1.00, "if_dip": 0.60,
    "mf_mcp": 1.30, "mf_rot": 0.00, "mf_pip": 1.00, "mf_dip": 0.60,
    "rf_mcp": 1.30, "rf_rot": -0.10, "rf_pip": 1.00, "rf_dip": 0.60,
    "th_cmc": 1.20, "th_axl": 0.80, "th_mcp": 1.20, "th_ipl": 1.00,
}


def build_scene():
    host = mujoco.MjSpec()
    host.option.timestep = DT
    host.option.gravity = [0.0, 0.0, -9.81]
    host.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    world = host.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[0, 0, 0.05], rgba=[0.08, 0.08, 0.1, 1.0])
    world.add_light(pos=[0.0, 0.3, 1.2], dir=[0.0, -0.2, -1.0],
                    diffuse=[1.0, 1.0, 1.0])

    # Mount: 3D translate + Y-hinge for tilt.
    mount_trans = world.add_body(name="mount_trans", pos=[0.0, 0.0, 0.20])
    mount_trans.add_joint(name="mount_x", type=mujoco.mjtJoint.mjJNT_SLIDE,
                          axis=[1, 0, 0], range=[-0.30, 0.30], damping=2.0)
    mount_trans.add_joint(name="mount_y", type=mujoco.mjtJoint.mjJNT_SLIDE,
                          axis=[0, 1, 0], range=[-0.30, 0.30], damping=2.0)
    mount_trans.add_joint(name="mount_z", type=mujoco.mjtJoint.mjJNT_SLIDE,
                          axis=[0, 0, 1], range=[-0.20, 0.20], damping=2.0)
    mount_trans.add_geom(name="trans_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.001, 0, 0], rgba=[1, 0, 0, 0],
                        contype=0, conaffinity=0)

    mount_rot = mount_trans.add_body(name="mount_rot", pos=[0.0, 0.0, 0.0])
    mount_rot.add_joint(name="mount_tilt", type=mujoco.mjtJoint.mjJNT_HINGE,
                        axis=[0, 1, 0], range=[-2.5, 2.5], damping=1.0)
    mount_rot.add_geom(name="rot_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[0.001, 0, 0], rgba=[0, 1, 0, 0],
                       contype=0, conaffinity=0)

    # LEAP under the rotating mount.
    hand_spec = mujoco.MjSpec.from_file(str(HAND_XML))
    rot_frame = mount_rot.add_frame(pos=[0.0, 0.0, 0.0])
    host.attach(hand_spec, prefix="hand_", frame=rot_frame)

    # Cup: 4 walls + bottom, all under one body with a free joint.
    cup = world.add_body(name="cup", pos=list(CUP_POS))
    cup.add_freejoint(name="cup_free")
    # Bottom
    cup.add_geom(name="cup_bottom", type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=[0, 0, -CUP_HEIGHT_HALF + CUP_WALL_T],
                 size=[CUP_OUTER_HALF, CUP_OUTER_HALF, CUP_WALL_T],
                 rgba=[0.6, 0.6, 0.7, 1.0], density=CUP_DENSITY,
                 friction=[1.5, 0.1, 0.001])
    # 4 walls (X+, X-, Y+, Y-)
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
                     rgba=[0.6, 0.6, 0.7, 1.0], density=CUP_DENSITY,
                     friction=[1.5, 0.1, 0.001])

    # 8 balls inside the cup in a 2x2x2 layout. Cup interior:
    #   xy half ≈ 23mm (CUP_OUTER_HALF - CUP_WALL_T), height ≈ 27mm.
    # Balls 3mm radius -> 6mm diameter, 2x2x2 = 12mm in each direction. Plenty of room.
    rng = np.random.default_rng(42)
    z_base = CUP_POS[2] - CUP_HEIGHT_HALF + CUP_WALL_T + BALL_R + 0.001
    grid = []
    off = BALL_R + 0.001
    for layer in range(2):
        for ix in [-1, 1]:
            for iy in [-1, 1]:
                grid.append((ix * off, iy * off, layer * (2 * BALL_R + 0.001)))
    for i, (dx, dy, dz) in enumerate(grid[:N_BALLS]):
        # tiny random jitter so balls don't sit perfectly still
        bx = CUP_POS[0] + dx + rng.uniform(-0.0005, 0.0005)
        by = CUP_POS[1] + dy + rng.uniform(-0.0005, 0.0005)
        bz = z_base + dz
        ball = world.add_body(name=f"ball_{i}", pos=[bx, by, bz])
        ball.add_freejoint(name=f"ball_{i}_free")
        # Distinct colors for visibility
        hue = i / N_BALLS
        r = 0.5 + 0.5 * math.cos(2 * math.pi * hue)
        g = 0.5 + 0.5 * math.cos(2 * math.pi * (hue + 1/3))
        b = 0.5 + 0.5 * math.cos(2 * math.pi * (hue + 2/3))
        ball.add_geom(name=f"ball_{i}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[BALL_R, 0, 0],
                      rgba=[r, g, b, 1.0], density=400.0,
                      friction=[0.5, 0.05, 0.001])

    # Position actuators
    for jn in ["mount_x", "mount_y", "mount_z"]:
        host.add_actuator(name=f"{jn}_act", target=jn,
                          trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                          gainprm=[400.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          biasprm=[0, -400.0, -40.0, 0, 0, 0, 0, 0, 0, 0],
                          ctrlrange=[-0.30, 0.30])
    host.add_actuator(name="mount_tilt_act", target="mount_tilt",
                      trntype=mujoco.mjtTrn.mjTRN_JOINT,
                      gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                      biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                      gainprm=[8.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      biasprm=[0, -8.0, -1.0, 0, 0, 0, 0, 0, 0, 0],
                      ctrlrange=[-2.5, 2.5])

    return host.compile()


def name2act_map(model):
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)}


def apply_pose(data, name2act, pose):
    for j, v in pose.items():
        key = f"hand_{j}_act"
        if key in name2act:
            data.ctrl[name2act[key]] = v


def main():
    print("=" * 60)
    print("POUR SPIKE")
    print("=" * 60)

    model = build_scene()
    print(f"  bodies: {model.nbody}, geoms: {model.ngeom}, actuators: {model.nu}")

    data = mujoco.MjData(model)
    name2act = name2act_map(model)

    cup_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    ball_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ball_{i}")
                 for i in range(N_BALLS)]

    apply_pose(data, name2act, CAGE_BASE)
    # mount at neutral
    for j in ["mount_x", "mount_y", "mount_z", "mount_tilt"]:
        data.ctrl[name2act[f"{j}_act"]] = 0.0

    # Phase A: settle
    print(f"\n  Phase A: settle ({SETTLE_S}s)")
    for _ in range(int(SETTLE_S / DT)):
        apply_pose(data, name2act, CAGE_BASE)
        mujoco.mj_step(model, data)

    cup_pos = data.xpos[cup_bid].copy()
    print(f"  cup after settle: {cup_pos}")
    cup_held = cup_pos[2] > 0.25
    print(f"  cup held (z > 0.25): {cup_held}")

    n_inside = 0
    for bid in ball_bids:
        bp = data.xpos[bid].copy()
        # ball inside cup if z is around cup interior and xy near cup center
        in_xy = (abs(bp[0] - cup_pos[0]) < CUP_OUTER_HALF and
                 abs(bp[1] - cup_pos[1]) < CUP_OUTER_HALF)
        in_z = bp[2] > cup_pos[2] - CUP_HEIGHT_HALF and bp[2] < cup_pos[2] + CUP_HEIGHT_HALF
        if in_xy and in_z:
            n_inside += 1
    print(f"  balls inside cup: {n_inside}/{N_BALLS}")

    # Phase B: tilt
    print(f"\n  Phase B: tilt to {TILT_DEG} deg over {TILT_S}s")
    target_tilt = math.radians(TILT_DEG)
    for step in range(int(TILT_S / DT)):
        t = step * DT
        ratio = min(1.0, t / TILT_S)
        data.ctrl[name2act["mount_tilt_act"]] = ratio * target_tilt
        apply_pose(data, name2act, CAGE_BASE)
        mujoco.mj_step(model, data)

    cup_pos2 = data.xpos[cup_bid].copy()
    print(f"  cup after tilt: {cup_pos2}")
    cup_held2 = cup_pos2[2] > 0.20

    n_inside2 = 0
    n_on_floor = 0
    for bid in ball_bids:
        bp = data.xpos[bid].copy()
        if bp[2] < 0.03:
            n_on_floor += 1
        in_xy = (abs(bp[0] - cup_pos2[0]) < CUP_OUTER_HALF and
                 abs(bp[1] - cup_pos2[1]) < CUP_OUTER_HALF)
        in_z = bp[2] > cup_pos2[2] - CUP_HEIGHT_HALF and bp[2] < cup_pos2[2] + CUP_HEIGHT_HALF
        if in_xy and in_z:
            n_inside2 += 1
    print(f"  cup held after tilt: {cup_held2}")
    print(f"  balls remaining inside cup: {n_inside2}")
    print(f"  balls poured out (on floor): {n_on_floor}")
    print(f"  balls in transit/lost: {N_BALLS - n_inside2 - n_on_floor}")

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    if cup_held and cup_held2 and n_inside >= 6 and n_on_floor >= 3:
        print("  PASS — cup grips, balls fill, tilt pours. Proceed.")
    elif not cup_held:
        print("  FAIL — cup not gripped after settle.")
    elif n_inside < 6:
        print("  FAIL — balls didn't end up inside cup after settle.")
    elif not cup_held2:
        print("  FAIL — cup dropped during tilt.")
    elif n_on_floor == 0:
        print("  FAIL — tilt didn't pour any balls.")
    else:
        print("  PARTIAL — borderline; check counts above.")


if __name__ == "__main__":
    main()
