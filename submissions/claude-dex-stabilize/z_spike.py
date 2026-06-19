"""Z spike — validate two unknowns:
  1) Can LEAP pinch-grip a cylindrical pen?
  2) Can a 2-slide mount drive the whole hand in a small circle
     without the pen flying off?

Outputs to console:
  - settled fingertip positions (for PINCH_BASE calibration)
  - pen position after 2s settle
  - pen position after 2s circular motion
  - PASS/FAIL verdict
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
HAND_XML = HERE / "assets" / "leap_hand" / "right_hand.xml"

DT = 0.002
SETTLE_S = 1.0
MOTION_S = 2.0
CIRCLE_R = 0.015  # 1.5 cm radius circle
CIRCLE_PERIOD = 4.0  # full revolution per 4s

# Pinch pose guesses. Thumb opposes index+middle; ring tucked away.
# Joint angles in radians. Tune iteratively.
PINCH_BASE = {
    # index: curl strongly, tip points back toward palm
    "if_mcp": 1.60, "if_rot": -0.30, "if_pip": 1.50, "if_dip": 0.90,
    # middle: curl too, helps support the pen
    "mf_mcp": 1.60, "mf_rot": -0.15, "mf_pip": 1.50, "mf_dip": 0.90,
    # ring: out of way
    "rf_mcp": 2.00, "rf_rot": -0.20, "rf_pip": 1.70, "rf_dip": 1.20,
    # thumb: max rotation across palm to oppose index, then curl forward
    "th_cmc": 2.00, "th_axl": 2.00, "th_mcp": 2.00, "th_ipl": 1.20,
}

# Pen geometry
PEN_RADIUS = 0.005      # 5mm
PEN_HALF_LEN = 0.06     # 12cm total
PEN_DENSITY = 300.0


def build_scene(with_pen: bool):
    host = mujoco.MjSpec()
    host.option.timestep = DT
    host.option.gravity = [0.0, 0.0, -9.81]
    host.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    world = host.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[0, 0, 0.05], rgba=[0.08, 0.08, 0.1, 1.0])
    world.add_light(pos=[0.0, 0.3, 1.2], dir=[0.0, -0.2, -1.0],
                    diffuse=[1.0, 1.0, 1.0])

    # Mount body with X+Y slide joints so we can drive hand in a horizontal circle.
    mount = world.add_body(name="mount", pos=[0.0, 0.0, 0.20])
    mount.add_joint(name="mount_x", type=mujoco.mjtJoint.mjJNT_SLIDE,
                    axis=[1, 0, 0], range=[-0.05, 0.05], damping=0.5)
    mount.add_joint(name="mount_y", type=mujoco.mjtJoint.mjJNT_SLIDE,
                    axis=[0, 1, 0], range=[-0.05, 0.05], damping=0.5)
    # invisible geom so the mount body is valid
    mount.add_geom(name="mount_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[0.002, 0, 0], rgba=[1, 0, 0, 0.3], contype=0, conaffinity=0)

    hand_spec = mujoco.MjSpec.from_file(str(HAND_XML))
    # Attach the LEAP under the mount frame at origin.
    mount_frame = mount.add_frame(pos=[0.0, 0.0, 0.0])
    host.attach(hand_spec, prefix="hand_", frame=mount_frame)

    if with_pen:
        # Pen oriented vertically (long axis along world Z).
        # Roughly between thumb tip and index tip. We'll see where they actually settle.
        pen = world.add_body(name="pen", pos=[-0.035, 0.020, 0.380])
        pen.add_freejoint(name="pen_free")
        pen.add_geom(name="pen_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                     size=[PEN_RADIUS, PEN_HALF_LEN, 0],
                     rgba=[0.9, 0.2, 0.2, 1.0],
                     density=PEN_DENSITY,
                     friction=[1.5, 0.1, 0.001])

    # actuators on mount slide joints
    host.add_actuator(name="mount_x_act", target="mount_x",
                      trntype=mujoco.mjtTrn.mjTRN_JOINT,
                      gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                      biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                      gainprm=[200.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      biasprm=[0, -200.0, -20.0, 0, 0, 0, 0, 0, 0, 0],
                      ctrlrange=[-0.05, 0.05])
    host.add_actuator(name="mount_y_act", target="mount_y",
                      trntype=mujoco.mjtTrn.mjTRN_JOINT,
                      gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                      biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                      gainprm=[200.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      biasprm=[0, -200.0, -20.0, 0, 0, 0, 0, 0, 0, 0],
                      ctrlrange=[-0.05, 0.05])

    return host.compile()


def get_actuator_map(model):
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)}


def apply_finger_pose(data, name2act, pose: dict):
    for j, v in pose.items():
        key = f"hand_{j}_act"
        if key in name2act:
            data.ctrl[name2act[key]] = v


def site_or_body_pos(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        return None
    return data.xpos[bid].copy()


def main():
    print("=" * 60)
    print("SPIKE 1: PINCH POSE CALIBRATION (no pen, just settle)")
    print("=" * 60)

    model = build_scene(with_pen=False)
    data = mujoco.MjData(model)
    name2act = get_actuator_map(model)
    print(f"actuators: {list(name2act.keys())}")

    apply_finger_pose(data, name2act, PINCH_BASE)
    # mount actuators target = 0
    if "mount_x_act" in name2act:
        data.ctrl[name2act["mount_x_act"]] = 0.0
        data.ctrl[name2act["mount_y_act"]] = 0.0

    for _ in range(int(SETTLE_S / DT)):
        apply_finger_pose(data, name2act, PINCH_BASE)
        mujoco.mj_step(model, data)

    if_tip = site_or_body_pos(model, data, "hand_if_ds")
    mf_tip = site_or_body_pos(model, data, "hand_mf_ds")
    rf_tip = site_or_body_pos(model, data, "hand_rf_ds")
    th_tip = site_or_body_pos(model, data, "hand_th_ds")
    palm  = site_or_body_pos(model, data, "hand_palm")
    print(f"  palm  : {palm}")
    print(f"  if_tip: {if_tip}")
    print(f"  mf_tip: {mf_tip}")
    print(f"  rf_tip: {rf_tip}")
    print(f"  th_tip: {th_tip}")
    if if_tip is not None and th_tip is not None:
        pinch_mid = (if_tip + th_tip) / 2
        gap = float(np.linalg.norm(if_tip - th_tip))
        print(f"  pinch midpoint (if-th): {pinch_mid}")
        print(f"  pinch gap   (if-th): {gap*1000:.1f} mm")

    print()
    print("=" * 60)
    print("SPIKE 2: PEN GRIP STABILITY")
    print("=" * 60)
    model = build_scene(with_pen=True)
    data = mujoco.MjData(model)
    name2act = get_actuator_map(model)
    pen_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pen")

    apply_finger_pose(data, name2act, PINCH_BASE)
    data.ctrl[name2act["mount_x_act"]] = 0.0
    data.ctrl[name2act["mount_y_act"]] = 0.0

    # Phase A: settle
    for _ in range(int(SETTLE_S / DT)):
        apply_finger_pose(data, name2act, PINCH_BASE)
        mujoco.mj_step(model, data)
    pen_after_settle = data.xpos[pen_bid].copy()
    print(f"  pen after settle: {pen_after_settle}")
    held_after_settle = pen_after_settle[2] > 0.30
    print(f"  pen held after settle (z > 0.30): {held_after_settle}")

    # Phase B: mount circular motion
    pen_positions = [pen_after_settle]
    for step in range(int(MOTION_S / DT)):
        t = step * DT
        theta = 2 * math.pi * (t / CIRCLE_PERIOD)
        tx = CIRCLE_R * math.cos(theta)
        ty = CIRCLE_R * math.sin(theta)
        data.ctrl[name2act["mount_x_act"]] = tx
        data.ctrl[name2act["mount_y_act"]] = ty
        apply_finger_pose(data, name2act, PINCH_BASE)
        mujoco.mj_step(model, data)
        if step % 100 == 0:
            pen_positions.append(data.xpos[pen_bid].copy())

    pen_final = data.xpos[pen_bid].copy()
    print(f"  pen after motion: {pen_final}")
    held_final = pen_final[2] > 0.25
    print(f"  pen held after motion (z > 0.25): {held_final}")

    # Did pen follow the mount?
    # Sample mount X actual position via joint qpos
    mount_x_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mount_x")
    mount_y_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mount_y")
    if mount_x_jid >= 0:
        mx_qpos = data.qpos[model.jnt_qposadr[mount_x_jid]]
        my_qpos = data.qpos[model.jnt_qposadr[mount_y_jid]]
        print(f"  mount actual pos: ({mx_qpos:.4f}, {my_qpos:.4f})")
        print(f"  mount target pos: ({CIRCLE_R * math.cos(2*math.pi*MOTION_S/CIRCLE_PERIOD):.4f}, "
              f"{CIRCLE_R * math.sin(2*math.pi*MOTION_S/CIRCLE_PERIOD):.4f})")

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    if held_after_settle and held_final:
        print("  PASS — both unknowns validated. Proceed with Z full build.")
    elif held_after_settle and not held_final:
        print("  PARTIAL — pinch holds but motion drops pen. Need stiffer grip.")
    elif not held_after_settle:
        print("  FAIL — pen not gripped after settle. Iterate PINCH_BASE pose.")
    else:
        print("  UNKNOWN")


if __name__ == "__main__":
    main()
