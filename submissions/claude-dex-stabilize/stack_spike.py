"""Stack spike — validate pick-and-place geometry.

Question: with LEAP attached fingers-down (carrier flipped 180° around X),
can the hand reach down to a cube on a surface, close cage, lift?
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
HAND_XML = HERE / "assets" / "leap_hand" / "right_hand.xml"

DT = 0.001
CUBE_HALF = 0.018
CUBE_DENSITY = 200.0

# LEAP cage pose
CAGE_BASE = {
    "if_mcp": 1.30, "if_rot": 0.10, "if_pip": 1.00, "if_dip": 0.60,
    "mf_mcp": 1.30, "mf_rot": 0.00, "mf_pip": 1.00, "mf_dip": 0.60,
    "rf_mcp": 1.30, "rf_rot": -0.10, "rf_pip": 1.00, "rf_dip": 0.60,
    "th_cmc": 1.20, "th_axl": 0.80, "th_mcp": 1.20, "th_ipl": 1.00,
}
OPEN_POSE = {k: v * 0.3 for k, v in CAGE_BASE.items()}  # all joints to 30%


def build_scene(carrier_quat=(1, 0, 0, 0), cube_pos=(0.0, 0.0, 0.018)):
    """carrier_quat lets us test different LEAP orientations."""
    host = mujoco.MjSpec()
    host.option.timestep = DT
    host.option.gravity = [0.0, 0.0, -9.81]
    host.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    host.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    host.option.impratio = 100.0

    world = host.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[0, 0, 0.05], rgba=[0.10, 0.10, 0.12, 1.0])
    world.add_light(pos=[0.0, 0.3, 1.0], dir=[0.0, -0.2, -1.0],
                    diffuse=[1.0, 1.0, 1.0])

    # Mocap puppeteer
    mocap = world.add_body(name="mocap", pos=[0.0, 0.0, 0.40], mocap=True)
    mocap.add_geom(name="mocap_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[0.003, 0, 0], rgba=[1, 0, 0, 0.4],
                   contype=0, conaffinity=0)

    # Carrier (free body, gravcomp), with optional quat to flip LEAP orientation
    carrier = world.add_body(name="carrier", pos=[0.0, 0.0, 0.40],
                             quat=list(carrier_quat), gravcomp=1.0)
    carrier.add_freejoint(name="carrier_free")
    carrier.add_geom(name="carrier_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                     size=[0.002, 0, 0], rgba=[0, 1, 0, 0.4],
                     contype=0, conaffinity=0)

    hand_spec = mujoco.MjSpec.from_file(str(HAND_XML))
    carrier_frame = carrier.add_frame(pos=[0.0, 0.0, 0.0])
    host.attach(hand_spec, prefix="hand_", frame=carrier_frame)

    # Target cube
    cube = world.add_body(name="cube", pos=list(cube_pos))
    cube.add_freejoint(name="cube_free")
    cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
                  rgba=[0.95, 0.55, 0.20, 1.0], density=CUBE_DENSITY,
                  friction=[1.5, 0.1, 0.001],
                  solref=[0.005, 1], solimp=[0.99, 0.999, 0.001, 0.5, 2])

    # Welds
    host.add_equality(name="carrier_weld",
                      type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1="mocap", name2="carrier",
                      data=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                      solref=[0.02, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])

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
    print("=" * 60)
    print("STACK SPIKE — find LEAP orientation for pick-from-table")
    print("=" * 60)

    # Test 4 orientations
    orientations = {
        "identity (fingers UP)": (1, 0, 0, 0),
        "180° X (flip)": (0, 1, 0, 0),
        "180° Y": (0, 0, 1, 0),
        "180° Z": (0, 0, 0, 1),
    }

    for name, q in orientations.items():
        print(f"\n  testing: {name}, carrier_quat={q}")
        model = build_scene(carrier_quat=q, cube_pos=(0.0, 0.0, 0.018))
        data = mujoco.MjData(model)
        name2act = get_actuator_map(model)

        # mocap stays at init pos
        data.mocap_pos[0] = [0.0, 0.0, 0.40]
        data.mocap_quat[0] = [1, 0, 0, 0]

        apply_pose(data, name2act, OPEN_POSE)

        # Settle 0.5s without grip (fingers open, hand at z=0.40)
        for _ in range(int(0.5 / DT)):
            apply_pose(data, name2act, OPEN_POSE)
            mujoco.mj_step(model, data)

        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_palm")
        if_ds = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_if_ds")
        mf_ds = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_mf_ds")
        rf_ds = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_rf_ds")
        th_ds = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_th_ds")
        print(f"    palm  z = {data.xpos[palm][2]:.3f}")
        print(f"    if_ds z = {data.xpos[if_ds][2]:.3f}  (z<palm = fingers DOWN YES)")
        print(f"    mf_ds z = {data.xpos[mf_ds][2]:.3f}")
        print(f"    rf_ds z = {data.xpos[rf_ds][2]:.3f}")
        print(f"    th_ds z = {data.xpos[th_ds][2]:.3f}")
        fingers_avg = (data.xpos[if_ds][2] + data.xpos[mf_ds][2] + data.xpos[rf_ds][2]) / 3
        verdict = "FINGERS DOWN YES" if fingers_avg < data.xpos[palm][2] else "fingers up"
        print(f"    -> {verdict}")


if __name__ == "__main__":
    main()
