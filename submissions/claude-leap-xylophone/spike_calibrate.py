"""Calibration spike: print LEAP fingertip world positions in rest/strike poses.

We'll use these numbers to place xylophone keys below each fingertip.
"""
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
HAND_XML = HERE / "assets" / "leap_hand" / "right_hand.xml"

# Two poses to test: rest (fingers extended-ish), strike (fingers curled to tap downward).
# In CAGE_BASE the existing demo uses mcp=1.3 (deeply curled around held cube).
# For striking down on a key, we want a SMALLER curl at rest and a LARGER curl on strike,
# so the tip swings downward.
REST_POSE = {
    "if_mcp": 0.20, "if_rot": 0.0, "if_pip": 0.30, "if_dip": 0.30,
    "mf_mcp": 0.20, "mf_rot": 0.0, "mf_pip": 0.30, "mf_dip": 0.30,
    "rf_mcp": 0.20, "rf_rot": 0.0, "rf_pip": 0.30, "rf_dip": 0.30,
    "th_cmc": 0.50, "th_axl": 0.50, "th_mcp": 0.50, "th_ipl": 0.50,
}
STRIKE_POSE = {
    "if_mcp": 1.20, "if_rot": 0.0, "if_pip": 1.20, "if_dip": 0.80,
    "mf_mcp": 1.20, "mf_rot": 0.0, "mf_pip": 1.20, "mf_dip": 0.80,
    "rf_mcp": 1.20, "rf_rot": 0.0, "rf_pip": 1.20, "rf_dip": 0.80,
    "th_cmc": 1.20, "th_axl": 0.80, "th_mcp": 1.20, "th_ipl": 1.20,
}

TIP_GEOMS = ["hand_if_tip", "hand_mf_tip", "hand_rf_tip", "hand_th_tip"]


def build():
    host = mujoco.MjSpec()
    host.option.timestep = 0.002
    host.option.gravity = [0, 0, -9.81]
    host.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world = host.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[0, 0, 0.05], rgba=[0.06, 0.07, 0.09, 1])
    world.add_light(pos=[0, 0.3, 1.2], dir=[0, -0.2, -1], diffuse=[1, 1, 1])
    hand = mujoco.MjSpec.from_file(str(HAND_XML))
    mount = world.add_frame(pos=[0.0, 0.0, 0.30])
    host.attach(hand, prefix="hand_", frame=mount)
    return host.compile()


def settle(model, data, pose, steps=400):
    name2act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
                for i in range(model.nu)}
    for j, v in pose.items():
        k = f"hand_{j}_act"
        if k in name2act:
            data.ctrl[name2act[k]] = v
    for _ in range(steps):
        mujoco.mj_step(model, data)


def dump_tips(model, data, label):
    print(f"--- {label} ---")
    for gname in TIP_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid < 0:
            print(f"  {gname}: NOT FOUND")
            continue
        p = data.geom_xpos[gid]
        print(f"  {gname}:  x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}")
    # also palm
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_palm")
    if pid >= 0:
        p = data.xpos[pid]
        print(f"  palm body: x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}")


def main():
    model = build()
    data = mujoco.MjData(model)

    settle(model, data, REST_POSE)
    dump_tips(model, data, "REST POSE (fingers slightly curled)")

    # reset & settle to strike pose
    data2 = mujoco.MjData(model)
    settle(model, data2, STRIKE_POSE)
    dump_tips(model, data2, "STRIKE POSE (fingers fully curled)")

    # Compute Z drop per finger (rest_z - strike_z = how far the tip moves downward
    # when going from rest to strike). Positive = drops down.
    print("--- DELTA Z (rest_z - strike_z, positive = strike drops tip down) ---")
    for gname in TIP_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid < 0:
            continue
        # need to re-settle each to read both
    # easier: hold both datas in arrays
    rest_xpos = {}
    strike_xpos = {}
    d_rest = mujoco.MjData(model); settle(model, d_rest, REST_POSE)
    d_strike = mujoco.MjData(model); settle(model, d_strike, STRIKE_POSE)
    for gname in TIP_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        rest_xpos[gname] = d_rest.geom_xpos[gid].copy()
        strike_xpos[gname] = d_strike.geom_xpos[gid].copy()
        d = rest_xpos[gname] - strike_xpos[gname]
        print(f"  {gname}: dx={d[0]:+.4f}  dy={d[1]:+.4f}  dz={d[2]:+.4f}  "
              f"(moves {abs(d[2])*1000:.1f} mm in Z)")


if __name__ == "__main__":
    main()
