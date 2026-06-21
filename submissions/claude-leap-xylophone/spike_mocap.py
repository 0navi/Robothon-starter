"""Spike: verify LEAP can be attached under a mocap body and slid in Y.

Goal: confirm that
  (a) MjSpec lets us attach the LEAP hand under a mocap-body frame
  (b) updating data.mocap_pos[wrist] kinematically translates the whole hand
  (c) the finger joints still actuate normally while the wrist moves
"""
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
HAND_XML = HERE / "assets" / "leap_hand" / "right_hand.xml"


def build():
    host = mujoco.MjSpec()
    host.option.timestep = 0.002
    host.option.gravity = [0, 0, -9.81]
    host.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world = host.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[0, 0, 0.05], rgba=[0.06, 0.07, 0.09, 1])
    world.add_light(pos=[0, 0.3, 1.2], dir=[0, -0.2, -1], diffuse=[1, 1, 1])

    wrist = world.add_body(name="wrist", mocap=True,
                           pos=[0.0, 0.0, 0.30])
    # Add a visible marker geom on the mocap body (no contact).
    wrist.add_geom(name="wrist_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[0.012, 0, 0], rgba=[0.9, 0.2, 0.2, 0.7],
                   contype=0, conaffinity=0)
    mount = wrist.add_frame(pos=[0, 0, 0])

    hand = mujoco.MjSpec.from_file(str(HAND_XML))
    host.attach(hand, prefix="hand_", frame=mount)
    model = host.compile()
    return model


def main():
    model = build()
    data = mujoco.MjData(model)
    print(f"compiled OK. nq={model.nq} nbody={model.nbody} nmocap={model.nmocap}")

    # find mocap body id + mocap index
    wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
    wrist_mocap = int(model.body_mocapid[wrist_bid])
    print(f"wrist bid={wrist_bid}, mocap index={wrist_mocap}")
    palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_palm")
    if_tip_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hand_if_tip")

    # default mocap pos
    print(f"initial mocap_pos: {data.mocap_pos[wrist_mocap]}")
    # step once to populate xpos
    mujoco.mj_forward(model, data)
    print(f"  palm world pos: {data.xpos[palm_bid]}")
    print(f"  if_tip world pos: {data.geom_xpos[if_tip_gid]}")

    # slide wrist along Y over 1.5s
    print(f"\nsliding wrist Y from -0.05 to +0.10 over 1.5 s...")
    for s in range(int(1.5 / 0.002)):
        t = s * 0.002
        # linear sweep
        target_y = -0.05 + 0.15 * (t / 1.5)
        data.mocap_pos[wrist_mocap] = [0.0, target_y, 0.30]
        mujoco.mj_step(model, data)
        if s % 200 == 0:
            print(f"  t={t:5.2f}  mocap_y={target_y:+.4f}  "
                  f"palm world y={data.xpos[palm_bid][1]:+.4f}  "
                  f"if_tip world y={data.geom_xpos[if_tip_gid][1]:+.4f}")


if __name__ == "__main__":
    main()
