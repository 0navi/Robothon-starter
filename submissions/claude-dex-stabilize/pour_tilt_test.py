"""Focused tilt test — just settle + tilt + report ball positions."""
from __future__ import annotations
import math
import numpy as np
import mujoco
from pour_main import build_scene, get_actuator_map, apply_finger_pose, CAGE_BASE, N_BALLS, DT, ball_classifier


def main():
    print("Loading scene...")
    model = build_scene(seed=12345)
    data = mujoco.MjData(model)
    name2act = get_actuator_map(model)
    cup_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    ball_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ball_{i}") for i in range(N_BALLS)]

    apply_finger_pose(data, name2act, CAGE_BASE)
    for j in ["mount_x", "mount_y", "mount_z", "mount_tilt"]:
        data.ctrl[name2act[f"{j}_act"]] = 0.0

    print("\nSETTLE 1.0s...")
    for _ in range(int(1.0 / DT)):
        apply_finger_pose(data, name2act, CAGE_BASE)
        mujoco.mj_step(model, data)
    cup0 = data.xpos[cup_bid].copy()
    print(f"  cup after settle: {cup0}")
    for i, bid in enumerate(ball_bids):
        bp = data.xpos[bid]
        print(f"  ball {i}: ({bp[0]:+.3f}, {bp[1]:+.3f}, {bp[2]:+.3f})")

    # Now tilt aggressively — full 180° (upside down) over 3s
    target_tilt = math.radians(170)
    tilt_s = 3.0
    print(f"\nTILT to {math.degrees(target_tilt):.0f}° over {tilt_s}s...")
    for step in range(int(tilt_s / DT)):
        t = step * DT
        u = min(1.0, t / tilt_s)
        # smooth ramp
        u_smooth = 0.5 - 0.5 * math.cos(math.pi * u)
        data.ctrl[name2act["mount_tilt_act"]] = u_smooth * target_tilt
        apply_finger_pose(data, name2act, CAGE_BASE)
        mujoco.mj_step(model, data)

        # Snapshot mid-tilt
        if step % int(0.5 / DT) == 0:
            cup_now = data.xpos[cup_bid]
            tilt_actual = data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mount_tilt")]]
            print(f"  t={t:.2f}  target={math.degrees(u_smooth*target_tilt):6.1f}°  "
                  f"actual={math.degrees(tilt_actual):6.1f}°  cup_z={cup_now[2]:.3f}")

    print("\nLet balls fall 2s more...")
    for _ in range(int(2.0 / DT)):
        data.ctrl[name2act["mount_tilt_act"]] = target_tilt
        apply_finger_pose(data, name2act, CAGE_BASE)
        mujoco.mj_step(model, data)

    cup_final = data.xpos[cup_bid].copy()
    print(f"\n  cup final: {cup_final}")
    n_floor = 0
    for i, bid in enumerate(ball_bids):
        bp = data.xpos[bid]
        cls = ball_classifier(bp)
        print(f"  ball {i}: state={cls:>5s}  xyz=({bp[0]:+.3f}, {bp[1]:+.3f}, {bp[2]:+.3f})")
        if bp[2] < 0.03:
            n_floor += 1
    print(f"\n  balls on/near floor: {n_floor}/{N_BALLS}")


if __name__ == "__main__":
    main()
