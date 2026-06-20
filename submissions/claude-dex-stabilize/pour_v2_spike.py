"""Pour v2 spike — mocap carrier + weld grasp architecture (per Tassa/Mochan).

Replaces the old 3-stage PD chain with:
  1. mocap body + weld -> drives LEAP root smoothly without actuator spring
  2. weld equality -> rigid grasp of cup by thumb tip (toggled on after settle)
  3. damped contacts (solref second arg = 1) -> no ball-launch artifacts
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
HAND_XML = HERE / "assets" / "leap_hand" / "right_hand.xml"

DT = 0.001  # 1 kHz per research recommendation
SETTLE_S = 1.0
TILT_S = 4.0

CUP_OUTER_HALF = 0.025
CUP_WALL_T = 0.0015
CUP_HEIGHT_HALF = 0.015
CUP_DENSITY = 350.0
CUP_INIT_POS = (-0.022, 0.020, 0.346)

BALL_R = 0.003
N_BALLS = 8

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
    host.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    host.option.impratio = 100.0

    world = host.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[0, 0, 0.05], rgba=[0.08, 0.08, 0.1, 1.0])
    world.add_light(pos=[0.0, 0.3, 1.2], dir=[0.0, -0.2, -1.0],
                    diffuse=[1.0, 1.0, 1.0])

    # Mocap body — kinematic, drives the carrier via weld
    mocap = world.add_body(name="mocap", pos=[0.0, 0.0, 0.20], mocap=True)
    mocap.add_geom(name="mocap_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[0.003, 0, 0], rgba=[1, 0, 0, 0.3],
                   contype=0, conaffinity=0)

    # Carrier: free body with freejoint, driven into the mocap pose by weld
    carrier = world.add_body(name="carrier", pos=[0.0, 0.0, 0.20], gravcomp=1.0)
    carrier.add_freejoint(name="carrier_free")
    carrier.add_geom(name="carrier_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                     size=[0.002, 0, 0], rgba=[0, 1, 0, 0.0],
                     contype=0, conaffinity=0)

    # LEAP attached under carrier
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

    # 8 balls — damped contacts
    rng = np.random.default_rng(42)
    z_base = CUP_INIT_POS[2] - CUP_HEIGHT_HALF + CUP_WALL_T + BALL_R + 0.001
    off = BALL_R + 0.001
    grid = []
    for layer in range(2):
        for ix in [-1, 1]:
            for iy in [-1, 1]:
                grid.append((ix * off, iy * off, layer * (2 * BALL_R + 0.001)))
    for i, (dx, dy, dz) in enumerate(grid[:N_BALLS]):
        bx = CUP_INIT_POS[0] + dx + rng.uniform(-0.0005, 0.0005)
        by = CUP_INIT_POS[1] + dy + rng.uniform(-0.0005, 0.0005)
        bz = z_base + dz
        ball = world.add_body(name=f"ball_{i}", pos=[bx, by, bz])
        ball.add_freejoint(name=f"ball_{i}_free")
        ball.add_geom(name=f"ball_{i}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[BALL_R, 0, 0],
                      rgba=[0.95, 0.30 + 0.08*i, 0.30, 1.0],
                      density=400.0,
                      friction=[1.0, 0.05, 0.001],
                      solref=[0.005, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])

    # Weld constraints
    # Carrier <- mocap: rigid kinematic drive (active=true from start)
    host.add_equality(name="carrier_weld",
                      type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1="mocap", name2="carrier",
                      data=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                      solref=[0.02, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])

    # Cup <- thumb_tip: rigid grasp, active=false initially (toggled after settle)
    host.add_equality(name="grasp_weld",
                      type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1="hand_th_ds", name2="cup",
                      data=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                      active=False,
                      solref=[0.02, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])

    return host.compile()


def get_actuator_map(model):
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)}


def apply_finger_pose(data, name2act, pose):
    for j, v in pose.items():
        key = f"hand_{j}_act"
        if key in name2act:
            data.ctrl[name2act[key]] = v


def quat_y_rotation(angle_rad):
    """Quaternion (w, x, y, z) for rotation around Y by angle_rad."""
    return [math.cos(angle_rad / 2), 0, math.sin(angle_rad / 2), 0]


def main():
    print("=" * 60)
    print("POUR V2 SPIKE — mocap + weld architecture")
    print("=" * 60)
    model = build_scene()
    print(f"  bodies: {model.nbody}, geoms: {model.ngeom}, "
          f"actuators: {model.nu}, equalities: {model.neq}")
    data = mujoco.MjData(model)
    name2act = get_actuator_map(model)
    print(f"  actuators: {list(name2act.keys())[:5]}... ({len(name2act)} total)")

    cup_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    ball_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ball_{i}")
                 for i in range(N_BALLS)]
    grasp_eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
    print(f"  grasp equality id: {grasp_eq_id}")

    # Initial mocap pose: at (0,0,0.20), no rotation
    data.mocap_pos[0] = [0.0, 0.0, 0.20]
    data.mocap_quat[0] = [1, 0, 0, 0]

    apply_finger_pose(data, name2act, CAGE_BASE)

    # Phase A: settle (grasp not yet active)
    print(f"\n  Phase A: settle ({SETTLE_S}s)")
    for _ in range(int(SETTLE_S / DT)):
        apply_finger_pose(data, name2act, CAGE_BASE)
        mujoco.mj_step(model, data)
    cup0 = data.xpos[cup_bid].copy()
    print(f"    cup after settle: {cup0}")
    print(f"    cup held (z > 0.25): {cup0[2] > 0.25}")
    n_in = 0
    for bid in ball_bids:
        bp = data.xpos[bid]
        if (abs(bp[0] - cup0[0]) < CUP_OUTER_HALF
                and abs(bp[1] - cup0[1]) < CUP_OUTER_HALF
                and bp[2] > cup0[2] - CUP_HEIGHT_HALF):
            n_in += 1
    print(f"    balls inside cup: {n_in}/{N_BALLS}")

    # Activate grasp weld now
    print("\n  Activating grasp weld...")
    model.eq_active0[grasp_eq_id] = 1
    data.eq_active[grasp_eq_id] = 1

    # Let the weld settle the cup pose
    for _ in range(int(0.3 / DT)):
        apply_finger_pose(data, name2act, CAGE_BASE)
        mujoco.mj_step(model, data)
    cup_grasped = data.xpos[cup_bid].copy()
    print(f"    cup after grasp-weld activate: {cup_grasped}")

    # Phase B: tilt by setting mocap quaternion sinusoidally
    print(f"\n  Phase B: tilt to 90° via mocap_quat ({TILT_S}s)...")
    target = math.radians(90)
    for step in range(int(TILT_S / DT)):
        t = step * DT
        u = 0.5 - 0.5 * math.cos(math.pi * min(1.0, t / TILT_S))
        angle = u * target
        data.mocap_quat[0] = quat_y_rotation(angle)
        apply_finger_pose(data, name2act, CAGE_BASE)
        mujoco.mj_step(model, data)
        if step % int(0.5 / DT) == 0:
            cup_now = data.xpos[cup_bid].copy()
            print(f"    t={t:.2f}  target_tilt={math.degrees(angle):5.1f}°  "
                  f"cup_z={cup_now[2]:.3f}")

    print("\n  Let balls settle 2s more...")
    for _ in range(int(2.0 / DT)):
        apply_finger_pose(data, name2act, CAGE_BASE)
        mujoco.mj_step(model, data)

    cup_final = data.xpos[cup_bid].copy()
    print(f"\n  cup final: {cup_final}")
    print("  ball final positions:")
    n_floor = 0
    for i, bid in enumerate(ball_bids):
        bp = data.xpos[bid]
        on_floor = bp[2] < 0.03
        print(f"    ball {i}: ({bp[0]:+.3f}, {bp[1]:+.3f}, {bp[2]:+.3f})"
              f"{' FLOOR' if on_floor else ''}")
        if on_floor:
            n_floor += 1

    # Check max ball velocity at end (signal for "balls launched")
    print(f"\n  balls on floor: {n_floor}/{N_BALLS}")
    print(f"  cup held: {cup_final[2] > 0.10}")


if __name__ == "__main__":
    main()
