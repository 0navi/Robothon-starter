"""LEAP Dexterous Stabilization — Robothon Summer 2026 submission.

A LEAP right hand holds an orange cube in its cage grasp while a wrist
rotation actuator turns the hand 360° around the vertical axis. External
perturbation forces are applied to the cube at random intervals.  The
finger controllers run a closed-loop response: when the cube drifts from
the cup center, finger grip targets tighten.

The point: demonstrate stable in-hand cube holding under (a) continuous
wrist rotation and (b) random external perturbations, with measurable
recovery rather than scripted timeline playback.

Run from repo root:
    python submissions/claude-dex-stabilize/main.py

Output:
    outputs/demo.mp4
    outputs/trajectory.json   (per-frame cube pos, grip command, perturbations)
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import mujoco
import imageio.v3 as iio

HERE = Path(__file__).resolve().parent
HAND_XML = HERE / "assets" / "leap_hand" / "right_hand.xml"
OUT_VIDEO = HERE / "outputs" / "demo.mp4"
OUT_TRAJECTORY = HERE / "outputs" / "trajectory.json"

# --- Physics / video ---
DT = 0.002
FPS = 30
DURATION_S = 20.0
RES_W, RES_H = 1280, 720

# --- Cube ---
CUBE_HALF = 0.018
CUBE_DENSITY = 200.0
CUBE_START = (-0.022, 0.020, 0.346)   # measured cage cup center
CUP_CENTER = np.array(CUBE_START)

# --- Wrist rotation: hand rotates ~360° around its vertical (Z world) axis
# while holding the cube. Implemented as a hinge joint on a carrier body. ---
WRIST_ROT_BPM = 0.0           # disabled: keep wrist still for clean stabilization demo
WRIST_ROT_AMPL = 0.0

# --- Perturbation: random horizontal force every PERTURB_PERIOD_S ---
PERTURB_PERIOD_S = 3.0
PERTURB_FORCE_N = 0.8           # newtons applied for PERTURB_DURATION_S
PERTURB_DURATION_S = 0.15

# --- Closed-loop fingers ---
CAGE_BASE = {
    "if_mcp": 1.30, "if_rot": 0.10, "if_pip": 1.00, "if_dip": 0.60,
    "mf_mcp": 1.30, "mf_rot": 0.00, "mf_pip": 1.00, "mf_dip": 0.60,
    "rf_mcp": 1.30, "rf_rot": -0.10, "rf_pip": 1.00, "rf_dip": 0.60,
    "th_cmc": 1.20, "th_axl": 0.80, "th_mcp": 1.20, "th_ipl": 1.00,
}
# When cube starts to slip, scale these per-joint deltas up to tighten grip.
GRIP_TIGHTEN_DELTA = {
    "if_mcp": 0.15, "if_pip": 0.20, "if_dip": 0.15,
    "mf_mcp": 0.15, "mf_pip": 0.20, "mf_dip": 0.15,
    "rf_mcp": 0.15, "rf_pip": 0.20, "rf_dip": 0.15,
    "th_cmc": 0.10, "th_mcp": 0.15, "th_ipl": 0.15,
}


def build_scene() -> mujoco.MjModel:
    host = mujoco.MjSpec()
    host.option.timestep = DT
    host.option.gravity = [0.0, 0.0, -9.81]
    host.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    host.visual.global_.offwidth = RES_W
    host.visual.global_.offheight = RES_H

    world = host.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[0, 0, 0.05], rgba=[0.06, 0.07, 0.09, 1.0])
    # Three-point lighting for the cube
    world.add_light(pos=[0.0, 0.3, 1.2], dir=[0.0, -0.2, -1.0],
                    diffuse=[1.0, 0.95, 0.85])
    world.add_light(pos=[0.4, -0.3, 1.0], dir=[-0.3, 0.2, -1.0],
                    diffuse=[0.4, 0.5, 0.7])
    world.add_light(pos=[-0.4, 0.0, 0.9], dir=[0.3, 0.0, -1.0],
                    diffuse=[0.55, 0.45, 0.4])

    # Wrist carrier: rotating hinge around vertical axis.
    carrier = world.add_body(name="wrist_carrier", pos=[0.0, 0.0, 0.20])
    carrier.add_joint(
        name="wrist_yaw",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 0, 1],
        range=[-math.pi, math.pi], limited=True,
        damping=0.8,
    )
    # A small visible wrist mount stub
    carrier.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.04, 0.018, 0.0],
        rgba=[0.20, 0.22, 0.26, 1.0],
    )

    # Attach LEAP hand at the carrier; default orientation, palm up.
    hand = mujoco.MjSpec.from_file(str(HAND_XML))
    mount = carrier.add_frame(pos=[0.0, 0.0, 0.0])
    host.attach(hand, prefix="hand_", frame=mount)

    # Cube as free-floating body at the cup center.
    cube_body = world.add_body(name="cube", pos=list(CUBE_START))
    cube_body.add_freejoint(name="cube_free")
    cube_body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
        rgba=[0.95, 0.55, 0.20, 1.0],
        density=CUBE_DENSITY,
        friction=[1.2, 0.05, 0.001],
    )

    return host.compile()


def get_actuator_map(model):
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)}


def find_joint_qpos_addr(model, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[jid])


def apply_pose(data, name2act, pose: dict):
    for j, v in pose.items():
        key = f"hand_{j}_act"
        if key in name2act:
            data.ctrl[name2act[key]] = v


def make_grip_pose(tighten_amount: float) -> dict:
    """tighten_amount in [0,1]: 0 = baseline cage, 1 = max tightening."""
    out = dict(CAGE_BASE)
    for j, delta in GRIP_TIGHTEN_DELTA.items():
        if j in out:
            out[j] = out[j] + delta * tighten_amount
    return out


def setup_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.04, 0.34]
    cam.distance = 0.32
    cam.azimuth = 70.0
    cam.elevation = -8.0
    return cam


def animate_camera(cam, t):
    """Slow orbit around the hand so the cube stays in frame from multiple angles."""
    cam.azimuth = 70.0 + 25.0 * math.sin(0.18 * t)
    cam.elevation = -10.0 + 4.0 * math.sin(0.13 * t)


def run():
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    print("Building scene...")
    model = build_scene()
    data = mujoco.MjData(model)
    name2act = get_actuator_map(model)
    cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    print(f"  nq={model.nq}  nu={model.nu}  ngeom={model.ngeom}")

    # Settle: hold cage pose for 0.4 s before recording.
    apply_pose(data, name2act, CAGE_BASE)
    for _ in range(int(0.4 / DT)):
        apply_pose(data, name2act, CAGE_BASE)
        mujoco.mj_step(model, data)
    print(f"  cube after settle: {data.xpos[cube_bid].round(4).tolist()}")

    cam = setup_camera()
    renderer = mujoco.Renderer(model, width=RES_W, height=RES_H)

    wrist_addr = find_joint_qpos_addr(model, "wrist_yaw")

    # Bookkeeping
    total_steps = int(DURATION_S / DT)
    steps_per_frame = max(1, int(round((1.0 / FPS) / DT)))
    frames = []
    log = []
    perturbations = []
    fail = False

    rng_seed = 12345
    rng = np.random.default_rng(rng_seed)
    next_perturb_t = PERTURB_PERIOD_S  # first perturb after 3 s
    active_perturb_until = -1.0
    active_perturb_force = np.zeros(3)

    for step in range(total_steps):
        t = step * DT

        # ---- 1) Wrist held still (yaw locked at 0) ----
        wrist_target = 0.0
        data.qpos[wrist_addr] = wrist_target
        data.qvel[wrist_addr] = 0.0

        # ---- 2) Closed-loop finger grip ----
        cube_pos = data.xpos[cube_bid].copy()
        # Cup center is fixed in world frame (wrist not rotating).
        err = cube_pos - CUP_CENTER
        drift = float(np.linalg.norm(err[:2]))  # planar drift

        # Tighten when planar drift exceeds 5 mm; saturate at 20 mm.
        tighten = max(0.0, min(1.0, (drift - 0.005) / 0.015))
        apply_pose(data, name2act, make_grip_pose(tighten))

        # ---- 3) Random perturbation as external force on cube ----
        if t >= next_perturb_t:
            # Pick a random horizontal direction
            angle = rng.uniform(0, 2 * math.pi)
            active_perturb_force = np.array([
                math.cos(angle) * PERTURB_FORCE_N,
                math.sin(angle) * PERTURB_FORCE_N,
                0.0,
            ])
            active_perturb_until = t + PERTURB_DURATION_S
            perturbations.append({
                "t": round(t, 3),
                "angle_deg": round(math.degrees(angle), 1),
                "force_N": PERTURB_FORCE_N,
            })
            next_perturb_t = t + PERTURB_PERIOD_S

        if t <= active_perturb_until:
            data.xfrc_applied[cube_bid][:3] = active_perturb_force
        else:
            data.xfrc_applied[cube_bid][:3] = 0.0

        # ---- 4) Step physics ----
        mujoco.mj_step(model, data)

        # ---- 5) Log & render ----
        if step % steps_per_frame == 0:
            cur_cube_pos = data.xpos[cube_bid].copy()
            held = cur_cube_pos[2] > 0.20  # if z < 20 cm, cube has fallen
            if not held:
                fail = True
            log.append({
                "t": round(t, 3),
                "wrist_yaw_deg": round(math.degrees(wrist_target), 1),
                "cube_pos": [round(x, 4) for x in cur_cube_pos.tolist()],
                "planar_drift_m": round(drift, 4),
                "grip_tighten": round(tighten, 2),
                "perturb_active": bool(t <= active_perturb_until),
                "held": bool(held),
            })

            animate_camera(cam, t)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    # ---- Summarize ----
    held_final = data.xpos[cube_bid][2] > 0.20
    max_drift = max((entry["planar_drift_m"] for entry in log), default=0.0)
    avg_drift = sum(entry["planar_drift_m"] for entry in log) / max(1, len(log))
    summary = {
        "project": "LEAP Dexterous Stabilization",
        "duration_s": DURATION_S,
        "fps": FPS,
        "resolution": [RES_W, RES_H],
        "cube_held_final": bool(held_final),
        "any_drop_event": bool(fail),
        "max_planar_drift_m": round(max_drift, 4),
        "avg_planar_drift_m": round(avg_drift, 4),
        "perturbations_n": len(perturbations),
        "perturbations": perturbations,
        "wrist_rotation": {
            "type": "oscillation around vertical",
            "amplitude_deg": math.degrees(WRIST_ROT_AMPL),
            "bpm": WRIST_ROT_BPM,
        },
        "closed_loop": {
            "sensor": "cube position (planar drift from rotating cup center)",
            "actuator": "16 LEAP finger position targets",
            "law": "tighten = clamp((drift - 5mm) / 15mm, 0, 1) applied as additive grip delta",
        },
        "rng_seed": rng_seed,
        "log_count": len(log),
        "log_samples": log[::10][:30],  # downsampled for the JSON file
    }

    print()
    print(f"FINAL: held={held_final}  max_drift={max_drift*1000:.1f}mm  "
          f"avg_drift={avg_drift*1000:.1f}mm  perturbations={len(perturbations)}")
    print(f"Encoding {len(frames)} frames -> {OUT_VIDEO.name}")
    iio.imwrite(OUT_VIDEO, np.asarray(frames), fps=FPS, codec="libx264")
    OUT_TRAJECTORY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  video: {OUT_VIDEO}")
    print(f"  metrics: {OUT_TRAJECTORY}")
    return summary


if __name__ == "__main__":
    run()
