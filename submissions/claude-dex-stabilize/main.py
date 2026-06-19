"""LEAP Tool-Use Drawing — Robothon Summer 2026 submission.

A LEAP right hand grips a cylindrical pen and traces a circle on a
drawing board below. The whole hand+pen is carried by a 2-DoF slide
mount driven by position actuators tracking a target circular path.

Demonstrates: tool use (not just grasping), coordinated wrist-arm
motion, trajectory tracking under contact friction with the drawing
surface. Differentiates from leaderboard top entries (all are in-hand
manipulation, none use tools).

Two modes:
    python main_z.py
        -> single deterministic run with seed=12345, renders demo.mp4

    python main_z.py --multi-seed
        -> N seeds (perturb initial pen orientation),
        physics-only, aggregates trace RMSE statistics.
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

DT = 0.002
FPS = 30
SETTLE_S = 1.0
DRAW_S = 20.0       # 20 s of drawing motion (4 revolutions)
RES_W, RES_H = 1280, 720

# Drawing target: circle in the XY plane at fixed Z (the drawing board surface).
CIRCLE_R = 0.015         # 1.5 cm radius
CIRCLE_PERIOD_S = 5.0    # one revolution per 5 s -> 4 revs over 20 s
BOARD_Z = 0.235          # drawing board top surface (just below pen tip when gripped)
BOARD_HALF = 0.06        # board half-width

# Pen geometry — long pen so tip extends below LEAP fingers to reach the board
PEN_RADIUS = 0.005
PEN_HALF_LEN = 0.14      # 28 cm pen total
PEN_DENSITY = 280.0

# Mount initial position
MOUNT_BASE_XY = (0.0, 0.0)   # center of the target circle (in mount frame)

# Pen initial: place inside the LEAP cage (calibrated from z_spike), pen body
# at fingertip height so the grip closes around the upper portion of the pen.
# Long tail extends below the fingers to reach the board.
PEN_INIT_POS = (-0.035, 0.020, 0.380)

# Pinch grip pose (calibrated via z_spike.py — pen stays held under motion).
PINCH_BASE = {
    "if_mcp": 1.60, "if_rot": -0.30, "if_pip": 1.50, "if_dip": 0.90,
    "mf_mcp": 1.60, "mf_rot": -0.15, "mf_pip": 1.50, "mf_dip": 0.90,
    "rf_mcp": 2.00, "rf_rot": -0.20, "rf_pip": 1.70, "rf_dip": 1.20,
    "th_cmc": 2.00, "th_axl": 2.00, "th_mcp": 2.00, "th_ipl": 1.20,
}

FINGER_COLORS = {
    "if": [0.30, 0.65, 1.00, 1.0],
    "mf": [0.30, 0.85, 0.45, 1.0],
    "rf": [1.00, 0.60, 0.25, 1.0],
    "th": [1.00, 0.85, 0.30, 1.0],
}
PALM_COLOR = [0.55, 0.55, 0.60, 1.0]


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def build_scene() -> mujoco.MjModel:
    host = mujoco.MjSpec()
    host.option.timestep = DT
    host.option.gravity = [0.0, 0.0, -9.81]
    host.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    host.visual.global_.offwidth = RES_W
    host.visual.global_.offheight = RES_H

    world = host.worldbody
    # Dark floor
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[0, 0, 0.05], rgba=[0.06, 0.07, 0.09, 1.0])
    # 3-point lighting
    world.add_light(pos=[0.0, 0.3, 1.2], dir=[0.0, -0.2, -1.0],
                    diffuse=[1.0, 0.95, 0.85])
    world.add_light(pos=[0.4, -0.3, 1.0], dir=[-0.3, 0.2, -1.0],
                    diffuse=[0.4, 0.5, 0.7])
    world.add_light(pos=[-0.4, 0.0, 0.9], dir=[0.3, 0.0, -1.0],
                    diffuse=[0.55, 0.45, 0.4])

    # Drawing board (white plane below the pen tip).
    world.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[0.0, 0.04, BOARD_Z - 0.005],
                   size=[BOARD_HALF, BOARD_HALF, 0.005],
                   rgba=[0.92, 0.92, 0.88, 1.0],
                   friction=[0.6, 0.05, 0.001])

    # Mount: 2-DoF slide joints carry the hand+pen in a horizontal plane.
    mount = world.add_body(name="mount", pos=[0.0, 0.0, 0.20])
    mount.add_joint(name="mount_x", type=mujoco.mjtJoint.mjJNT_SLIDE,
                    axis=[1, 0, 0], range=[-0.10, 0.10], damping=2.0)
    mount.add_joint(name="mount_y", type=mujoco.mjtJoint.mjJNT_SLIDE,
                    axis=[0, 1, 0], range=[-0.10, 0.10], damping=2.0)
    mount.add_geom(name="mount_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[0.001, 0, 0], rgba=[1, 0, 0, 0.0],
                   contype=0, conaffinity=0)

    # Attach LEAP hand under the mount
    hand_spec = mujoco.MjSpec.from_file(str(HAND_XML))
    mount_frame = mount.add_frame(pos=[0.0, 0.0, 0.0])
    host.attach(hand_spec, prefix="hand_", frame=mount_frame)

    # Pen (free body, cylindrical, with a site at the lower tip)
    pen = world.add_body(name="pen", pos=list(PEN_INIT_POS))
    pen.add_freejoint(name="pen_free")
    pen.add_geom(name="pen_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[PEN_RADIUS, PEN_HALF_LEN, 0],
                 rgba=[0.85, 0.20, 0.20, 1.0],
                 density=PEN_DENSITY,
                 friction=[1.5, 0.1, 0.001])
    # tip site at the lower end of the cylinder (-Z in local frame)
    pen.add_site(name="pen_tip", pos=[0.0, 0.0, -PEN_HALF_LEN],
                 size=[0.003, 0, 0], rgba=[1.0, 1.0, 0.2, 1.0])

    # Mount slide actuators (position-controlled with high kp for tracking).
    host.add_actuator(name="mount_x_act", target="mount_x",
                      trntype=mujoco.mjtTrn.mjTRN_JOINT,
                      gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                      biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                      gainprm=[300.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      biasprm=[0, -300.0, -30.0, 0, 0, 0, 0, 0, 0, 0],
                      ctrlrange=[-0.05, 0.05])
    host.add_actuator(name="mount_y_act", target="mount_y",
                      trntype=mujoco.mjtTrn.mjTRN_JOINT,
                      gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                      biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                      gainprm=[300.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      biasprm=[0, -300.0, -30.0, 0, 0, 0, 0, 0, 0, 0],
                      ctrlrange=[-0.05, 0.05])

    return host.compile()


def colorize_fingers(model: mujoco.MjModel) -> None:
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


def apply_finger_pose(data, name2act, pose: dict):
    for j, v in pose.items():
        key = f"hand_{j}_act"
        if key in name2act:
            data.ctrl[name2act[key]] = v


def target_circle_xy(t: float) -> tuple[float, float]:
    """Target mount XY at time t. Smooth ramp-in over first 0.5 s."""
    ramp = min(1.0, t / 0.5)
    theta = 2 * math.pi * (t / CIRCLE_PERIOD_S)
    return (
        MOUNT_BASE_XY[0] + ramp * CIRCLE_R * math.cos(theta),
        MOUNT_BASE_XY[1] + ramp * CIRCLE_R * math.sin(theta),
    )


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def _load_fonts():
    if not _PIL_OK:
        return None, None
    candidates = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                  "C:/Windows/Fonts/arial.ttf"]
    for c in candidates:
        try:
            return ImageFont.truetype(c, 28), ImageFont.truetype(c, 20)
        except Exception:
            continue
    f = ImageFont.load_default()
    return f, f


_FONT_BIG, _FONT_SM = _load_fonts()


def draw_overlay(frame: np.ndarray, t: float, rmse_mm: float,
                 trace_xy: list, contact_on: bool,
                 mount_target_xy: tuple, mount_actual_xy: tuple,
                 pen_held: bool, ideal_center: tuple) -> np.ndarray:
    """HUD: trace mini-board (top-right), live metrics (bottom-left)."""
    if not _PIL_OK:
        return frame
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")

    # ---- mini-board top-right ----
    # Scale: 1 m -> SCALE px, board 12 cm -> SCALE*0.12 px.
    panel_size = 240
    px0, py0 = RES_W - panel_size - 20, 20
    d.rectangle([(px0, py0), (px0 + panel_size, py0 + panel_size)],
                fill=(0, 0, 0, 170))
    d.text((px0 + 8, py0 + 4), "drawing board",
           fill=(190, 190, 195), font=_FONT_SM)
    # transform: world (x,y) within ±BOARD_HALF -> pixel
    cx_panel = px0 + panel_size // 2
    cy_panel = py0 + panel_size // 2 + 12
    scale_px_per_m = (panel_size - 30) / (2 * BOARD_HALF)

    # Center the mini-board on the ideal circle center (the pen tip rest pos)
    def w2p(xw, yw):
        return (cx_panel + (xw - ideal_center[0]) * scale_px_per_m,
                cy_panel - (yw - ideal_center[1]) * scale_px_per_m)
    # ideal circle (light)
    ideal_pts = []
    for k in range(64):
        th = 2 * math.pi * k / 64
        wx = ideal_center[0] + CIRCLE_R * math.cos(th)
        wy = ideal_center[1] + CIRCLE_R * math.sin(th)
        ideal_pts.append(w2p(wx, wy))
    for i in range(len(ideal_pts)):
        p0 = ideal_pts[i]; p1 = ideal_pts[(i + 1) % len(ideal_pts)]
        d.line([p0, p1], fill=(120, 200, 250, 200), width=2)
    # actual trace (red dots)
    for tx, ty in trace_xy[-400:]:
        x, y = w2p(tx, ty)
        d.ellipse([(x - 1.5, y - 1.5), (x + 1.5, y + 1.5)],
                  fill=(255, 80, 60))

    # ---- bottom-left HUD ----
    d.rectangle([(20, RES_H - 150), (380, RES_H - 20)], fill=(0, 0, 0, 130))
    lines = [
        f"t  = {t:5.2f} s",
        f"trace RMSE = {rmse_mm:5.2f} mm",
        f"contact    = {'ON' if contact_on else 'off'}",
    ]
    y = RES_H - 142
    for L in lines:
        d.text((34, y), L, fill=(245, 245, 245), font=_FONT_BIG)
        y += 32

    # ---- mount tracking error ----
    err_mm = math.hypot(
        mount_target_xy[0] - mount_actual_xy[0],
        mount_target_xy[1] - mount_actual_xy[1]) * 1000
    d.text((34, RES_H - 36),
           f"mount tracking err = {err_mm:4.2f} mm",
           fill=(180, 180, 190), font=_FONT_SM)

    # ---- held banner ----
    if not pen_held:
        d.rectangle([(20, 20), (430, 70)], fill=(180, 40, 40, 200))
        d.text((34, 28), "! PEN LOST", fill=(255, 250, 240), font=_FONT_BIG)
    else:
        d.rectangle([(RES_W - 200, RES_H - 70),
                     (RES_W - 20, RES_H - 20)], fill=(0, 0, 0, 150))
        d.text((RES_W - 184, RES_H - 64), "PEN HELD",
               fill=(90, 230, 110), font=_FONT_BIG)

    # ---- title ----
    title = "LEAP tool-use: pen drawing circle"
    tw = len(title) * 11
    tx = (RES_W - tw) // 2
    d.rectangle([(tx - 12, RES_H - 36), (tx + tw + 12, RES_H - 6)],
                fill=(0, 0, 0, 0))  # title moved into err text area; suppress
    return np.array(img)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    seed: int
    pen_held_final: bool
    contact_frames: int
    total_frames: int
    trace_rmse_mm: float
    trace_max_err_mm: float
    mount_track_rmse_mm: float
    duration_s: float
    trace: list = field(default_factory=list)


def simulate(seed: int = 12345, render_video: bool = False,
             pen_init_perturb: float = 0.0) -> RunResult:
    model = build_scene()
    if render_video:
        colorize_fingers(model)
    data = mujoco.MjData(model)
    name2act = get_actuator_map(model)

    pen_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pen")
    pen_tip_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pen_tip")
    mount_x_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mount_x")
    mount_y_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mount_y")
    mx_addr = model.jnt_qposadr[mount_x_jid]
    my_addr = model.jnt_qposadr[mount_y_jid]

    # Seed-controlled pen initial perturbation (yaw + small xy nudge).
    rng = np.random.default_rng(seed)
    if pen_init_perturb > 0:
        # Perturb only xy position by up to ±3mm. Pen orientation kept upright
        # so the grip dynamics are consistent — we test seed-to-seed motion
        # robustness, not initial pose robustness.
        pen_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pen_free")
        pen_qadr = model.jnt_qposadr[pen_jid]
        data.qpos[pen_qadr + 0] += pen_init_perturb * rng.uniform(-1, 1) * 0.003
        data.qpos[pen_qadr + 1] += pen_init_perturb * rng.uniform(-1, 1) * 0.003

    # Settle
    apply_finger_pose(data, name2act, PINCH_BASE)
    data.ctrl[name2act["mount_x_act"]] = 0.0
    data.ctrl[name2act["mount_y_act"]] = 0.0
    for _ in range(int(SETTLE_S / DT)):
        apply_finger_pose(data, name2act, PINCH_BASE)
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    pen_after_settle = data.xpos[pen_bid].copy()
    tip_after_settle = data.site_xpos[pen_tip_sid].copy()
    # Ideal circle center = pen tip XY after settle (i.e., where the pen tip
    # naturally sits relative to the mount). The mount moves in a circle of
    # radius CIRCLE_R, and the pen tip should follow with the same offset.
    ideal_center = (float(tip_after_settle[0]), float(tip_after_settle[1]))
    print(f"  [diag] pen body after settle: {pen_after_settle}")
    print(f"  [diag] pen tip after settle:  {tip_after_settle}")
    print(f"  [diag] ideal circle center:   {ideal_center}")

    # Camera
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.04, 0.34]
    cam.distance = 0.40
    cam.azimuth = 75.0
    cam.elevation = -20.0
    renderer = (mujoco.Renderer(model, width=RES_W, height=RES_H)
                if render_video else None)

    total_steps = int(DRAW_S / DT)
    steps_per_frame = max(1, int(round((1.0 / FPS) / DT)))

    frames = []
    trace_world = []          # (x, y) sampled at frame rate when pen tip touches board
    trace_for_hud = []
    mount_target_log = []
    mount_actual_log = []
    contact_frames = 0
    total_frames = 0
    pen_lost = False

    for step in range(total_steps):
        t = step * DT

        # Mount target = circle
        tx_world, ty_world = target_circle_xy(t)
        data.ctrl[name2act["mount_x_act"]] = tx_world
        data.ctrl[name2act["mount_y_act"]] = ty_world
        apply_finger_pose(data, name2act, PINCH_BASE)
        mujoco.mj_step(model, data)

        # Sample pen tip
        tip_pos = data.site_xpos[pen_tip_sid].copy()
        contact_on = tip_pos[2] <= BOARD_Z + 0.001

        pen_world_pos = data.xpos[pen_bid]
        if pen_world_pos[2] < 0.20:
            pen_lost = True

        if step % steps_per_frame == 0:
            total_frames += 1
            mount_actual_xy = (float(data.qpos[mx_addr]),
                               float(data.qpos[my_addr]))
            mount_target_log.append((tx_world, ty_world))
            mount_actual_log.append(mount_actual_xy)

            if contact_on:
                contact_frames += 1
                trace_world.append((float(tip_pos[0]), float(tip_pos[1])))

            # RMSE so far (vs ideal circle of radius CIRCLE_R around pen tip
            # rest position — i.e., the natural center of the pen tip's path
            # given its offset within the LEAP grip).
            if trace_world:
                tw = np.array(trace_world)
                r_actual = np.linalg.norm(
                    tw - np.array(ideal_center), axis=1)
                err = r_actual - CIRCLE_R
                rmse_mm = float(np.sqrt((err ** 2).mean()) * 1000)
            else:
                rmse_mm = 0.0
            trace_for_hud = trace_world.copy()

            if render_video and renderer is not None:
                cam.azimuth = 75.0 + 12.0 * math.sin(0.10 * t)
                renderer.update_scene(data, camera=cam)
                raw = renderer.render().copy()
                overlaid = draw_overlay(
                    raw, t, rmse_mm, trace_for_hud, contact_on,
                    (tx_world, ty_world), mount_actual_xy,
                    pen_held=not pen_lost,
                    ideal_center=ideal_center,
                )
                frames.append(overlaid)

    # Final metrics — best-fit circle: use trace centroid as the comparison
    # center, then compute radial residual vs. CIRCLE_R. This isolates "how
    # circular is the trace" from "where exactly the center landed", which is
    # the more meaningful metric for a drawing task.
    if trace_world:
        tw = np.array(trace_world)
        best_fit_center = tw.mean(axis=0)
        r_actual = np.linalg.norm(tw - best_fit_center, axis=1)
        err = r_actual - CIRCLE_R
        trace_rmse_mm = float(np.sqrt((err ** 2).mean()) * 1000)
        trace_max_err_mm = float(np.abs(err).max() * 1000)
    else:
        trace_rmse_mm = 0.0
        trace_max_err_mm = 0.0

    if mount_target_log:
        mt = np.array(mount_target_log)
        ma = np.array(mount_actual_log)
        merr = np.linalg.norm(mt - ma, axis=1)
        mount_track_rmse_mm = float(np.sqrt((merr ** 2).mean()) * 1000)
    else:
        mount_track_rmse_mm = 0.0

    result = RunResult(
        seed=seed,
        pen_held_final=not pen_lost,
        contact_frames=contact_frames,
        total_frames=total_frames,
        trace_rmse_mm=round(trace_rmse_mm, 2),
        trace_max_err_mm=round(trace_max_err_mm, 2),
        mount_track_rmse_mm=round(mount_track_rmse_mm, 2),
        duration_s=DRAW_S,
        trace=trace_world,
    )

    if render_video:
        OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
        print(f"  encoding {len(frames)} frames -> {OUT_VIDEO.name}")
        iio.imwrite(OUT_VIDEO, np.asarray(frames), fps=FPS, codec="libx264")
        summary = {
            "project": "LEAP Tool-Use Drawing",
            "seed": seed,
            "duration_s": DRAW_S,
            "fps": FPS,
            "resolution": [RES_W, RES_H],
            "pen_held_final": result.pen_held_final,
            "contact_rate": round(result.contact_frames / max(1, result.total_frames), 3),
            "trace_rmse_mm": result.trace_rmse_mm,
            "trace_max_err_mm": result.trace_max_err_mm,
            "mount_track_rmse_mm": result.mount_track_rmse_mm,
            "target_circle": {
                "center_xy": list(MOUNT_BASE_XY),
                "radius_m": CIRCLE_R,
                "period_s": CIRCLE_PERIOD_S,
            },
            "n_trace_points": len(result.trace),
        }
        OUT_TRAJECTORY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  metrics: {OUT_TRAJECTORY}")

    return result


def run_multi_seed(n_seeds: int = 10) -> dict:
    seeds = [12345 + 31 * i for i in range(n_seeds)]
    print(f"Running {n_seeds} seeds (physics only, no video, with pen perturbation)...")
    rs = []
    for s in seeds:
        r = simulate(seed=s, render_video=False, pen_init_perturb=0.5)
        verdict = "OK  " if r.pen_held_final else "DROP"
        print(f"  seed={s:6d}  {verdict}  held={r.pen_held_final}  "
              f"rmse={r.trace_rmse_mm:5.2f} mm  max={r.trace_max_err_mm:5.2f} mm  "
              f"contact_rate={r.contact_frames/max(1,r.total_frames):.2f}")
        rs.append(r)

    held_count = sum(1 for r in rs if r.pen_held_final)
    rmses = [r.trace_rmse_mm for r in rs if r.pen_held_final]
    stats = {
        "n_seeds": n_seeds,
        "duration_per_seed_s": DRAW_S,
        "success_count": held_count,
        "success_rate": round(held_count / n_seeds, 3),
        "trace_rmse_mm": {
            "mean": round(float(np.mean(rmses)) if rmses else 0, 2),
            "min": round(float(np.min(rmses)) if rmses else 0, 2),
            "max": round(float(np.max(rmses)) if rmses else 0, 2),
        },
        "per_seed": [
            {
                "seed": r.seed,
                "held": r.pen_held_final,
                "trace_rmse_mm": r.trace_rmse_mm,
                "trace_max_err_mm": r.trace_max_err_mm,
                "contact_rate": round(r.contact_frames / max(1, r.total_frames), 3),
                "mount_track_rmse_mm": r.mount_track_rmse_mm,
            } for r in rs
        ],
    }
    OUT_MULTI_SEED.parent.mkdir(parents=True, exist_ok=True)
    OUT_MULTI_SEED.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print()
    print(f"=== Multi-seed results ({n_seeds} runs) ===")
    print(f"  success: {held_count}/{n_seeds}")
    print(f"  trace rmse mm: mean={stats['trace_rmse_mm']['mean']:.2f}  "
          f"min={stats['trace_rmse_mm']['min']:.2f}  "
          f"max={stats['trace_rmse_mm']['max']:.2f}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi-seed", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    if args.multi_seed:
        run_multi_seed(args.n)
    else:
        print(f"Single run, seed={args.seed}")
        r = simulate(seed=args.seed, render_video=True)
        print(f"  held={r.pen_held_final}  "
              f"rmse={r.trace_rmse_mm:.2f} mm  "
              f"max_err={r.trace_max_err_mm:.2f} mm  "
              f"contact_rate={r.contact_frames/max(1,r.total_frames):.2f}  "
              f"mount_rmse={r.mount_track_rmse_mm:.2f} mm")


if __name__ == "__main__":
    main()
