"""LEAP Dexterous Stabilization — Robothon Summer 2026 submission.

A LEAP right hand holds an orange cube in its cage grasp. External horizontal
perturbations are applied to the cube at random intervals. A closed-loop
finger-grip controller reads the cube's planar drift each step and tightens
finger position targets when drift exceeds a threshold.

Demonstrates: real `mj_step` physics, real contact, real perturbation
rejection — not a scripted timeline.

Two modes:
    python submissions/claude-dex-stabilize/main.py
        -> single deterministic run with seed=12345, renders demo.mp4 + JSON

    python submissions/claude-dex-stabilize/main.py --multi-seed
        -> ten different seeds, physics-only, aggregates robustness stats
        into multi_seed_stats.json (no video).
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
DURATION_S = 20.0
RES_W, RES_H = 1280, 720

CUBE_HALF = 0.018
CUBE_DENSITY = 200.0
CUBE_START = (-0.022, 0.020, 0.346)
CUP_CENTER = np.array(CUBE_START)

PERTURB_PERIOD_S = 3.0
PERTURB_FORCE_N = 0.8
PERTURB_DURATION_S = 0.15

CAGE_BASE = {
    "if_mcp": 1.30, "if_rot": 0.10, "if_pip": 1.00, "if_dip": 0.60,
    "mf_mcp": 1.30, "mf_rot": 0.00, "mf_pip": 1.00, "mf_dip": 0.60,
    "rf_mcp": 1.30, "rf_rot": -0.10, "rf_pip": 1.00, "rf_dip": 0.60,
    "th_cmc": 1.20, "th_axl": 0.80, "th_mcp": 1.20, "th_ipl": 1.00,
}
GRIP_TIGHTEN_DELTA = {
    "if_mcp": 0.15, "if_pip": 0.20, "if_dip": 0.15,
    "mf_mcp": 0.15, "mf_pip": 0.20, "mf_dip": 0.15,
    "rf_mcp": 0.15, "rf_pip": 0.20, "rf_dip": 0.15,
    "th_cmc": 0.10, "th_mcp": 0.15, "th_ipl": 0.15,
}


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

FINGER_COLORS = {
    "if": [0.30, 0.65, 1.00, 1.0],   # blue   — index
    "mf": [0.30, 0.85, 0.45, 1.0],   # green  — middle
    "rf": [1.00, 0.60, 0.25, 1.0],   # orange — ring
    "th": [1.00, 0.85, 0.30, 1.0],   # yellow — thumb
}
PALM_COLOR = [0.55, 0.55, 0.60, 1.0]  # neutral gray
TIP_HILITE_COLORS = {
    "if": [0.55, 0.85, 1.00, 1.0],
    "mf": [0.55, 1.00, 0.65, 1.0],
    "rf": [1.00, 0.80, 0.45, 1.0],
    "th": [1.00, 1.00, 0.50, 1.0],
}


def colorize_fingers(model: mujoco.MjModel) -> None:
    """Recolor LEAP geoms per finger so each finger is visually distinct."""
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not bname.startswith("hand_"):
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        rest = bname[len("hand_"):]
        # palm
        if rest.startswith("palm"):
            model.geom_rgba[gid] = PALM_COLOR
            continue
        # finger segments — first 2 letters identify finger
        fcode = rest[:2]
        if fcode in FINGER_COLORS:
            # Tip geoms get a slightly brighter hilite so contact points pop.
            if gname.endswith("_tip"):
                model.geom_rgba[gid] = TIP_HILITE_COLORS[fcode]
            else:
                model.geom_rgba[gid] = FINGER_COLORS[fcode]


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
    world.add_light(pos=[0.0, 0.3, 1.2], dir=[0.0, -0.2, -1.0],
                    diffuse=[1.0, 0.95, 0.85])
    world.add_light(pos=[0.4, -0.3, 1.0], dir=[-0.3, 0.2, -1.0],
                    diffuse=[0.4, 0.5, 0.7])
    world.add_light(pos=[-0.4, 0.0, 0.9], dir=[0.3, 0.0, -1.0],
                    diffuse=[0.55, 0.45, 0.4])

    hand = mujoco.MjSpec.from_file(str(HAND_XML))
    mount = world.add_frame(pos=[0.0, 0.0, 0.20])
    host.attach(hand, prefix="hand_", frame=mount)

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


def apply_pose(data, name2act, pose: dict):
    for j, v in pose.items():
        key = f"hand_{j}_act"
        if key in name2act:
            data.ctrl[name2act[key]] = v


def make_grip_pose(tighten_amount: float) -> dict:
    out = dict(CAGE_BASE)
    for j, delta in GRIP_TIGHTEN_DELTA.items():
        if j in out:
            out[j] = out[j] + delta * tighten_amount
    return out


# ---------------------------------------------------------------------------
# Overlay HUD (PIL)
# ---------------------------------------------------------------------------

def _load_fonts():
    """Try several common system fonts; fallback to PIL default."""
    if not _PIL_OK:
        return None, None
    candidates = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                  "Helvetica.ttf", "C:/Windows/Fonts/arial.ttf"]
    for c in candidates:
        try:
            return ImageFont.truetype(c, 28), ImageFont.truetype(c, 20)
        except Exception:
            continue
    f = ImageFont.load_default()
    return f, f


_FONT_BIG, _FONT_SM = _load_fonts()


def _draw_perturb_arrow(d, angle_rad: float):
    """Red force-vector arrow originating from cube screen center."""
    cx, cy = RES_W // 2, RES_H // 2 + 20  # rough cube screen position
    L = 130
    # World-frame angle: +X right, +Y up (we map screen Y inversely)
    ax = cx + L * math.cos(angle_rad)
    ay = cy - L * math.sin(angle_rad)
    d.line([(cx, cy), (ax, ay)], fill=(255, 70, 70), width=7)
    head = 18
    tip = math.atan2(ay - cy, ax - cx)
    la = tip + math.radians(150)
    ra = tip - math.radians(150)
    d.polygon([
        (ax, ay),
        (ax + head * math.cos(la), ay + head * math.sin(la)),
        (ax + head * math.cos(ra), ay + head * math.sin(ra)),
    ], fill=(255, 70, 70))


def _draw_drift_chart(d, drift_history_mm: list):
    """Bottom-right strip showing drift-over-time line plot (last ~5 s)."""
    x0, y0, w, h = RES_W - 270, RES_H - 200, 250, 90
    d.rectangle([(x0, y0), (x0 + w, y0 + h)], fill=(0, 0, 0, 160))
    d.text((x0 + 8, y0 + 4), "drift over time (mm)",
           fill=(190, 190, 195), font=_FONT_SM)
    if len(drift_history_mm) < 2:
        return
    # show last 5 s = 150 frames at 30 fps
    window = drift_history_mm[-150:]
    ymax = max(8.0, max(window) * 1.15)
    pts = []
    for i, v in enumerate(window):
        px = x0 + 6 + (w - 12) * i / max(1, len(window) - 1)
        py = (y0 + h - 6) - (h - 28) * (v / ymax)
        pts.append((px, py))
    # baseline at 5mm threshold (where grip starts tightening)
    y_thresh = (y0 + h - 6) - (h - 28) * (5.0 / ymax)
    d.line([(x0 + 6, y_thresh), (x0 + w - 6, y_thresh)],
           fill=(120, 120, 120, 180), width=1)
    d.text((x0 + w - 60, y_thresh - 16), "5 mm",
           fill=(150, 150, 150), font=_FONT_SM)
    # line plot
    for i in range(len(pts) - 1):
        d.line([pts[i], pts[i + 1]], fill=(120, 200, 250), width=2)


def draw_overlay(frame: np.ndarray, t: float, drift_m: float,
                 grip_tighten: float, perturb_on: bool, held: bool,
                 perturb_count: int,
                 drift_history_mm: list | None = None,
                 perturb_angle_rad: float | None = None) -> np.ndarray:
    if not _PIL_OK:
        return frame
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")

    # ---- force-vector arrow (when perturb active) ----
    if perturb_on and perturb_angle_rad is not None:
        _draw_perturb_arrow(d, perturb_angle_rad)

    # ---- left HUD: time, drift, grip ----
    d.rectangle([(20, RES_H - 130), (380, RES_H - 20)], fill=(0, 0, 0, 130))
    lines = [
        f"t  = {t:5.2f} s",
        f"drift = {drift_m*1000:5.1f} mm",
        f"grip tighten = {grip_tighten:.2f}",
    ]
    y = RES_H - 122
    for L in lines:
        d.text((34, y), L, fill=(245, 245, 245), font=_FONT_BIG)
        y += 32

    # ---- top-right: held status ----
    status_color = (90, 230, 110) if held else (255, 90, 90)
    status_text = "HOLD" if held else "DROP"
    d.rectangle([(RES_W - 200, 20), (RES_W - 20, 70)], fill=(0, 0, 0, 150))
    d.text((RES_W - 184, 26), status_text, fill=status_color, font=_FONT_BIG)
    d.text((RES_W - 184, 56), "cube", fill=(200, 200, 200), font=_FONT_SM)

    # ---- top-left: perturbation banner ----
    if perturb_on:
        d.rectangle([(20, 20), (430, 70)], fill=(180, 40, 40, 200))
        ang_txt = (f"  @ {math.degrees(perturb_angle_rad):4.0f}°"
                   if perturb_angle_rad is not None else "")
        d.text((34, 28), f"! PERTURBATION #{perturb_count}{ang_txt}",
               fill=(255, 250, 240), font=_FONT_BIG)

    # ---- finger color legend (bottom-left of chart area) ----
    legend = [("index", FINGER_COLORS["if"]),
              ("middle", FINGER_COLORS["mf"]),
              ("ring", FINGER_COLORS["rf"]),
              ("thumb", FINGER_COLORS["th"])]
    lx, ly = RES_W - 270, RES_H - 100
    d.rectangle([(lx, ly), (lx + 250, ly + 90)], fill=(0, 0, 0, 160))
    d.text((lx + 8, ly + 4), "fingers", fill=(190, 190, 195), font=_FONT_SM)
    for i, (label, color) in enumerate(legend):
        sx = lx + 12 + (i % 2) * 120
        sy = ly + 28 + (i // 2) * 28
        rgb = tuple(int(c * 255) for c in color[:3])
        d.rectangle([(sx, sy), (sx + 18, sy + 18)], fill=rgb)
        d.text((sx + 24, sy - 2), label, fill=(220, 220, 220), font=_FONT_SM)

    # ---- drift over time chart ----
    if drift_history_mm:
        _draw_drift_chart(d, drift_history_mm)

    # ---- bottom-center title ----
    title = "LEAP closed-loop stabilization"
    # rough width estimate
    tw = len(title) * 11
    tx = (RES_W - tw) // 2
    d.rectangle([(tx - 12, RES_H - 36), (tx + tw + 12, RES_H - 6)],
                fill=(0, 0, 0, 130))
    d.text((tx, RES_H - 30), title, fill=(220, 220, 220), font=_FONT_SM)

    return np.array(img)


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    seed: int
    held_final: bool
    any_drop: bool
    max_drift_m: float
    avg_drift_m: float
    perturbations_n: int
    duration_s: float
    perturbations: list = field(default_factory=list)
    log_samples: list = field(default_factory=list)


def simulate(seed: int = 12345, render_video: bool = False) -> RunResult:
    """Run one stabilization episode. If render_video=True, also produces
    OUT_VIDEO and OUT_TRAJECTORY."""
    model = build_scene()
    if render_video:
        colorize_fingers(model)  # only when we actually want the colored video
    data = mujoco.MjData(model)
    name2act = get_actuator_map(model)
    cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")

    # settle
    apply_pose(data, name2act, CAGE_BASE)
    for _ in range(int(0.4 / DT)):
        apply_pose(data, name2act, CAGE_BASE)
        mujoco.mj_step(model, data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.04, 0.34]
    cam.distance = 0.32
    cam.azimuth = 70.0
    cam.elevation = -8.0
    renderer = mujoco.Renderer(model, width=RES_W, height=RES_H) if render_video else None

    rng = np.random.default_rng(seed)
    total_steps = int(DURATION_S / DT)
    steps_per_frame = max(1, int(round((1.0 / FPS) / DT)))

    frames = []
    log_samples = []
    perturbations = []
    fail = False

    next_perturb_t = PERTURB_PERIOD_S
    active_perturb_until = -1.0
    active_perturb_force = np.zeros(3)
    active_perturb_angle: float | None = None
    perturb_count = 0
    drift_history = []
    drift_history_mm_per_frame = []  # for the on-screen chart

    for step in range(total_steps):
        t = step * DT

        cube_pos = data.xpos[cube_bid].copy()
        err = cube_pos - CUP_CENTER
        drift = float(np.linalg.norm(err[:2]))
        drift_history.append(drift)

        tighten = max(0.0, min(1.0, (drift - 0.005) / 0.015))
        apply_pose(data, name2act, make_grip_pose(tighten))

        if t >= next_perturb_t:
            angle = rng.uniform(0, 2 * math.pi)
            active_perturb_force = np.array([
                math.cos(angle) * PERTURB_FORCE_N,
                math.sin(angle) * PERTURB_FORCE_N,
                0.0,
            ])
            active_perturb_until = t + PERTURB_DURATION_S
            active_perturb_angle = angle
            perturb_count += 1
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

        mujoco.mj_step(model, data)

        if step % steps_per_frame == 0:
            cur_cube_pos = data.xpos[cube_bid].copy()
            held = cur_cube_pos[2] > 0.20
            if not held:
                fail = True
            perturb_on = t <= active_perturb_until

            if render_video and renderer is not None:
                cam.azimuth = 70.0 + 25.0 * math.sin(0.18 * t)
                cam.elevation = -10.0 + 4.0 * math.sin(0.13 * t)
                renderer.update_scene(data, camera=cam)
                raw = renderer.render().copy()
                drift_history_mm_per_frame.append(drift * 1000)
                overlaid = draw_overlay(
                    raw, t, drift, tighten, perturb_on, held, perturb_count,
                    drift_history_mm=drift_history_mm_per_frame,
                    perturb_angle_rad=active_perturb_angle if perturb_on else None,
                )
                frames.append(overlaid)

            if step % (steps_per_frame * 5) == 0:
                log_samples.append({
                    "t": round(t, 3),
                    "cube_pos": [round(x, 4) for x in cur_cube_pos.tolist()],
                    "planar_drift_m": round(drift, 4),
                    "grip_tighten": round(tighten, 2),
                    "perturb_active": bool(perturb_on),
                    "held": bool(held),
                })

    held_final = bool(data.xpos[cube_bid][2] > 0.20)
    max_drift = max(drift_history) if drift_history else 0.0
    avg_drift = sum(drift_history) / len(drift_history) if drift_history else 0.0

    result = RunResult(
        seed=seed,
        held_final=held_final,
        any_drop=bool(fail),
        max_drift_m=round(max_drift, 5),
        avg_drift_m=round(avg_drift, 5),
        perturbations_n=len(perturbations),
        duration_s=DURATION_S,
        perturbations=perturbations,
        log_samples=log_samples,
    )

    if render_video:
        OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
        print(f"  encoding {len(frames)} frames -> {OUT_VIDEO.name}")
        iio.imwrite(OUT_VIDEO, np.asarray(frames), fps=FPS, codec="libx264")

        summary = {
            "project": "LEAP Dexterous Stabilization",
            "seed": seed,
            "duration_s": DURATION_S,
            "fps": FPS,
            "resolution": [RES_W, RES_H],
            "cube_held_final": result.held_final,
            "any_drop_event": result.any_drop,
            "max_planar_drift_m": result.max_drift_m,
            "avg_planar_drift_m": result.avg_drift_m,
            "perturbations_n": result.perturbations_n,
            "perturbations": result.perturbations,
            "closed_loop": {
                "sensor": "cube position (planar drift from cup center)",
                "actuator": "16 LEAP finger position targets",
                "law": "tighten = clamp((drift - 5mm) / 15mm, 0, 1) applied as additive grip delta",
            },
            "log_samples": result.log_samples,
        }
        OUT_TRAJECTORY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  metrics: {OUT_TRAJECTORY}")

    return result


# ---------------------------------------------------------------------------
# Multi-seed robustness runner
# ---------------------------------------------------------------------------

def run_multi_seed(n_seeds: int = 10) -> dict:
    seeds = [12345 + 31 * i for i in range(n_seeds)]
    print(f"Running {n_seeds} seeds (physics only, no video)...")
    rs = []
    for s in seeds:
        r = simulate(seed=s, render_video=False)
        verdict = "HELD" if r.held_final and not r.any_drop else "DROP"
        print(f"  seed={s:6d}  {verdict:4s}  max_drift={r.max_drift_m*1000:6.2f} mm  "
              f"avg_drift={r.avg_drift_m*1000:6.2f} mm  "
              f"perturb={r.perturbations_n}")
        rs.append(r)

    held_count = sum(1 for r in rs if r.held_final and not r.any_drop)
    success_rate = held_count / len(rs)
    max_drifts = [r.max_drift_m for r in rs]
    avg_drifts = [r.avg_drift_m for r in rs]
    stats = {
        "n_seeds": n_seeds,
        "duration_per_seed_s": DURATION_S,
        "success_count": held_count,
        "success_rate": round(success_rate, 3),
        "max_drift_mm": {
            "mean": round(sum(max_drifts) / len(max_drifts) * 1000, 2),
            "min": round(min(max_drifts) * 1000, 2),
            "max": round(max(max_drifts) * 1000, 2),
        },
        "avg_drift_mm": {
            "mean": round(sum(avg_drifts) / len(avg_drifts) * 1000, 2),
            "min": round(min(avg_drifts) * 1000, 2),
            "max": round(max(avg_drifts) * 1000, 2),
        },
        "per_seed": [
            {
                "seed": r.seed,
                "held": r.held_final and not r.any_drop,
                "max_drift_mm": round(r.max_drift_m * 1000, 2),
                "avg_drift_mm": round(r.avg_drift_m * 1000, 2),
                "perturbations": r.perturbations_n,
            } for r in rs
        ],
    }
    OUT_MULTI_SEED.parent.mkdir(parents=True, exist_ok=True)
    OUT_MULTI_SEED.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print()
    print(f"=== Multi-seed results ({n_seeds} runs) ===")
    print(f"  success: {held_count}/{n_seeds} ({success_rate*100:.0f}%)")
    print(f"  max drift mm:  mean={stats['max_drift_mm']['mean']:.2f}  "
          f"min={stats['max_drift_mm']['min']:.2f}  max={stats['max_drift_mm']['max']:.2f}")
    print(f"  avg drift mm:  mean={stats['avg_drift_mm']['mean']:.2f}  "
          f"min={stats['avg_drift_mm']['min']:.2f}  max={stats['avg_drift_mm']['max']:.2f}")
    print(f"  stats saved -> {OUT_MULTI_SEED.name}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi-seed", action="store_true",
                    help="Run N=10 seeds, physics only, output multi_seed_stats.json")
    ap.add_argument("--n", type=int, default=10, help="N for --multi-seed (default 10)")
    ap.add_argument("--seed", type=int, default=12345, help="Seed for single-run mode")
    args = ap.parse_args()

    if args.multi_seed:
        run_multi_seed(args.n)
    else:
        print(f"Single run, seed={args.seed}")
        r = simulate(seed=args.seed, render_video=True)
        print(f"  held={r.held_final}  max_drift={r.max_drift_m*1000:.2f} mm  "
              f"avg_drift={r.avg_drift_m*1000:.2f} mm  perturbations={r.perturbations_n}")


if __name__ == "__main__":
    main()
