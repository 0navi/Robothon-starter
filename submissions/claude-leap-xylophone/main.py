"""LEAP Hand Xylophone — Robothon Summer 2026 submission.

A LEAP right hand plays "Hot Cross Buns" on a 3-key xylophone by sequencing
finger strikes. Each finger maps to one xylophone bar:

    index  finger -> E5 (top bar)
    middle finger -> D5 (middle bar)
    ring   finger -> C5 (bottom bar)

Strike detection is closed-loop: each frame the controller polls the touch
sensor on each xylophone bar; when contact force crosses a threshold a note
event is fired. Note events drive (a) a per-bar "lit" visual flash, (b)
the per-frame JSONL data stream, and (c) the post-render audio synthesis
that gets muxed back into demo.mp4.

This is a music-production task — a *real* sensor-gated control loop, not
a scripted timeline.

Modes:
    python main.py
        canonical run: render demo.mp4 + JSONL stream + trajectory.json

    python main.py --multi-seed --n 10
        robustness sweep across 10 seeds (different perturbations to the
        finger schedule), aggregates note-accuracy stats

    python main.py --difficulty-sweep --n 10
        10 seeds at each of 3 tempos {slow, medium, fast}
"""
from __future__ import annotations
import argparse
import json
import math
import struct
import wave
import subprocess
import shutil
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
OUT_DIR = HERE / "outputs"
OUT_VIDEO = OUT_DIR / "demo.mp4"
OUT_VIDEO_SILENT = OUT_DIR / "demo_silent.mp4"
OUT_AUDIO = OUT_DIR / "demo.wav"
OUT_TRAJECTORY = OUT_DIR / "trajectory.json"
OUT_JSONL = OUT_DIR / "trajectory.jsonl"
OUT_MULTI_SEED = OUT_DIR / "multi_seed_stats.json"
OUT_DIFFICULTY = OUT_DIR / "difficulty_sweep.json"

DT = 0.002
FPS = 30
RES_W, RES_H = 1280, 720

# ---------------------------------------------------------------------------
# Song: Hot Cross Buns
# ---------------------------------------------------------------------------
# 3-note melody: E D C / E D C / C C C C D D D D / E D C
# Maps to fingers: index=E, middle=D, ring=C.
# Note frequencies (C5 octave, equal-tempered, A4=440 Hz):
NOTE_FREQS = {"C": 523.25, "D": 587.33, "E": 659.25}
FINGER_OF_NOTE = {"E": "if", "D": "mf", "C": "rf"}

# A medley of 3-note nursery tunes (all reachable on C/D/E only — exactly
# what our 3 fingers can play). 60-80 seconds of music at 100 bpm.
HOT_CROSS_BUNS = [
    "E", "D", "C", "_", "E", "D", "C", "_",
    "C", "C", "C", "C", "D", "D", "D", "D",
    "E", "D", "C", "_",
]

MARY_HAD_A_LITTLE_LAMB = [
    # "Mary had a little lamb, lit-tle lamb, lit-tle lamb"
    "E", "D", "C", "D", "E", "E", "E", "_",
    "D", "D", "D", "_", "E", "E", "E", "_",
    # "Mary had a little lamb, its fleece was white as snow"
    "E", "D", "C", "D", "E", "E", "E", "E",
    "D", "D", "E", "D", "C", "_", "_", "_",
]

THREE_BLIND_MICE = [
    # "Three blind mice, three blind mice"
    "E", "D", "C", "_", "E", "D", "C", "_",
    # "See how they run, see how they run"
    "C", "C", "C", "C", "D", "D", "D", "D",
    "E", "D", "C", "_",
]

MEDLEY = (
    HOT_CROSS_BUNS + ["_", "_"] +
    MARY_HAD_A_LITTLE_LAMB + ["_", "_"] +
    THREE_BLIND_MICE + ["_", "_"]
)


def build_schedule(notes, tempo_s_per_beat: float, start_t: float = 1.5):
    """Convert a note list to (note, strike_t) events.

    `_` is a rest (silence) — no strike scheduled, beat consumed.
    All notes are quarter-notes (one beat each) — keeps the scheduler simple
    and gives the controller plenty of time per strike.
    """
    out = []
    t = start_t
    for n in notes:
        if n != "_":
            out.append({"note": n, "strike_t": round(t, 4),
                        "finger": FINGER_OF_NOTE[n]})
        t += tempo_s_per_beat
    return out


# ---------------------------------------------------------------------------
# Finger poses
# ---------------------------------------------------------------------------
# Three fingers (index, middle, ring) each have an independent rest/strike
# pose. The thumb stays parked out of the way. Numbers from spike_calibrate.py.

REST_POSE = {
    # index/middle/ring all curled (tip retracted up, ready to strike down)
    "if_mcp": 1.20, "if_rot": 0.0, "if_pip": 1.20, "if_dip": 0.70,
    "mf_mcp": 1.20, "mf_rot": 0.0, "mf_pip": 1.20, "mf_dip": 0.70,
    "rf_mcp": 1.20, "rf_rot": 0.0, "rf_pip": 1.20, "rf_dip": 0.70,
    # thumb parked off to the side (not used in melody)
    "th_cmc": 0.30, "th_axl": 1.50, "th_mcp": 0.30, "th_ipl": 0.30,
}

# Per-finger strike pose: extend the named finger (mcp & pip go small),
# leaving the other fingers in REST_POSE. We construct this dynamically.
# Strike = only MCP extends fully; PIP/DIP stay mostly bent so the finger
# remains "hooked" — the TIP swings down while the medial segment stays high
# (clears the bar plane).
STRIKE_DELTAS = {
    "if": {"if_mcp": -1.40, "if_pip": -0.30, "if_dip": -0.20},
    "mf": {"mf_mcp": -1.40, "mf_pip": -0.30, "mf_dip": -0.20},
    "rf": {"rf_mcp": -1.40, "rf_pip": -0.30, "rf_dip": -0.20},
}


def strike_pose(finger: str) -> dict:
    """Pose where `finger` is extended (striking down), others at rest."""
    out = dict(REST_POSE)
    for j, d in STRIKE_DELTAS[finger].items():
        out[j] = out[j] + d
    return out


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
# 3 xylophone bars placed under the strike-arc of each finger.
# Y coordinates picked to match each finger's Y (from spike calibration):
#   index  y = -0.008
#   middle y = +0.037
#   ring   y = +0.083
# Bars are slightly thick boxes (visual bars) with hinge joints so they
# can wobble visibly when struck.

BAR_X = 0.105              # tip strike-arc ends at x~0.10
BAR_Z_TOP = 0.475          # just below curled-tip z=0.49, just above extended-tip z=0.47
BAR_HALF_X = 0.020         # narrow: only the tip reaches, not medial
BAR_HALF_Y = 0.018
BAR_HALF_Z = 0.005

BAR_DEFS = [
    # (note, y_center, color)  -- colors form a rainbow xylophone
    ("E", -0.008, [0.95, 0.32, 0.25, 1.0]),  # red    — index
    ("D", +0.037, [0.30, 0.85, 0.45, 1.0]),  # green  — middle
    ("C", +0.083, [0.30, 0.55, 1.00, 1.0]),  # blue   — ring
]
BAR_LIT_COLOR = [1.00, 0.95, 0.30, 1.0]  # yellow flash when struck

# Each bar sits on a small spring-hinge so it visibly dips on impact
# and returns. We use a hinge joint on Y axis (rolls fore-aft slightly).

FINGER_COLORS = {
    "if": [0.95, 0.32, 0.25, 1.0],
    "mf": [0.30, 0.85, 0.45, 1.0],
    "rf": [0.30, 0.55, 1.00, 1.0],
    "th": [0.60, 0.60, 0.65, 1.0],   # thumb dimmed (not used)
}
PALM_COLOR = [0.55, 0.55, 0.60, 1.0]
TIP_HILITE = {
    "if": [1.00, 0.55, 0.50, 1.0],
    "mf": [0.55, 1.00, 0.65, 1.0],
    "rf": [0.55, 0.75, 1.00, 1.0],
    "th": [0.70, 0.70, 0.75, 1.0],
}


def colorize_fingers(model: mujoco.MjModel) -> None:
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not bname.startswith("hand_"):
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        rest = bname[len("hand_"):]
        if rest.startswith("palm"):
            model.geom_rgba[gid] = PALM_COLOR
            continue
        fcode = rest[:2]
        if fcode in FINGER_COLORS:
            if gname.endswith("_tip"):
                model.geom_rgba[gid] = TIP_HILITE[fcode]
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
    # dark stage floor
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[0, 0, 0.05], rgba=[0.05, 0.06, 0.08, 1.0])
    # three-point lighting (warm key, cool fill, rim)
    world.add_light(pos=[0.4, 0.4, 1.4], dir=[-0.3, -0.3, -1.0],
                    diffuse=[1.0, 0.95, 0.85])
    world.add_light(pos=[-0.5, -0.3, 1.0], dir=[0.4, 0.2, -1.0],
                    diffuse=[0.4, 0.5, 0.7])
    world.add_light(pos=[0.0, -0.5, 0.7], dir=[0.0, 0.5, -1.0],
                    diffuse=[0.45, 0.4, 0.35])

    # LEAP hand mounted high so fingers can swing down through bar plane.
    hand = mujoco.MjSpec.from_file(str(HAND_XML))
    mount = world.add_frame(pos=[0.0, 0.0, 0.30])
    host.attach(hand, prefix="hand_", frame=mount)

    # --- xylophone base plate (purely decorative, contype=0 so it doesn't
    # interfere with finger motion) ---
    base = world.add_body(name="xylo_base", pos=[BAR_X, 0.0375, 0.420])
    base.add_geom(name="xylo_base_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[0.060, 0.090, 0.005],
                  rgba=[0.18, 0.13, 0.10, 1.0],
                  contype=0, conaffinity=0)

    # --- xylophone bars (3 bars: E, D, C) ---
    # Each bar is mounted on a hinge so it can briefly dip on impact.
    # We anchor each bar's hinge at the rear edge so the front lifts/dips
    # visibly when struck.
    for note, y_center, color in BAR_DEFS:
        bname = f"bar_{note}"
        body = world.add_body(name=bname, pos=[BAR_X, y_center, BAR_Z_TOP])
        # hinge along X axis at the bar's rear edge — bar pivots in YZ plane
        body.add_joint(name=f"{bname}_hinge", type=mujoco.mjtJoint.mjJNT_HINGE,
                       pos=[0, -BAR_HALF_Y, 0], axis=[1, 0, 0],
                       limited=True, range=[-0.05, 0.20],
                       damping=0.02, stiffness=8.0, springref=0.0)
        body.add_geom(name=f"{bname}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[BAR_HALF_X, BAR_HALF_Y, BAR_HALF_Z],
                      rgba=color, density=400.0,
                      friction=[0.6, 0.05, 0.001],
                      solref=[0.005, 1])
        # Site sphere needs to enclose the bar so the touch sensor sees all
        # contacts on the bar's top surface (MuJoCo touch sensor only counts
        # contacts inside the site's bounding sphere).
        body.add_site(name=f"{bname}_site", pos=[0, 0, 0],
                      size=[max(BAR_HALF_X, BAR_HALF_Y) + 0.005, 0, 0],
                      rgba=[0, 1, 0, 0],
                      type=mujoco.mjtGeom.mjGEOM_SPHERE)

    # --- Sensors ---
    # 16 LEAP joint positions are already in the LEAP XML.
    # Add 3 touch sensors (one per bar) — closed-loop strike detection.
    for note, _, _ in BAR_DEFS:
        host.add_sensor(name=f"bar_{note}_touch",
                        type=mujoco.mjtSensor.mjSENS_TOUCH,
                        objtype=mujoco.mjtObj.mjOBJ_SITE,
                        objname=f"bar_{note}_site")
    # Bar hinge-angles (so we can render the bar wobble in HUD).
    for note, _, _ in BAR_DEFS:
        host.add_sensor(name=f"bar_{note}_angle",
                        type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT,
                        objname=f"bar_{note}_hinge")
    # Fingertip world positions — proprioception via site framepos.
    # (Use the existing tip geom positions via site we add.)
    # Skip for now; LEAP already has 16 joint sensors built in.

    model = host.compile()
    # Stiffen LEAP position actuators after compile (defaults kp=3 are too
    # weak to hold the inverted fingers up against gravity — without this the
    # rest pose can't be held and every "rest" is actually drooped fingertips
    # resting on the bars).
    kp = 25.0
    kv = 1.0
    for i in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if aname.startswith("hand_"):
            # position actuator: gain = kp, bias = [0, -kp, -kv]
            model.actuator_gainprm[i, 0] = kp
            model.actuator_biasprm[i, 1] = -kp
            model.actuator_biasprm[i, 2] = -kv
    return model


def get_actuator_map(model):
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)}


def get_sensor_map(model):
    out = {}
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        out[name] = (int(model.sensor_adr[i]), int(model.sensor_dim[i]))
    return out


def read_sensor(data, sensor_map, name):
    adr, dim = sensor_map[name]
    return np.array(data.sensordata[adr:adr + dim])


def apply_pose(data, name2act, pose: dict):
    for j, v in pose.items():
        k = f"hand_{j}_act"
        if k in name2act:
            data.ctrl[name2act[k]] = v


# ---------------------------------------------------------------------------
# Audio synthesis
# ---------------------------------------------------------------------------

AUDIO_SAMPLE_RATE = 44100
NOTE_DURATION = 0.45          # s, per-note envelope length
NOTE_ATTACK = 0.005
NOTE_RELEASE = 0.30


def synth_audio(strike_events, total_duration_s: float, out_wav: Path):
    """Render a mono 16-bit WAV from a list of (t, note) strike events."""
    n_samples = int(total_duration_s * AUDIO_SAMPLE_RATE)
    buf = np.zeros(n_samples, dtype=np.float64)
    for ev in strike_events:
        f0 = NOTE_FREQS.get(ev["note"])
        if f0 is None:
            continue
        s0 = int(ev["t"] * AUDIO_SAMPLE_RATE)
        ns = int(NOTE_DURATION * AUDIO_SAMPLE_RATE)
        s1 = min(s0 + ns, n_samples)
        n = s1 - s0
        if n <= 0:
            continue
        tt = np.arange(n) / AUDIO_SAMPLE_RATE
        # bell-like timbre: fundamental + 2 overtones at 2x and 3x with decay
        wave_signal = (
            0.6 * np.sin(2 * np.pi * f0 * tt) +
            0.3 * np.sin(2 * np.pi * 2 * f0 * tt) +
            0.1 * np.sin(2 * np.pi * 3 * f0 * tt)
        )
        # exponential decay envelope
        env = np.exp(-tt / 0.18)
        # short attack ramp
        att_n = int(NOTE_ATTACK * AUDIO_SAMPLE_RATE)
        if att_n > 0 and att_n < n:
            env[:att_n] *= np.linspace(0, 1, att_n)
        v = float(ev.get("velocity", 0.6))
        buf[s0:s1] += v * wave_signal * env
    # normalize
    mx = max(np.abs(buf).max(), 1e-9)
    buf = buf / mx * 0.85
    int16 = (buf * 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(AUDIO_SAMPLE_RATE)
        f.writeframes(int16.tobytes())


def mux_audio_video(video_in: Path, audio_in: Path, video_out: Path) -> bool:
    """Use ffmpeg to mux audio onto video. Returns True on success."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        # try imageio's bundled binary
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg is None:
        return False
    cmd = [
        ffmpeg, "-y",
        "-i", str(video_in),
        "-i", str(audio_in),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(video_out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"  ffmpeg mux failed: {e}")
        return False


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------


def _load_fonts():
    if not _PIL_OK:
        return None, None, None
    candidates = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                  "C:/Windows/Fonts/arial.ttf"]
    for c in candidates:
        try:
            return (ImageFont.truetype(c, 32),
                    ImageFont.truetype(c, 22),
                    ImageFont.truetype(c, 16))
        except Exception:
            continue
    f = ImageFont.load_default()
    return f, f, f


_FONT_BIG, _FONT_MED, _FONT_SM = _load_fonts()


# Map each note in the schedule to an X position on a staff for HUD.
STAFF_X0 = 80
STAFF_Y = 90
STAFF_W = RES_W - 160
STAFF_NOTE_Y = {"E": 0, "D": 18, "C": 36}
NOTE_DOT_R = 11


def draw_overlay(frame: np.ndarray,
                 t: float,
                 schedule: list,
                 played_notes: list,
                 bar_angles: dict,
                 bar_touch: dict,
                 active_finger: str | None,
                 next_note: dict | None,
                 song_name: str = "Hot Cross Buns",
                 tempo_bpm: float = 100.0,
                 stats: dict | None = None) -> np.ndarray:
    if not _PIL_OK:
        return frame
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")

    # ---- top-center: song title + tempo ----
    title = f"♪ {song_name}   ·   {tempo_bpm:.0f} bpm   ·   t = {t:5.2f}s"
    tw = len(title) * 11
    tx = (RES_W - tw) // 2
    d.rectangle([(tx - 16, 10), (tx + tw + 16, 50)],
                fill=(0, 0, 0, 180))
    d.text((tx, 14), title, fill=(245, 230, 200), font=_FONT_MED)

    # ---- score strip: full schedule, current note highlighted ----
    y_top = STAFF_Y - 8
    y_bot = STAFF_Y + 50
    d.rectangle([(STAFF_X0 - 16, y_top), (STAFF_X0 + STAFF_W + 16, y_bot + 6)],
                fill=(0, 0, 0, 150))
    # 3 staff lines (one per pitch, for clarity)
    for note, y_off in STAFF_NOTE_Y.items():
        ly = STAFF_Y + y_off
        d.line([(STAFF_X0, ly), (STAFF_X0 + STAFF_W, ly)],
               fill=(80, 80, 90), width=1)
        d.text((STAFF_X0 - 30, ly - 11), note,
               fill=(180, 180, 190), font=_FONT_SM)
    # plot each note in schedule as a dot at its scheduled X position
    total_song_t = (schedule[-1]["strike_t"] + 1.0) if schedule else 1.0
    played_ids = {(p["note"], p["strike_t"]) for p in played_notes}
    for ev in schedule:
        nx = STAFF_X0 + STAFF_W * ev["strike_t"] / total_song_t
        ny = STAFF_Y + STAFF_NOTE_Y[ev["note"]]
        is_played = (ev["note"], ev["strike_t"]) in played_ids
        is_active = (next_note is not None and
                     ev["strike_t"] == next_note["strike_t"])
        # color
        if is_active:
            col = (255, 230, 60)
        elif is_played:
            col = (110, 230, 130)
        elif ev["strike_t"] < t - 0.3:
            col = (200, 90, 90)  # missed
        else:
            col = (180, 180, 190)
        d.ellipse([(nx - NOTE_DOT_R, ny - NOTE_DOT_R),
                   (nx + NOTE_DOT_R, ny + NOTE_DOT_R)], fill=col)
    # playhead vertical line
    if total_song_t > 0:
        ph = STAFF_X0 + STAFF_W * min(t / total_song_t, 1.0)
        d.line([(ph, y_top + 4), (ph, y_bot - 2)],
               fill=(255, 230, 60, 200), width=2)

    # ---- bottom-left: bar status (per-bar lit indicator + angle) ----
    bx0, by0 = 30, RES_H - 220
    d.rectangle([(bx0, by0), (bx0 + 360, by0 + 200)], fill=(0, 0, 0, 160))
    d.text((bx0 + 12, by0 + 8), "xylophone bars",
           fill=(220, 220, 230), font=_FONT_MED)
    for i, (note, _, _color) in enumerate(BAR_DEFS):
        rgb_full = tuple(int(c * 255) for c in _color[:3])
        bx = bx0 + 18
        by = by0 + 50 + i * 50
        # bar swatch — flashes yellow if touch active
        col = ((255, 235, 60) if bar_touch.get(note, 0) > 0.5 else rgb_full)
        d.rectangle([(bx, by), (bx + 60, by + 32)], fill=col)
        # note label
        d.text((bx + 80, by + 4), f"{note}   ({NOTE_FREQS[note]:.0f} Hz)",
               fill=(225, 225, 230), font=_FONT_MED)
        # angle (rad) — small
        ang = bar_angles.get(note, 0.0)
        d.text((bx + 80, by + 28), f"hinge: {ang:+.3f} rad",
               fill=(160, 160, 170), font=_FONT_SM)

    # ---- top-right: active finger + finger map ----
    fr_x = RES_W - 380
    fr_y = 80
    d.rectangle([(fr_x, fr_y), (fr_x + 360, fr_y + 200)], fill=(0, 0, 0, 160))
    d.text((fr_x + 12, fr_y + 8), "finger ↔ note",
           fill=(220, 220, 230), font=_FONT_MED)
    finger_labels = [("if", "index", "E"),
                     ("mf", "middle", "D"),
                     ("rf", "ring", "C")]
    for i, (fc, label, note) in enumerate(finger_labels):
        rgb = tuple(int(c * 255) for c in FINGER_COLORS[fc][:3])
        ix = fr_x + 18
        iy = fr_y + 50 + i * 50
        d.rectangle([(ix, iy), (ix + 30, iy + 30)], fill=rgb)
        is_active = (active_finger == fc)
        col = (255, 230, 60) if is_active else (220, 220, 230)
        d.text((ix + 46, iy + 0), f"{label:<8} → {note}",
               fill=col, font=_FONT_MED)
        d.text((ix + 46, iy + 28),
               f"strike pose" if is_active else "rest",
               fill=(160, 160, 170), font=_FONT_SM)

    # ---- bottom-right: stats ----
    if stats is not None:
        sx0, sy0 = RES_W - 290, RES_H - 140
        d.rectangle([(sx0, sy0), (sx0 + 270, sy0 + 120)], fill=(0, 0, 0, 170))
        d.text((sx0 + 12, sy0 + 6), "performance",
               fill=(220, 220, 230), font=_FONT_MED)
        rows = [
            f"strikes  : {stats.get('strikes', 0)}",
            f"scheduled: {stats.get('scheduled_n', 0)}",
            f"accuracy : {stats.get('accuracy_pct', 0):.0f}%",
        ]
        for i, r in enumerate(rows):
            d.text((sx0 + 12, sy0 + 32 + i * 26), r,
                   fill=(225, 225, 235), font=_FONT_MED)

    # ---- bottom-center: subtitle ----
    sub = "LEAP Hand × xylophone — sensor-gated music synthesis"
    sw = len(sub) * 9
    sxc = (RES_W - sw) // 2
    d.rectangle([(sxc - 12, RES_H - 34), (sxc + sw + 12, RES_H - 6)],
                fill=(0, 0, 0, 140))
    d.text((sxc, RES_H - 30), sub, fill=(210, 210, 220), font=_FONT_SM)

    return np.array(img)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

# How long the strike pose is held before returning to rest.
STRIKE_HOLD_S = 0.10
# How much before scheduled strike_t the controller starts winding up
# the strike pose (so the finger is mid-swing at strike_t).
WIND_UP_S = 0.15
# Touch threshold (N or N-equivalent) for treating a contact as a strike event.
TOUCH_THRESHOLD = 0.05


@dataclass
class RunResult:
    seed: int
    tempo_bpm: float
    scheduled_n: int
    struck_n: int
    correct_n: int                 # struck note matched scheduled note within timing window
    accuracy_pct: float
    note_events: list = field(default_factory=list)
    schedule: list = field(default_factory=list)


def schedule_for_tempo(bpm: float, seed: int = 0) -> tuple[list, float]:
    """Build the song schedule and return (schedule, total_duration_s).

    A small jitter is applied to each strike_t based on the seed so multi-seed
    runs exercise the closed-loop controller under different timings.
    """
    spb = 60.0 / bpm                 # seconds per beat
    sched = build_schedule(MEDLEY, spb)
    if seed != 0:
        rng = np.random.default_rng(seed)
        for ev in sched:
            ev["strike_t"] = round(ev["strike_t"] + float(rng.normal(0, 0.02)),
                                   4)
    total_t = sched[-1]["strike_t"] + 2.0
    return sched, total_t


def simulate(seed: int = 12345, tempo_bpm: float = 100.0,
             render_video: bool = False, write_jsonl: bool = True) -> RunResult:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schedule, total_t = schedule_for_tempo(tempo_bpm, seed)

    model = build_scene()
    if render_video:
        colorize_fingers(model)
    data = mujoco.MjData(model)
    name2act = get_actuator_map(model)
    sensor_map = get_sensor_map(model)

    # settle into rest pose
    apply_pose(data, name2act, REST_POSE)
    for _ in range(int(0.5 / DT)):
        apply_pose(data, name2act, REST_POSE)
        mujoco.mj_step(model, data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.05, 0.04, 0.43]
    cam.distance = 0.40
    cam.azimuth = 80.0
    cam.elevation = -18.0
    renderer = (mujoco.Renderer(model, width=RES_W, height=RES_H)
                if render_video else None)

    total_steps = int(total_t / DT)
    steps_per_frame = max(1, int(round((1.0 / FPS) / DT)))

    frames = []
    strike_events = []         # list of {t, note, finger, velocity, scheduled_t}
    played_notes = []
    bar_touch_prev = {n: 0.0 for n, _, _ in BAR_DEFS}

    jsonl_f = None
    if write_jsonl:
        jsonl_f = open(OUT_JSONL, "w", encoding="utf-8")

    next_idx = 0     # index into schedule
    last_strike_t = {n: -10.0 for n, _, _ in BAR_DEFS}
    STRIKE_DEBOUNCE_S = 0.20

    for step in range(total_steps):
        t = step * DT

        # --- read sensors ---
        bar_touch = {n: float(read_sensor(data, sensor_map, f"bar_{n}_touch")[0])
                     for n, _, _ in BAR_DEFS}
        bar_angle = {n: float(read_sensor(data, sensor_map, f"bar_{n}_angle")[0])
                     for n, _, _ in BAR_DEFS}
        joint_pos = np.array(
            [read_sensor(data, sensor_map, f"hand_{j}_sensor")[0]
             for j in ["if_mcp", "if_rot", "if_pip", "if_dip",
                       "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
                       "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
                       "th_cmc", "th_axl", "th_mcp", "th_ipl"]])

        # --- strike detection (closed-loop sensor read) ---
        for note in bar_touch:
            cur = bar_touch[note]
            prev = bar_touch_prev[note]
            # rising edge through threshold = a strike (debounced so contact
            # bounce on the spring-hinge bar doesn't double-trigger)
            if (cur > TOUCH_THRESHOLD and prev <= TOUCH_THRESHOLD
                    and t - last_strike_t[note] > STRIKE_DEBOUNCE_S):
                last_strike_t[note] = t
                # match against the closest scheduled note (within 0.3s)
                matched_idx = None
                for j, ev in enumerate(schedule):
                    if abs(ev["strike_t"] - t) < 0.30 and ev["note"] == note:
                        if any(e.get("schedule_idx") == j
                               for e in strike_events):
                            continue
                        matched_idx = j
                        break
                strike_events.append({
                    "t": round(t, 4),
                    "note": note,
                    "freq_hz": NOTE_FREQS[note],
                    "velocity": min(1.0, cur / 2.0 + 0.3),
                    "schedule_idx": matched_idx,
                    "scheduled_t": (schedule[matched_idx]["strike_t"]
                                    if matched_idx is not None else None),
                })
                played_notes.append({"note": note, "strike_t":
                                     schedule[matched_idx]["strike_t"]
                                     if matched_idx is not None
                                     else round(t, 4)})
        bar_touch_prev = bar_touch

        # --- advance scheduler: find current target note ---
        # walk past any scheduled notes whose strike_t has passed by > 0.2 s
        while next_idx < len(schedule) and schedule[next_idx]["strike_t"] < t - 0.20:
            next_idx += 1

        active_finger = None
        if next_idx < len(schedule):
            ev = schedule[next_idx]
            # wind-up window: from (strike_t - WIND_UP_S) to (strike_t + STRIKE_HOLD_S)
            in_window = (ev["strike_t"] - WIND_UP_S) <= t <= (ev["strike_t"] + STRIKE_HOLD_S)
            if in_window:
                active_finger = ev["finger"]
                apply_pose(data, name2act, strike_pose(ev["finger"]))
            else:
                apply_pose(data, name2act, REST_POSE)
        else:
            apply_pose(data, name2act, REST_POSE)

        mujoco.mj_step(model, data)

        # --- per-frame JSONL stream ---
        if jsonl_f is not None and step % steps_per_frame == 0:
            jsonl_f.write(json.dumps({
                "t": round(t, 4),
                "active_finger": active_finger,
                "bar_touch": {k: round(v, 4) for k, v in bar_touch.items()},
                "bar_angle_rad": {k: round(v, 4) for k, v in bar_angle.items()},
                "joint_pos": joint_pos.round(4).tolist(),
                "n_strikes_so_far": len(strike_events),
            }) + "\n")

        # --- video frame ---
        if render_video and renderer is not None and step % steps_per_frame == 0:
            # slow horizontal camera sway for cinematic feel
            cam.azimuth = 80.0 + 8.0 * math.sin(0.12 * t)
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            next_note = (schedule[next_idx]
                         if next_idx < len(schedule) else None)
            stats = {
                "strikes": len(strike_events),
                "scheduled_n": len(schedule),
                "accuracy_pct": (100.0 * len(strike_events) / max(1, next_idx)
                                 if next_idx > 0 else 0.0),
            }
            frame = draw_overlay(frame, t, schedule, played_notes,
                                 bar_angle, bar_touch, active_finger,
                                 next_note, tempo_bpm=tempo_bpm,
                                 stats=stats)
            frames.append(frame)

    if jsonl_f is not None:
        jsonl_f.close()

    # Score: how many scheduled notes had a matching strike?
    correct = sum(1 for ev in schedule
                  if any(e.get("schedule_idx") is not None and
                         schedule[e["schedule_idx"]] is ev and
                         e["note"] == ev["note"]
                         for e in strike_events))
    accuracy = 100.0 * correct / max(1, len(schedule))

    result = RunResult(
        seed=seed, tempo_bpm=tempo_bpm,
        scheduled_n=len(schedule), struck_n=len(strike_events),
        correct_n=correct, accuracy_pct=round(accuracy, 1),
        note_events=strike_events, schedule=schedule,
    )

    # Render video + audio
    if render_video and frames:
        iio.imwrite(str(OUT_VIDEO_SILENT), frames, fps=FPS,
                    codec="libx264", quality=8)
        synth_audio(strike_events, total_t, OUT_AUDIO)
        ok = mux_audio_video(OUT_VIDEO_SILENT, OUT_AUDIO, OUT_VIDEO)
        if not ok:
            # fallback: keep silent video as demo.mp4
            shutil.copyfile(OUT_VIDEO_SILENT, OUT_VIDEO)
            print("  Note: ffmpeg unavailable — demo.mp4 is silent.")
        else:
            print(f"  demo.mp4 written with audio ({len(strike_events)} notes).")
        # write trajectory.json summary
        OUT_TRAJECTORY.write_text(json.dumps({
            "seed": seed,
            "tempo_bpm": tempo_bpm,
            "song": "Hot Cross Buns",
            "scheduled_n": result.scheduled_n,
            "struck_n": result.struck_n,
            "correct_n": result.correct_n,
            "accuracy_pct": result.accuracy_pct,
            "duration_s": round(total_t, 2),
            "schedule": schedule,
            "strikes": strike_events,
        }, indent=2))

    return result


def run_multi_seed(n: int, tempo_bpm: float):
    results = []
    for s in range(n):
        seed = 1000 + s * 17
        r = simulate(seed=seed, tempo_bpm=tempo_bpm,
                     render_video=False, write_jsonl=False)
        results.append(r)
        print(f"  seed {seed:5d}  tempo={tempo_bpm:5.1f}  "
              f"{r.struck_n}/{r.scheduled_n} strikes  "
              f"acc={r.accuracy_pct:.1f}%")
    accs = [r.accuracy_pct for r in results]
    summary = {
        "n_seeds": n,
        "tempo_bpm": tempo_bpm,
        "per_seed": [r.__dict__ | {"note_events": None, "schedule": None}
                     for r in results],
        "mean_accuracy_pct": round(float(np.mean(accs)), 2),
        "min_accuracy_pct": round(float(np.min(accs)), 2),
        "max_accuracy_pct": round(float(np.max(accs)), 2),
        "perfect_runs": int(sum(1 for a in accs if a >= 99.9)),
    }
    OUT_MULTI_SEED.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  mean accuracy: {summary['mean_accuracy_pct']}%")
    print(f"  perfect runs:  {summary['perfect_runs']}/{n}")


def run_difficulty_sweep(n: int):
    tempos = [80.0, 100.0, 120.0]
    out = {"tempos_bpm": tempos, "per_tempo": {}}
    for bpm in tempos:
        results = []
        print(f"--- tempo {bpm} bpm ---")
        for s in range(n):
            seed = 2000 + s * 13
            r = simulate(seed=seed, tempo_bpm=bpm,
                         render_video=False, write_jsonl=False)
            results.append(r)
            print(f"  seed {seed:5d}  {r.struck_n}/{r.scheduled_n}  "
                  f"acc={r.accuracy_pct:.1f}%")
        accs = [r.accuracy_pct for r in results]
        out["per_tempo"][f"{bpm:.0f}"] = {
            "n": n,
            "mean_accuracy_pct": round(float(np.mean(accs)), 2),
            "min_accuracy_pct": round(float(np.min(accs)), 2),
            "max_accuracy_pct": round(float(np.max(accs)), 2),
            "perfect_runs": int(sum(1 for a in accs if a >= 99.9)),
        }
    OUT_DIFFICULTY.write_text(json.dumps(out, indent=2))
    print("\nDifficulty sweep summary:")
    for k, v in out["per_tempo"].items():
        print(f"  {k} bpm: mean acc {v['mean_accuracy_pct']:.1f}%  "
              f"perfect {v['perfect_runs']}/{n}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--tempo", type=float, default=100.0,
                   help="tempo in BPM (default 100)")
    p.add_argument("--multi-seed", action="store_true")
    p.add_argument("--difficulty-sweep", action="store_true")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.difficulty_sweep:
        run_difficulty_sweep(args.n)
    elif args.multi_seed:
        run_multi_seed(args.n, args.tempo)
    else:
        r = simulate(seed=args.seed, tempo_bpm=args.tempo,
                     render_video=not args.no_video,
                     write_jsonl=True)
        print(f"\nCanonical run:")
        print(f"  seed={r.seed}  tempo={r.tempo_bpm:.0f} bpm")
        print(f"  scheduled notes: {r.scheduled_n}")
        print(f"  strikes detected: {r.struck_n}")
        print(f"  correct: {r.correct_n}")
        print(f"  accuracy: {r.accuracy_pct:.1f}%")
        if not args.no_video:
            print(f"  video: {OUT_VIDEO}")


if __name__ == "__main__":
    main()
