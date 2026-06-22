"""LEAP Hand Xylophone v2 — Robothon Summer 2026 submission.

A LEAP right hand plays Twinkle Twinkle Little Star (and Ode to Joy
opening) on an 8-bar xylophone tuned to the C-major scale. The hand is
mounted on a mocap wrist that slides along Y between strikes, so the
index finger lands on the right bar each beat. Strikes are detected via
per-bar touch sensors; every audio sample in `demo.mp4` is triggered by
a real fingertip↔bar contact event.

Modes:
    python main.py                          canonical render
    python main.py --multi-seed --n 10      robustness across 10 seeds
    python main.py --difficulty-sweep --n 10  10 seeds × 3 tempos
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
import subprocess
import wave
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
# Notes & scale
# ---------------------------------------------------------------------------
# 8-bar C major scale (C5 .. C6) sounding at equal-temperament frequencies.
NOTE_FREQS = {
    "C": 523.25,  "D": 587.33,  "E": 659.25,  "F": 698.46,
    "G": 783.99,  "A": 880.00,  "B": 987.77,  "c": 1046.50,
}
NOTE_ORDER = ["C", "D", "E", "F", "G", "A", "B", "c"]
NOTE_INDEX = {n: i for i, n in enumerate(NOTE_ORDER)}

# Every note is struck by the same finger (index); the hand's Y position
# selects which bar the strike lands on.
STRIKE_FINGER = "if"

# ---------------------------------------------------------------------------
# Songs
# ---------------------------------------------------------------------------
# Notation: a list of (note, beats) tuples. `_` = rest. Lowercase c = C6.
TWINKLE_TWINKLE = [
    # "Twinkle twinkle little star"
    ("C", 1), ("C", 1), ("G", 1), ("G", 1), ("A", 1), ("A", 1), ("G", 2),
    # "How I wonder what you are"
    ("F", 1), ("F", 1), ("E", 1), ("E", 1), ("D", 1), ("D", 1), ("C", 2),
    # "Up above the world so high"
    ("G", 1), ("G", 1), ("F", 1), ("F", 1), ("E", 1), ("E", 1), ("D", 2),
    # "Like a diamond in the sky"
    ("G", 1), ("G", 1), ("F", 1), ("F", 1), ("E", 1), ("E", 1), ("D", 2),
    # "Twinkle twinkle little star"
    ("C", 1), ("C", 1), ("G", 1), ("G", 1), ("A", 1), ("A", 1), ("G", 2),
    # "How I wonder what you are"
    ("F", 1), ("F", 1), ("E", 1), ("E", 1), ("D", 1), ("D", 1), ("C", 2),
]

ODE_TO_JOY = [
    # "Joy, joyful, joyful we adore thee"  (first phrase)
    ("E", 1), ("E", 1), ("F", 1), ("G", 1),
    ("G", 1), ("F", 1), ("E", 1), ("D", 1),
    ("C", 1), ("C", 1), ("D", 1), ("E", 1),
    ("E", 1), ("D", 1), ("D", 2),
    # second phrase
    ("E", 1), ("E", 1), ("F", 1), ("G", 1),
    ("G", 1), ("F", 1), ("E", 1), ("D", 1),
    ("C", 1), ("C", 1), ("D", 1), ("E", 1),
    ("D", 1), ("C", 1), ("C", 2),
]

MARY_HAD_A_LITTLE_LAMB = [
    # "Mary had a little lamb, little lamb, little lamb"
    ("E", 1), ("D", 1), ("C", 1), ("D", 1),
    ("E", 1), ("E", 1), ("E", 2),
    ("D", 1), ("D", 1), ("D", 2),
    ("E", 1), ("G", 1), ("G", 2),
    # "Mary had a little lamb, its fleece was white as snow"
    ("E", 1), ("D", 1), ("C", 1), ("D", 1),
    ("E", 1), ("E", 1), ("E", 1), ("E", 1),
    ("D", 1), ("D", 1), ("E", 1), ("D", 1),
    ("C", 2), ("C", 2),
]

HAPPY_BIRTHDAY = [
    # Simplified to all quarter-notes — the dotted-eighth + sixteenth pickup
    # in the canonical rhythm is faster than our strike-debounce window.
    # Same melody, robotic but recognizable.
    # "Hap-py birth-day to you"
    ("C", 1), ("C", 1), ("D", 1), ("C", 1), ("F", 1), ("E", 2),
    # "Hap-py birth-day to you"
    ("C", 1), ("C", 1), ("D", 1), ("C", 1), ("G", 1), ("F", 2),
    # "Hap-py birth-day dear (name)"    — uses high C ('c') and A
    ("C", 1), ("C", 1), ("c", 1), ("A", 1), ("F", 1), ("E", 1), ("D", 2),
    # "Hap-py birth-day to you"         — uses B
    ("B", 1), ("B", 1), ("A", 1), ("F", 1), ("G", 1), ("F", 2),
]

# The concert program: ordered list of (title, song, tempo_bpm) tuples.
# Each piece gets its own tempo so the show breathes.
CONCERT_PROGRAM = [
    ("Mary Had a Little Lamb",        MARY_HAD_A_LITTLE_LAMB, 110.0),
    ("Twinkle Twinkle Little Star",   TWINKLE_TWINKLE,        110.0),
    ("Ode to Joy",                    ODE_TO_JOY,             100.0),
    ("Happy Birthday",                HAPPY_BIRTHDAY,         105.0),
]

INTRO_DURATION_S = 3.0      # opening title card
INTER_PIECE_S = 1.8         # silence between pieces (programme transitions)
OUTRO_DURATION_S = 3.5      # closing title + bow hold


def build_schedule(song, tempo_s_per_beat: float, start_t: float = 0.0):
    """Convert a (note, beats) list to a list of strike events."""
    out = []
    t = start_t
    for n, beats in song:
        if n != "_":
            out.append({
                "note": n,
                "note_idx": NOTE_INDEX[n],
                "strike_t": round(t, 4),
                "duration_s": round(tempo_s_per_beat * beats, 4),
            })
        t += tempo_s_per_beat * beats
    return out, t


def make_concert_schedule(tempo_scale: float = 1.0):
    """Build the full concert: 4 pieces with announced titles and pauses.

    Returns (schedule, pieces, total_duration_s) where:
      schedule: flat list of strike events with `piece_idx`
      pieces:   list of {title, start_t, end_t, tempo_bpm, n_notes}
      total_duration_s: includes intro + inter-piece pauses + outro
    """
    schedule = []
    pieces = []
    t = INTRO_DURATION_S
    for piece_idx, (title, notes, bpm) in enumerate(CONCERT_PROGRAM):
        bpm_scaled = bpm * tempo_scale
        spb = 60.0 / bpm_scaled
        piece_start = t
        events, t = build_schedule(notes, spb, start_t=t)
        for ev in events:
            ev["piece_idx"] = piece_idx
        schedule.extend(events)
        pieces.append({
            "title": title,
            "start_t": round(piece_start, 3),
            "end_t": round(t, 3),
            "tempo_bpm": round(bpm_scaled, 1),
            "n_notes": len(events),
        })
        # inter-piece pause (except after the last piece — outro handles that)
        if piece_idx < len(CONCERT_PROGRAM) - 1:
            t += INTER_PIECE_S
    total = t + OUTRO_DURATION_S
    return schedule, pieces, total


# ---------------------------------------------------------------------------
# Finger poses (only the index finger strikes; the other three stay parked)
# ---------------------------------------------------------------------------
REST_POSE = {
    "if_mcp": 1.20, "if_rot": 0.0, "if_pip": 1.20, "if_dip": 0.70,
    # park middle / ring / thumb out of the way
    "mf_mcp": 1.50, "mf_rot": 0.0, "mf_pip": 1.50, "mf_dip": 0.80,
    "rf_mcp": 1.50, "rf_rot": 0.0, "rf_pip": 1.50, "rf_dip": 0.80,
    "th_cmc": 0.30, "th_axl": 1.50, "th_mcp": 0.30, "th_ipl": 0.30,
}
# Strike pose: extend INDEX only (MCP big swing, PIP/DIP stay bent so the
# tip arcs in while the medial section clears the bar plane).
STRIKE_POSE = dict(REST_POSE)
STRIKE_POSE["if_mcp"] = -0.20
STRIKE_POSE["if_pip"] = 0.90
STRIKE_POSE["if_dip"] = 0.50


# ---------------------------------------------------------------------------
# Xylophone geometry
# ---------------------------------------------------------------------------
# 8 bars laid out along Y (rainbow). The index fingertip in strike pose lands
# at world x ≈ 0.085, y ≈ wrist_y - 0.008 (offset from the wrist), z ≈ 0.476.
BAR_X = 0.105
BAR_Z_TOP = 0.475
BAR_HALF_X = 0.020
BAR_HALF_Y = 0.014
BAR_HALF_Z = 0.005
BAR_Y_SPACING = 0.032          # 3.2 cm between adjacent bar centers
BAR_Y_FIRST = -0.07            # Y of bar 0 (C5)

def bar_y(note_index: int) -> float:
    """World Y of bar `note_index` (0..7)."""
    return BAR_Y_FIRST + note_index * BAR_Y_SPACING

# Wrist Y target so the index fingertip lands on a given bar. The fingertip
# is offset from wrist by ≈ −0.008 in Y, so wrist_y = bar_y + 0.008.
WRIST_TIP_Y_OFFSET = -0.008

def wrist_y_for_note(note_index: int) -> float:
    return bar_y(note_index) - WRIST_TIP_Y_OFFSET


# A pleasant rainbow of bar colours.
BAR_COLORS = [
    [0.95, 0.32, 0.25, 1.0],  # C — red
    [0.98, 0.55, 0.20, 1.0],  # D — orange
    [0.96, 0.85, 0.30, 1.0],  # E — yellow
    [0.45, 0.85, 0.35, 1.0],  # F — green
    [0.30, 0.75, 0.85, 1.0],  # G — cyan
    [0.30, 0.55, 1.00, 1.0],  # A — blue
    [0.65, 0.40, 0.90, 1.0],  # B — purple
    [0.95, 0.50, 0.85, 1.0],  # c — pink
]


# Finger colour overrides (so the index is most prominent in the demo).
FINGER_COLORS = {
    "if": [0.95, 0.95, 0.95, 1.0],  # bright white — striker
    "mf": [0.55, 0.55, 0.60, 1.0],  # grey
    "rf": [0.55, 0.55, 0.60, 1.0],  # grey
    "th": [0.55, 0.55, 0.60, 1.0],  # grey
}
PALM_COLOR = [0.45, 0.45, 0.50, 1.0]
TIP_HILITE = {
    "if": [1.00, 1.00, 0.55, 1.0],  # warm yellow — the striker tip pops
    "mf": [0.65, 0.65, 0.70, 1.0],
    "rf": [0.65, 0.65, 0.70, 1.0],
    "th": [0.65, 0.65, 0.70, 1.0],
}


def colorize_fingers(model):
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


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------

def build_scene():
    host = mujoco.MjSpec()
    host.option.timestep = DT
    host.option.gravity = [0.0, 0.0, -9.81]
    host.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    host.visual.global_.offwidth = RES_W
    host.visual.global_.offheight = RES_H

    world = host.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[0, 0, 0.05], rgba=[0.04, 0.05, 0.07, 1.0])
    # three-point lighting
    world.add_light(pos=[0.5, 0.4, 1.4], dir=[-0.3, -0.3, -1.0],
                    diffuse=[1.0, 0.95, 0.85])
    world.add_light(pos=[-0.6, -0.3, 1.0], dir=[0.4, 0.2, -1.0],
                    diffuse=[0.45, 0.55, 0.70])
    world.add_light(pos=[0.0, -0.5, 0.7], dir=[0.0, 0.5, -1.0],
                    diffuse=[0.50, 0.45, 0.40])

    # --- mocap wrist body (LEAP attaches under this) ---
    # Initial wrist position is over bar 0 (C5); the simulation moves it
    # in Y between strikes.
    wrist_y0 = wrist_y_for_note(0)
    wrist = world.add_body(name="wrist", mocap=True,
                           pos=[0.0, wrist_y0, 0.30])
    # subtle visual marker so the wrist is locatable on the video
    wrist.add_geom(name="wrist_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[0.014, 0, 0], rgba=[0.85, 0.20, 0.20, 0.45],
                   contype=0, conaffinity=0)
    mount = wrist.add_frame(pos=[0, 0, 0])

    hand = mujoco.MjSpec.from_file(str(HAND_XML))
    host.attach(hand, prefix="hand_", frame=mount)

    # --- xylophone base plate (visual, no collision) ---
    base = world.add_body(name="xylo_base",
                          pos=[BAR_X, (BAR_Y_FIRST + bar_y(7)) / 2, 0.420])
    base.add_geom(name="xylo_base_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[0.060, 0.135, 0.006],
                  rgba=[0.16, 0.11, 0.08, 1.0],
                  contype=0, conaffinity=0)

    # --- 8 xylophone bars ---
    for i, note in enumerate(NOTE_ORDER):
        bname = f"bar_{note}"
        body = world.add_body(name=bname,
                              pos=[BAR_X, bar_y(i), BAR_Z_TOP])
        # hinge on X axis at the rear edge — bar visibly dips on impact
        body.add_joint(name=f"{bname}_hinge",
                       type=mujoco.mjtJoint.mjJNT_HINGE,
                       pos=[0, -BAR_HALF_Y, 0], axis=[1, 0, 0],
                       limited=True, range=[-0.05, 0.20],
                       damping=0.02, stiffness=8.0, springref=0.0)
        body.add_geom(name=f"{bname}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[BAR_HALF_X, BAR_HALF_Y, BAR_HALF_Z],
                      rgba=BAR_COLORS[i], density=400.0,
                      friction=[0.6, 0.05, 0.001],
                      solref=[0.005, 1])
        # touch-sensor site: sphere large enough to enclose the bar geom
        body.add_site(name=f"{bname}_site", pos=[0, 0, 0],
                      size=[max(BAR_HALF_X, BAR_HALF_Y) + 0.005, 0, 0],
                      rgba=[0, 1, 0, 0],
                      type=mujoco.mjtGeom.mjGEOM_SPHERE)

    # --- sensors ---
    for note in NOTE_ORDER:
        host.add_sensor(name=f"bar_{note}_touch",
                        type=mujoco.mjtSensor.mjSENS_TOUCH,
                        objtype=mujoco.mjtObj.mjOBJ_SITE,
                        objname=f"bar_{note}_site")
    for note in NOTE_ORDER:
        host.add_sensor(name=f"bar_{note}_angle",
                        type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT,
                        objname=f"bar_{note}_hinge")

    model = host.compile()

    # Stiffen LEAP position actuators (default kp=3 droops in palm-down).
    kp, kv = 25.0, 1.0
    for i in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if aname.startswith("hand_"):
            model.actuator_gainprm[i, 0] = kp
            model.actuator_biasprm[i, 1] = -kp
            model.actuator_biasprm[i, 2] = -kv

    # Disable collision on all non-index finger segments. The middle/ring/thumb
    # parked poses inevitably leave segments at or below the bar plane; if they
    # collide, every wrist slide drags them across bars and triggers spurious
    # strikes. We keep their visual geoms visible (visual class already has
    # contype=0); here we zero contype/conaffinity on their COLLISION geoms.
    KEEP_COLLIDE_PREFIXES = ("hand_palm", "hand_if_")  # palm + index only
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not bname.startswith("hand_"):
            continue
        if any(bname.startswith(p) for p in KEEP_COLLIDE_PREFIXES):
            continue
        # this body is mf / rf / th — disable any collision-class geom
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
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
# Wrist trajectory (closed-form Y schedule)
# ---------------------------------------------------------------------------

# Timing constants
WIND_UP_S = 0.18           # how long before strike_t the strike pose engages
RETRACT_S = 0.15           # how long after strike_t the strike pose holds
SLIDE_LEAD_S = 0.10        # extra dwell at target Y before strike

def wrist_y_at(t: float, schedule: list) -> float:
    """Closed-form wrist Y position at time t.

    - before the first note  → first note's Y
    - between notes A and B  → linear slide from A's Y to B's Y,
      finishing SLIDE_LEAD_S before B's strike_t
    - after the last note    → last note's Y
    """
    if not schedule:
        return wrist_y_for_note(0)
    if t <= schedule[0]["strike_t"]:
        return wrist_y_for_note(schedule[0]["note_idx"])
    if t >= schedule[-1]["strike_t"]:
        return wrist_y_for_note(schedule[-1]["note_idx"])
    # Find the pair (a, b) with a.strike_t <= t < b.strike_t.
    for i in range(len(schedule) - 1):
        a = schedule[i]
        b = schedule[i + 1]
        if not (a["strike_t"] <= t < b["strike_t"]):
            continue
        slide_start = max(a["strike_t"] + RETRACT_S,
                          b["strike_t"] - 0.5)
        slide_end = b["strike_t"] - SLIDE_LEAD_S
        ya = wrist_y_for_note(a["note_idx"])
        yb = wrist_y_for_note(b["note_idx"])
        if t < slide_start:
            return ya
        if t >= slide_end:
            return yb
        alpha = (t - slide_start) / max(0.01, slide_end - slide_start)
        return ya + alpha * (yb - ya)
    return wrist_y_for_note(schedule[-1]["note_idx"])


# ---------------------------------------------------------------------------
# Audio synthesis
# ---------------------------------------------------------------------------

AUDIO_SAMPLE_RATE = 44100
NOTE_DURATION = 0.50

def synth_audio(strike_events, total_duration_s: float, out_wav: Path):
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
        # bell-like timbre: fundamental + overtones
        wave_signal = (
            0.65 * np.sin(2 * np.pi * f0 * tt) +
            0.25 * np.sin(2 * np.pi * 2 * f0 * tt) +
            0.10 * np.sin(2 * np.pi * 3 * f0 * tt)
        )
        env = np.exp(-tt / 0.22)
        att_n = int(0.005 * AUDIO_SAMPLE_RATE)
        if 0 < att_n < n:
            env[:att_n] *= np.linspace(0, 1, att_n)
        v = float(ev.get("velocity", 0.7))
        buf[s0:s1] += v * wave_signal * env
    mx = max(np.abs(buf).max(), 1e-9)
    buf = buf / mx * 0.85
    int16 = (buf * 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(AUDIO_SAMPLE_RATE)
        f.writeframes(int16.tobytes())


def mux_audio_video(video_in, audio_in, video_out) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg is None:
        return False
    cmd = [ffmpeg, "-y",
           "-i", str(video_in), "-i", str(audio_in),
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-shortest", str(video_out)]
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
    for c in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
              "C:/Windows/Fonts/arial.ttf"]:
        try:
            return (ImageFont.truetype(c, 30),
                    ImageFont.truetype(c, 22),
                    ImageFont.truetype(c, 16))
        except Exception:
            continue
    f = ImageFont.load_default()
    return f, f, f


_FONT_BIG, _FONT_MED, _FONT_SM = _load_fonts()

STAFF_X0 = 80
STAFF_W = RES_W - 160
# Each note has its own Y on the staff (top = high pitch).
STAFF_Y_OF_NOTE = {n: 110 + (7 - i) * 9 for i, n in enumerate(NOTE_ORDER)}


def _current_piece(t, pieces):
    """Return (piece_dict, position_in_piece) for time t, or (None, None)
    if we're in intro / outro / inter-piece silence."""
    for p in pieces:
        if p["start_t"] <= t < p["end_t"]:
            return p, t - p["start_t"]
    return None, None


def _draw_marquee(d, t, pieces, total_t):
    """Programme marquee at the top: shows current piece title, or
    'PROGRAMME' panel listing all 4 pieces during intro / outro."""
    cur, _ = _current_piece(t, pieces)
    if cur is not None:
        idx = pieces.index(cur) + 1
        title = f"♪  Piece {idx} of {len(pieces)}  ·  {cur['title']}"
        tw = len(title) * 11
        tx = (RES_W - tw) // 2
        # subtle gold panel
        d.rectangle([(tx - 24, 10), (tx + tw + 24, 56)],
                    fill=(15, 10, 5, 220))
        d.rectangle([(tx - 24, 10), (tx + tw + 24, 14)],
                    fill=(220, 180, 90, 255))  # gold strip
        d.text((tx, 18), title, fill=(250, 235, 195), font=_FONT_MED)
    else:
        # intro / inter-piece / outro: show programme list
        d.rectangle([(RES_W // 2 - 240, 10), (RES_W // 2 + 240, 56)],
                    fill=(15, 10, 5, 220))
        d.rectangle([(RES_W // 2 - 240, 10), (RES_W // 2 + 240, 14)],
                    fill=(220, 180, 90, 255))
        # decide what phase we're in
        if t < pieces[0]["start_t"]:
            line = "Claude × LEAP    —    Robothon Concert"
        elif t >= pieces[-1]["end_t"]:
            line = "♪  Programme complete  ·  thank you"
        else:
            # between pieces
            nxt = next((p for p in pieces if p["start_t"] > t), None)
            line = f"Up next:  {nxt['title']}" if nxt else " "
        tw = len(line) * 11
        d.text((RES_W // 2 - tw // 2, 18), line,
               fill=(250, 235, 195), font=_FONT_MED)


def _draw_intro_card(d, t):
    """Big centered title card during intro window."""
    alpha = int(255 * max(0.0, min(1.0, 1.0 - abs(t - INTRO_DURATION_S * 0.5)
                                   / (INTRO_DURATION_S * 0.5))))
    # box
    d.rectangle([(RES_W // 2 - 380, RES_H // 2 - 110),
                 (RES_W // 2 + 380, RES_H // 2 + 110)],
                fill=(8, 6, 4, min(220, alpha)))
    d.rectangle([(RES_W // 2 - 380, RES_H // 2 - 110),
                 (RES_W // 2 + 380, RES_H // 2 - 106)],
                fill=(220, 180, 90, alpha))
    d.rectangle([(RES_W // 2 - 380, RES_H // 2 + 106),
                 (RES_W // 2 + 380, RES_H // 2 + 110)],
                fill=(220, 180, 90, alpha))
    line1 = "Claude  ×  LEAP Hand"
    line2 = "Robothon Summer 2026 — Concert in C Major"
    line3 = "Mary  ·  Twinkle  ·  Ode to Joy  ·  Happy Birthday"
    d.text((RES_W // 2 - len(line1) * 10, RES_H // 2 - 80),
           line1, fill=(250, 235, 195, alpha), font=_FONT_BIG)
    d.text((RES_W // 2 - len(line2) * 6, RES_H // 2 - 20),
           line2, fill=(220, 220, 230, alpha), font=_FONT_MED)
    d.text((RES_W // 2 - len(line3) * 5, RES_H // 2 + 30),
           line3, fill=(180, 180, 200, alpha), font=_FONT_SM)


def _draw_outro_card(d, t, pieces, stats):
    """Closing title card during outro window."""
    outro_start = pieces[-1]["end_t"]
    a = max(0.0, min(1.0, (t - outro_start) / OUTRO_DURATION_S * 2.0))
    alpha = int(220 * a)
    if alpha <= 0:
        return
    d.rectangle([(RES_W // 2 - 380, RES_H // 2 - 130),
                 (RES_W // 2 + 380, RES_H // 2 + 130)],
                fill=(8, 6, 4, alpha))
    d.rectangle([(RES_W // 2 - 380, RES_H // 2 - 130),
                 (RES_W // 2 + 380, RES_H // 2 - 126)],
                fill=(220, 180, 90, alpha))
    d.rectangle([(RES_W // 2 - 380, RES_H // 2 + 126),
                 (RES_W // 2 + 380, RES_H // 2 + 130)],
                fill=(220, 180, 90, alpha))
    line1 = "♪  thank you  ♪"
    line2 = f"{stats.get('strikes', 0)} notes performed   ·   " \
            f"{stats.get('accuracy_pct', 0):.0f}% accuracy"
    line3 = "Performed by Claude Opus 4.7   ·   built end-to-end by AI"
    d.text((RES_W // 2 - len(line1) * 9, RES_H // 2 - 90),
           line1, fill=(250, 235, 195, alpha), font=_FONT_BIG)
    d.text((RES_W // 2 - len(line2) * 7, RES_H // 2 - 20),
           line2, fill=(220, 220, 230, alpha), font=_FONT_MED)
    d.text((RES_W // 2 - len(line3) * 5, RES_H // 2 + 30),
           line3, fill=(180, 180, 200, alpha), font=_FONT_SM)


def draw_overlay(frame, t, schedule, pieces, played_notes,
                 bar_angles, bar_touch, wrist_y, stats):
    if not _PIL_OK:
        return frame
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")

    total_t = (pieces[-1]["end_t"] + OUTRO_DURATION_S)

    # --- top marquee (piece title or programme list) ---
    _draw_marquee(d, t, pieces, total_t)

    # --- big centered title card during intro / outro ---
    if t < INTRO_DURATION_S:
        _draw_intro_card(d, t)
    if t > pieces[-1]["end_t"]:
        _draw_outro_card(d, t, pieces, stats)

    # --- score strip with full octave ---
    y_top = 95
    y_bot = 188
    d.rectangle([(STAFF_X0 - 16, y_top), (STAFF_X0 + STAFF_W + 16, y_bot)],
                fill=(0, 0, 0, 150))
    # 8 staff lines + note labels
    for note in NOTE_ORDER:
        ly = STAFF_Y_OF_NOTE[note]
        d.line([(STAFF_X0, ly), (STAFF_X0 + STAFF_W, ly)],
               fill=(70, 70, 80), width=1)
        d.text((STAFF_X0 - 28, ly - 9), note,
               fill=(160, 160, 180), font=_FONT_SM)
    # plot scheduled notes
    total_t = (schedule[-1]["strike_t"] +
               schedule[-1]["duration_s"]) if schedule else 1.0
    played_keys = {(p["note"], p["strike_t"]) for p in played_notes}
    next_note = None
    for ev in schedule:
        if ev["strike_t"] >= t - 0.05:
            next_note = ev
            break
    for ev in schedule:
        nx = STAFF_X0 + STAFF_W * ev["strike_t"] / total_t
        ny = STAFF_Y_OF_NOTE[ev["note"]]
        if (ev["note"], ev["strike_t"]) in played_keys:
            col = (110, 230, 130)            # green: played
        elif next_note is not None and ev["strike_t"] == next_note["strike_t"]:
            col = (255, 230, 60)             # yellow: next
        elif ev["strike_t"] < t - 0.3:
            col = (210, 100, 100)            # red: missed
        else:
            col = (170, 170, 185)            # grey: future
        r = 6
        d.ellipse([(nx - r, ny - r), (nx + r, ny + r)], fill=col)
    # playhead
    ph = STAFF_X0 + STAFF_W * min(t / max(total_t, 1e-3), 1.0)
    d.line([(ph, y_top + 5), (ph, y_bot - 5)],
           fill=(255, 220, 50, 200), width=2)

    # --- bottom-left: per-bar lit status (compact 8-row strip) ---
    bx0, by0 = 30, RES_H - 360
    d.rectangle([(bx0, by0), (bx0 + 320, by0 + 340)], fill=(0, 0, 0, 165))
    d.text((bx0 + 12, by0 + 8), "xylophone bars",
           fill=(220, 220, 230), font=_FONT_MED)
    for i, note in enumerate(NOTE_ORDER):
        rgb = tuple(int(c * 255) for c in BAR_COLORS[i][:3])
        bx = bx0 + 16
        by = by0 + 50 + i * 34
        lit = bar_touch.get(note, 0) > 0.5
        col = (255, 235, 60) if lit else rgb
        d.rectangle([(bx, by), (bx + 36, by + 24)], fill=col)
        f = NOTE_FREQS[note]
        d.text((bx + 50, by + 2), f"{note}   {f:.0f} Hz",
               fill=(225, 225, 230), font=_FONT_MED)
        ang = bar_angles.get(note, 0.0)
        d.text((bx + 200, by + 6), f"{ang:+.3f} rad",
               fill=(155, 155, 165), font=_FONT_SM)

    # --- top-right: wrist Y + active key ---
    rx, ry = RES_W - 340, 220
    d.rectangle([(rx, ry), (rx + 320, ry + 130)], fill=(0, 0, 0, 165))
    d.text((rx + 12, ry + 8), "wrist (mocap)",
           fill=(220, 220, 230), font=_FONT_MED)
    d.text((rx + 12, ry + 42), f"y position = {wrist_y:+.4f} m",
           fill=(245, 230, 130), font=_FONT_MED)
    # find which bar the wrist is currently over
    closest = min(range(8), key=lambda i: abs(wrist_y - wrist_y_for_note(i)))
    err_mm = abs(wrist_y - wrist_y_for_note(closest)) * 1000
    d.text((rx + 12, ry + 74),
           f"over: {NOTE_ORDER[closest]}  (err {err_mm:.1f} mm)",
           fill=(225, 225, 230), font=_FONT_MED)
    d.text((rx + 12, ry + 100),
           f"next note: {next_note['note'] if next_note else '—'}",
           fill=(255, 230, 60), font=_FONT_MED)

    # --- bottom-right stats ---
    if stats:
        sx0, sy0 = RES_W - 290, RES_H - 130
        d.rectangle([(sx0, sy0), (sx0 + 270, sy0 + 110)], fill=(0, 0, 0, 170))
        d.text((sx0 + 12, sy0 + 6), "performance",
               fill=(220, 220, 230), font=_FONT_MED)
        rows = [
            f"strikes : {stats.get('strikes', 0)}",
            f"correct : {stats.get('correct', 0)} / {stats.get('past', 0)}",
            f"acc     : {stats.get('accuracy_pct', 0):.0f}%",
        ]
        for i, r in enumerate(rows):
            d.text((sx0 + 12, sy0 + 32 + i * 24), r,
                   fill=(225, 225, 235), font=_FONT_MED)

    # --- subtitle ---
    sub = f"LEAP Hand × C-major xylophone   ·   t = {t:5.2f}s   ·   sensor-gated music synthesis"
    sw = len(sub) * 9
    sxc = (RES_W - sw) // 2
    d.rectangle([(sxc - 12, RES_H - 34), (sxc + sw + 12, RES_H - 6)],
                fill=(0, 0, 0, 140))
    d.text((sxc, RES_H - 30), sub, fill=(210, 210, 220), font=_FONT_SM)

    return np.array(img)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

TOUCH_THRESHOLD = 0.05
STRIKE_DEBOUNCE_S = 0.30   # > rebound period, < beat period (0.46s at 130bpm)


@dataclass
class RunResult:
    seed: int
    tempo_bpm: float
    song: str
    scheduled_n: int
    struck_n: int
    correct_n: int
    accuracy_pct: float
    note_events: list = field(default_factory=list)
    schedule: list = field(default_factory=list)


def schedule_for_tempo(bpm: float, seed: int = 0):
    """Build the concert schedule. `bpm` scales the per-piece tempos
    (110 / 110 / 100 / 105 in CONCERT_PROGRAM) — a `bpm` of 110 leaves
    them unchanged. `seed != 0` jitters strike_t by σ ≈ 20 ms.
    Returns (schedule, pieces, total_duration_s).
    """
    scale = bpm / 110.0
    sched, pieces, total_t = make_concert_schedule(tempo_scale=scale)
    if seed != 0:
        rng = np.random.default_rng(seed)
        for ev in sched:
            ev["strike_t"] = round(ev["strike_t"] +
                                   float(rng.normal(0, 0.02)), 4)
    return sched, pieces, total_t


def simulate(seed: int = 12345, tempo_bpm: float = 110.0,
             render_video: bool = False, write_jsonl: bool = True) -> RunResult:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schedule, pieces, total_t = schedule_for_tempo(tempo_bpm, seed)

    model = build_scene()
    if render_video:
        colorize_fingers(model)
    data = mujoco.MjData(model)
    name2act = get_actuator_map(model)
    sensor_map = get_sensor_map(model)

    wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
    wrist_mocap = int(model.body_mocapid[wrist_bid])
    # initial wrist position
    data.mocap_pos[wrist_mocap] = [0.0, wrist_y_for_note(schedule[0]["note_idx"]), 0.30]
    data.mocap_quat[wrist_mocap] = [1, 0, 0, 0]

    # settle into rest pose
    apply_pose(data, name2act, REST_POSE)
    for _ in range(int(0.5 / DT)):
        apply_pose(data, name2act, REST_POSE)
        mujoco.mj_step(model, data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Camera locked to the xylophone's center — DO NOT track the wrist,
    # otherwise the hand looks stationary while the bars appear to scroll.
    # With a fixed camera the viewer sees the hand actually sliding across
    # the keyboard, which is what physically happens.
    xylo_y_center = (bar_y(0) + bar_y(7)) / 2.0   # ~ 0.05
    cam.lookat[:] = [0.06, xylo_y_center, 0.43]
    cam.distance = 0.60
    cam.azimuth = 90.0
    cam.elevation = -22.0
    renderer = (mujoco.Renderer(model, width=RES_W, height=RES_H)
                if render_video else None)

    total_steps = int(total_t / DT)
    steps_per_frame = max(1, int(round((1.0 / FPS) / DT)))

    frames = []
    strike_events = []
    played_notes = []
    bar_touch_prev = {n: 0.0 for n in NOTE_ORDER}
    last_strike_t_per_bar = {n: -10.0 for n in NOTE_ORDER}

    jsonl_f = open(OUT_JSONL, "w", encoding="utf-8") if write_jsonl else None

    next_idx = 0    # index into schedule

    for step in range(total_steps):
        t = step * DT

        # --- update wrist Y (mocap) ---
        target_wrist_y = wrist_y_at(t, schedule)
        data.mocap_pos[wrist_mocap] = [0.0, target_wrist_y, 0.30]

        # --- read sensors ---
        bar_touch = {n: float(read_sensor(data, sensor_map,
                                          f"bar_{n}_touch")[0])
                     for n in NOTE_ORDER}
        bar_angle = {n: float(read_sensor(data, sensor_map,
                                          f"bar_{n}_angle")[0])
                     for n in NOTE_ORDER}
        joint_pos = np.array(
            [read_sensor(data, sensor_map, f"hand_{j}_sensor")[0]
             for j in ["if_mcp", "if_rot", "if_pip", "if_dip",
                       "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
                       "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
                       "th_cmc", "th_axl", "th_mcp", "th_ipl"]])

        # --- strike detection (rising-edge with per-bar debounce) ---
        for note in NOTE_ORDER:
            cur = bar_touch[note]
            prev = bar_touch_prev[note]
            if (cur > TOUCH_THRESHOLD and prev <= TOUCH_THRESHOLD
                    and t - last_strike_t_per_bar[note] > STRIKE_DEBOUNCE_S):
                last_strike_t_per_bar[note] = t
                # match the closest scheduled note (within ±0.30s)
                matched = None
                for j, ev in enumerate(schedule):
                    if (ev["note"] == note and
                            abs(ev["strike_t"] - t) < 0.30 and
                            not any(e.get("schedule_idx") == j
                                    for e in strike_events)):
                        matched = j
                        break
                strike_events.append({
                    "t": round(t, 4),
                    "note": note,
                    "freq_hz": NOTE_FREQS[note],
                    "velocity": min(1.0, cur / 2.0 + 0.30),
                    "schedule_idx": matched,
                    "scheduled_t": (schedule[matched]["strike_t"]
                                    if matched is not None else None),
                })
                if matched is not None:
                    played_notes.append({"note": note,
                                         "strike_t": schedule[matched]["strike_t"]})
        bar_touch_prev = bar_touch

        # --- advance scheduler pointer ---
        while (next_idx < len(schedule) and
               schedule[next_idx]["strike_t"] < t - 0.30):
            next_idx += 1

        # --- finger pose: strike around the current note's window ---
        if next_idx < len(schedule):
            ev = schedule[next_idx]
            in_window = (ev["strike_t"] - WIND_UP_S) <= t <= (ev["strike_t"] + RETRACT_S)
            if in_window:
                apply_pose(data, name2act, STRIKE_POSE)
            else:
                apply_pose(data, name2act, REST_POSE)
        else:
            apply_pose(data, name2act, REST_POSE)

        mujoco.mj_step(model, data)

        # --- per-frame JSONL stream ---
        if jsonl_f is not None and step % steps_per_frame == 0:
            jsonl_f.write(json.dumps({
                "t": round(t, 4),
                "wrist_y": round(target_wrist_y, 5),
                "bar_touch": {k: round(v, 4) for k, v in bar_touch.items()},
                "bar_angle_rad": {k: round(v, 4) for k, v in bar_angle.items()},
                "joint_pos": joint_pos.round(4).tolist(),
                "n_strikes_so_far": len(strike_events),
            }) + "\n")

        # --- video frame ---
        if render_video and renderer is not None and step % steps_per_frame == 0:
            # Camera is fixed — no wrist tracking. Tiny azimuth sway only,
            # for cinematic feel without obscuring the hand's lateral motion.
            cam.azimuth = 90.0 + 4.0 * math.sin(0.15 * t)
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()

            past = sum(1 for ev in schedule if ev["strike_t"] < t)
            correct = sum(1 for ev in schedule
                          if any(e.get("schedule_idx") is not None and
                                 schedule[e["schedule_idx"]] is ev and
                                 e["note"] == ev["note"]
                                 for e in strike_events))
            stats = {
                "strikes": len(strike_events),
                "correct": correct,
                "past": past,
                "accuracy_pct": (100.0 * correct / max(1, past)),
            }
            frame = draw_overlay(frame, t, schedule, pieces, played_notes,
                                 bar_angle, bar_touch, target_wrist_y,
                                 stats=stats)
            frames.append(frame)

    if jsonl_f is not None:
        jsonl_f.close()

    correct = sum(1 for ev in schedule
                  if any(e.get("schedule_idx") is not None and
                         schedule[e["schedule_idx"]] is ev and
                         e["note"] == ev["note"]
                         for e in strike_events))
    accuracy = 100.0 * correct / max(1, len(schedule))

    concert_title = "  ·  ".join(p["title"] for p in pieces)
    result = RunResult(
        seed=seed, tempo_bpm=tempo_bpm,
        song=concert_title,
        scheduled_n=len(schedule), struck_n=len(strike_events),
        correct_n=correct, accuracy_pct=round(accuracy, 1),
        note_events=strike_events, schedule=schedule,
    )

    if render_video and frames:
        iio.imwrite(str(OUT_VIDEO_SILENT), frames, fps=FPS,
                    codec="libx264", quality=8)
        synth_audio(strike_events, total_t, OUT_AUDIO)
        if mux_audio_video(OUT_VIDEO_SILENT, OUT_AUDIO, OUT_VIDEO):
            print(f"  demo.mp4 written with audio ({len(strike_events)} notes).")
        else:
            shutil.copyfile(OUT_VIDEO_SILENT, OUT_VIDEO)
            print("  Note: ffmpeg unavailable — demo.mp4 is silent.")
        OUT_TRAJECTORY.write_text(json.dumps({
            "seed": seed,
            "tempo_bpm": tempo_bpm,
            "concert_program": pieces,
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
              f"{r.correct_n}/{r.scheduled_n} correct  "
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
    tempos = [90.0, 110.0, 130.0]
    out = {"tempos_bpm": tempos, "per_tempo": {}}
    for bpm in tempos:
        results = []
        print(f"--- tempo {bpm} bpm ---")
        for s in range(n):
            seed = 2000 + s * 13
            r = simulate(seed=seed, tempo_bpm=bpm,
                         render_video=False, write_jsonl=False)
            results.append(r)
            print(f"  seed {seed:5d}  {r.correct_n}/{r.scheduled_n}  "
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
    p.add_argument("--tempo", type=float, default=110.0)
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
        print(f"  song: {r.song}")
        print(f"  scheduled notes: {r.scheduled_n}")
        print(f"  strikes detected: {r.struck_n}")
        print(f"  correct: {r.correct_n}")
        print(f"  accuracy: {r.accuracy_pct:.1f}%")
        if not args.no_video:
            print(f"  video: {OUT_VIDEO}")


if __name__ == "__main__":
    main()
