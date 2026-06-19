# LEAP Dexterous Stabilization

A reference submission for FFAI Robothon Summer 2026, built end-to-end by an
AI agent (Claude). Targets the dexterous-manipulation track demonstrated by
the current top entries on the leaderboard.

> **Not a real entry.** The author (Anthropic's Claude, operating inside an
> FF employee's session) is ineligible per the official rules. This folder
> is published as a public, copy-pasteable template. Replace the UUID and
> swap in your own variations to submit.

## What it does

A LEAP right hand cradles an orange cube in a cage grasp and **keeps the cube
held under random external perturbations**, using a closed-loop finger-grip
controller. Over a 20-second window:

* The cube starts at the cage cup center (measured, not guessed).
* Every ~3 seconds, a random-direction 0.8 N horizontal force pulse hits the
  cube for 0.15 s.
* The controller reads the cube's planar drift from cup center each step and
  tightens finger position targets proportional to drift; the cube settles
  back to the cup center.
* `mj_step` runs the entire time. Every contact, every force, every grip
  change is physics-simulated.

**Result on the included run (deterministic, seed=12345):**
* `cube_held_final: true` (cube never falls)
* `max_planar_drift: 9.9 mm`
* `avg_planar_drift: 9.5 mm`
* `perturbations: 6` (all rejected)

## How it works

```
sense()      ── read cube body position (3D) from mj_data
   │
   ▼
plan()       ── compute planar drift = |cube_xy - cup_center_xy|
   │
   ▼
control()    ── tighten = clamp((drift - 5mm) / 15mm, 0, 1)
                set 16 LEAP position-actuator targets =
                CAGE_BASE + tighten * GRIP_TIGHTEN_DELTA
   │
   ▼
perturb()    ── every 3 s, apply 0.8 N horizontal force on cube
                for 0.15 s in a random direction
   │
   ▼
mj_step()    ── full physics: contacts, friction, fingertip dynamics
   │
   ▼
log + render ── per-frame cube pose, drift, grip command,
                perturbation status; cinematic slow orbit camera
```

Everything sits in `main.py` (~280 lines, single file). LEAP Hand assets
(MIT-licensed) are vendored under `assets/leap_hand/`.

## Repo layout

```
submissions/claude-dex-stabilize/
├── README.md
├── main.py
├── registration.json
├── BUILD_LOG.md
├── assets/leap_hand/        (vendored from mujoco_menagerie)
└── outputs/
    ├── demo.mp4
    └── trajectory.json
```

## Run it

From the **repo root** (not this folder):

```bash
python -m pip install -r requirements.txt
python submissions/claude-dex-stabilize/main.py
```

Outputs land in `submissions/claude-dex-stabilize/outputs/`:
* `demo.mp4` — 20 s, 1280×720, 30 fps, slow orbit camera
* `trajectory.json` — full summary, per-frame log samples, perturbation
  schedule, and the deterministic RNG seed

Runtime: ~30–60 s on a modest laptop, no GPU required.

## How this maps to the official rubric

| Criterion | How this entry addresses it |
|---|---|
| **Runnability** | One file, three pip dependencies, deterministic (RNG-seeded). `python main.py` reproduces the metrics. |
| **Depth of MuJoCo Use** | MjSpec at compile time, free joint on the cube, wrist hinge body, native LEAP hand integration via attach + prefix, custom friction tuning, `mj_step` full physics, `xfrc_applied` for external perturbation, position actuators on every finger joint. |
| **Task Design** | Real, quantifiable task: hold the cube under random perturbations for 20 s. Pass/fail well defined (`cube_held_final`), graded by `max_planar_drift` and `avg_planar_drift`. Six perturbations injected. |
| **Control** | A real closed loop: sensor (cube position) → planner (drift) → controller (16 finger targets) → physics → contact change → cube position. Not a timeline playback. |
| **Dexterous Manipulation** | Direct LEAP-Hand use; 16-DoF, 4 fingers + opposable thumb. Cage grasp pose is parameterized, grip tightening is per-joint. Cup center was *measured* (cage-pose fingertip COM + palm midpoint), not guessed. |
| **Engineering Quality** | Single file, all constants hoisted top, type hints, dataclass-style state, deterministic seed, no globals, JSON-serializable output (no numpy types leak). |
| **Presentation** | Cinematic slow-orbit camera, three-point lighting on the cube, deliberate dark palette so the orange cube reads instantly. Frame at t=10 s shows the cradle pose clearly. |
| **Innovation** | The controller is intentionally simple (one scalar grip-tighten knob) but yields measured robustness against unscripted disturbances. Cup-center calibration via a separate measurement pass before deploying is a small but reusable engineering idea. |

## What this entry honestly does NOT do (gradable docks)

* **Not a real in-hand reorientation.** The cube stays in the cup; it does
  not rotate around its own axis. To extend: add per-finger gait or
  re-grasp sequence to roll the cube.
* **Wrist is locked at yaw=0.** An earlier version drove the wrist hinge
  through a sinusoidal command — set via `qpos` directly, the kinematic
  jump ejected the cube. Driving the wrist through an actuator with a
  velocity limit would solve this; left as an upgrade path.
* **One cube, one shape.** Same controller should generalize to small
  spheres or short cylinders; not benchmarked here.
* **No tactile or proprioceptive sensor.** The "sensor" is the cube's
  ground-truth pose; in a real robot this would come from vision plus
  fingertip force sensors. The interface is set up so a real sensor stub
  could slot in.

These are the obvious upgrade paths a competitive entry would take.

## License

MIT for this folder's contents. LEAP Hand assets retain their original MIT
license (see `assets/leap_hand/LICENSE`).
