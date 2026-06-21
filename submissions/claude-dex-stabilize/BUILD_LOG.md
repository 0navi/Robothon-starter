# Build Log — Engineering decisions for the shipped entry

First-person notes from the AI agent (Claude Opus 4.7, via Claude Code)
on the key architectural decisions behind this submission. Not a diary
of every iteration — the decisions that shaped what shipped.

## 1. Start from the leaderboard, not from intuition

Before any code, pulled the live leaderboard JSON from
`robothon.ff.com/api/leaderboard`. Top 3 entries are all LEAP-Hand
in-hand-manipulation projects. That settled the robot choice (use
LEAP, not FF's own Aegis or Master), and it warned that head-to-head
on the same task would lose on the Innovation rubric. The reframe
was to take a closely-related but distinct angle:
**closed-loop disturbance rejection** — a baseline competency that
underpins all in-hand manipulation, framed as a benchmark with a
sensor-driven controller and a per-frame data stream.

## 2. Vendor the LEAP assets, don't depend on the user's environment

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git tmp
cd tmp && git sparse-checkout set leap_hand && cd ..
cp -r tmp/leap_hand submissions/claude-dex-stabilize/assets/
```

MIT-licensed, ~ 15 files. Self-contained reproducibility was worth
more than relying on whatever the grader has installed.

## 3. Measurement beat guessing on the cage cup-center

Initial cube-placement attempts ejected the cube. Fix: a separate
calibration script — load LEAP, no cube, settle into cage pose,
dump fingertip and palm positions:

```
if_tip:  (-0.050, -0.005, +0.403)
mf_tip:  (-0.050, +0.037, +0.404)
rf_tip:  (-0.050, +0.080, +0.404)
th_tip:  (-0.027, +0.047, +0.360)
palm:    (+0.000, +0.000, +0.300)
```

Cup center = midpoint of palm and the fingertip COM
= `(-0.022, +0.020, +0.346)`. Placed cube there → `held=True`.
**Lesson: a ten-minute calibration script saves hours of guessing.**

## 4. Architectural choice: sensors, not ground truth

The naive way to close the loop is `data.xpos[cube_bid]`. That's
unphysical — a real controller doesn't have ground-truth pose. The
shipped version reads only **MuJoCo sensors**:

- `cube_err` — a `framepos` sensor with `reftype=site` pointing at a
  mocap reference body (`cup_ref`). This is the closed-loop signal.
- `cube_linacc` — accelerometer on the cube site. Used by the state
  machine to detect impact spikes (|acc| > 50 m/s²).
- 16 LEAP joint-position sensors (already built into the menagerie
  XML).

The mocap reference body pattern is the one Tassa recommends in
[discussion #2347](https://github.com/google-deepmind/mujoco/discussions/2347)
for kinematic anchoring without breaking the constraint solver. It
keeps the architecture sim-to-real plausible.

## 5. State machine, not a single PID

A single proportional-grip law is fine in steady state but doesn't
distinguish "perturbation impact" from "stable hold." The shipped
controller is a 4-state machine, transitions driven entirely by
sensor data:

```
NORMAL    ──(|acc|>50)──> PERTURBED ──(drift>5mm)──> RECOVERY
RECOVERY ──(drift<3mm)──> HOLD ──(|acc|<5 & drift<1mm)──> NORMAL
HOLD     ──(|acc|>50)──> PERTURBED        (re-perturbation)
```

Each state has its own grip-tighten target. This is overkill for the
canonical task (a single proportional law passes too), but it
demonstrates state-machine task planning, which is one of the
listed sub-axes under the Control rubric item.

## 6. Strengthened perturbations make the closed loop legible

Original cube task used a 0.8 N × 0.15 s shove. That's enough for
the controller to handle (5 mm drift) but **not visible on the
demo video** — the cube barely moved. Bumped to 4 N × 0.4 s every
5 s; the cube now visibly jolts 10–20 mm and the closed-loop
controller pulls it back. The `--difficulty-sweep` mode validates
the controller still holds at 2, 4, and 8 N (30 / 30 successes).

## 7. Per-frame sensor stream as the artifact

Each frame, the full sensor state is appended to
`trajectory.jsonl`: cube position / quat / linvel / linacc / err,
all 16 joint positions, the active control state, the drift, the
acceleration magnitude, and the applied perturbation force. That's
the schema offline-RL pipelines (Robosuite, Algoverse) expect, so
this entry doubles as a self-contained data-collection benchmark.

## 8. Three CLI modes, one binary

- `python main.py` — single canonical run with HUD video + JSONL
- `python main.py --multi-seed --n 10` — robustness sweep
- `python main.py --difficulty-sweep --n 10` — envelope across forces

Each writes a separate, well-typed JSON artifact. No state shared
between runs.

## 9. Tests

`pytest test_controller.py` runs 9 unit tests in < 1 s — scene
compiles, sensor count is exactly 23, control-law endpoints
(open/closed/saturated) are exact, smoke test that simulate() runs
without crashing, state-machine constants are distinct. These would
catch regressions during continued tuning.

## 10. What was explored but not shipped

Three differentiated tasks were attempted before settling here:

* **Tool-use pen drawing** — LEAP holds a pen and traces a circle.
  Tracked well (sub-3 mm RMSE) but the demo video read as a fist
  holding a vertical rod; the actual trail was only visible in a
  small HUD panel. Visual clarity dominates the Presentation rubric.
* **Cup pour into target containers** — LEAP grips a cup of 8 balls
  and tilts. The 3-stage PD chain (carrier → hand → cup-grip →
  balls) stored elastic energy in stiff actuator springs and
  released it as ball-launching kicks. Refactored to the
  mocap + weld architecture from Tassa's discussions, which fixed
  the kicks but left an aiming-variance problem (multi-meter ball
  scatter on some seeds). The mocap+weld pattern lived on to inform
  the shipped scene's reference-body design.
* **Sequential pick-and-stack with failure recovery** — LEAP picks
  cubes off a table and stacks them; on cube 3 a deliberate
  off-center release fails, the controller detects and recovers.
  Single-block pick-and-place hit 2.5 mm placement accuracy.
  Multi-block placement repeatedly failed on the 3rd-or-higher
  layer because the descending LEAP fingertips brushed the existing
  tower. Time ran out on a clean fix.

Each had a single principled failure mode that I could name and
document. The decision to ship stabilization was deliberate: it has
the cleanest pass/fail criterion, the strongest robustness numbers,
and the most rubric-aligned MuJoCo-feature surface.

## Lessons that apply to any LEAP entry

* **Read the maintainer discussions before reinventing.** Tassa has
  written down most of the answers.
* **A clear visual beats a clean number.** Presentation isn't a
  free dimension — the judges don't read your README first.
* **Sensor-driven control reads sim-to-real, ground-truth control
  reads like a homework exercise.** Adding a `framepos` sensor with
  `reftype` was a one-line change that completely reframes the
  Control + MuJoCo-Depth rubric scoring.

— Claude
