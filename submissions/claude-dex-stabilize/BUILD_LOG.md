# Build Log — Three failed experiments and the ship decision

First-person log of what an AI agent (me, Claude Opus 4.7, via Claude
Code) actually did to take this submission from "what's on the
leaderboard" to a working 90-second `demo.mp4`. The journey is the
product: four iterations, three principled failures, one ship.

## TL;DR

1. **Piano (abandoned, < 1 hour)** — LEAP's finger curls retract the
   tip *along the palm direction*, not vertically. Cannot press keys.
   Wrong hardware for the task.
2. **Cube stabilization (working baseline)** — held a 36 mm cube under
   0.8 N perturbations, 10/10 success, < 5 mm drift. *Same track as
   leaderboard top 3*, would score 80-ish — not satisfying.
3. **Pen drawing (Z, ~ 3 hours)** — LEAP grips a 28 cm pen, 2-DoF
   carrier traces a 1.5 cm circle. Metrics OK (2.80 mm RMSE) but the
   visual is unreadable — the "grip" looks like a fist around a red
   pole, no obvious "drawing" action.
4. **Cup pour into two containers (~ 2.5 hours)** — LEAP grips a cup
   of 8 balls, tilts to pour. Tilt actuator either too soft (stalls
   at 15°) or too stiff (whips balls at multi-m/s).
5. **Cup pour v2 with mocap+weld (~ 1.5 hours, *plus* research)** —
   refactored after reading [Tassa's discussion #2347](https://github.com/google-deepmind/mujoco/discussions/2347).
   Stable rigid grasp, no more launches, but aiming variance still
   high — balls land in unpredictable XY.
6. **Sequential pick-and-stack with failure recovery (~ 3 hours)** —
   ported mocap+weld to a pick-place task. Weld activation
   teleport/instability caused cubes to detach mid-transport.
   Couldn't ship reliably in the time budget.
7. **Ship: cube stabilization with strengthened perturbations** — the
   only entry that ran reliably across multi-seed sweeps within the
   remaining budget. Bumped perturbation force 0.8 N → 4 N and
   duration 0.15 s → 0.4 s so the closed-loop control is **visible
   on the demo video** (cube visibly jolts and is pulled back into
   the cage).

## Step-by-step

### 1. Looked at the leaderboard first

Pulled the live leaderboard JSON from `robothon.ff.com/api/leaderboard`.
Top 3 are all LEAP-Hand in-hand-manipulation projects (Dexterous Triage,
Closed-Loop In-Hand Reorientation, DexSuite). Direction obvious: do
something plausible on LEAP, not on FF's own Aegis or Master. But also
clear: head-to-head against the top 3 on the same task = automatic loss
on Innovation rubric.

### 2. Vendored LEAP from mujoco_menagerie

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git tmp
cd tmp && git sparse-checkout set leap_hand && cd ..
cp -r tmp/leap_hand submissions/claude-dex-stabilize/assets/
```

MIT license, no auth, ~ 15 files total. Self-contained reproducibility
was worth more than relying on the user's environment.

### 3. First attempt — piano. Aborted.

Tried to make LEAP play a small piano. Confirmed via empirical
calibration that LEAP's curl mostly retracts the tip **along the palm
direction**, not vertically. About 5 cm of X-axis motion vs 3 mm of
Z-axis motion at a fingertip. Piano keys need vertical press. LEAP is
designed for grasping, not key-pressing — wrong hardware. Cut.

### 4. Cube stabilization baseline — calibration unlock

First non-trivial result was the cube cage. Initial pose-guessing
attempts ejected the cube. Solved by writing a calibration script:
load LEAP, no cube, settle into cage pose, dump fingertip + palm
positions. Result:

```
if_tip:  (-0.0504, -0.0049, +0.4034)
mf_tip:  (-0.0505, +0.0373, +0.4036)
rf_tip:  (-0.0505, +0.0795, +0.4035)
th_tip:  (-0.0267, +0.0471, +0.3599)
palm:    (+0.0000, +0.0000, +0.3000)
```

Cup center = midpoint of palm and fingertip COM = `(-0.022, +0.020, +0.346)`.
Placed cube there. `held=True`. **Lesson: measurement beats guessing.**

Stabilization with 0.8 N perturbations, 10/10 seeds successful, < 5 mm
drift. Worked but *visually static* — the cube barely moves on screen,
the stabilization is too good to look like anything is happening.

### 5. Z (pen drawing) — visual flop

Pivoted to tool use: LEAP grips a long cylindrical pen, a 2-DoF slide
carrier drives the hand in a 1.5 cm horizontal circle, the pen tip
drags on a drawing board below. Single canonical metric: 2.80 mm
radial RMSE against best-fit circle of radius 15 mm.

**It worked technically** — `outputs/multi_seed_stats.json` from that
phase shows 10/10 successful pen holds. But the user looked at the
render and called it: the visual reads as a fist holding a vertical
red pole, the drawing board is small and tan in the bottom of frame,
the actual trail is rendered only inside a small HUD panel. Nothing
about the video says "robot drawing." Lesson:
**a technically-correct demo with bad visual cues is worse than a
boring demo with clear visual cues.** Headline metrics don't carry the
score if the judge can't read what's happening.

### 6. Pour task v1 — actuator stall vs whip

Pivoted to pouring: LEAP grips a 50×50×30 mm open-top cup containing
eight 6 mm balls. A 4-DoF mount (XYZ slide + Y-axis tilt hinge) brings
the cup over a big container, tilts it, then over a small container,
tilts more. Scoring is `1 × big + 2 × small`.

First architecture: 3-stage PD chain (mocap-less, all actuator-driven).
Hit a sharp tradeoff:
- `kp=12` on the tilt hinge — couldn't overcome gravity at 90°,
  hinge stalled at 15°.
- `kp=80` — tilt reached 113°, but the resulting angular acceleration
  whipped balls out of the cup at multi-meter velocities (some balls
  ended 5 m off-grid).

### 7. Research detour — Tassa's discussions

At this point I asked the user for permission to research outside the
codebase. Spent ~ 20 minutes reading MuJoCo maintainer discussions:

- **[#2318](https://github.com/google-deepmind/mujoco/discussions/2318)** —
  Tassa: "you cannot get perfect tracking with position actuators
  unless you add gravity compensation. Don't set qpos during sim, this
  is unphysical."
- **[#2347](https://github.com/google-deepmind/mujoco/discussions/2347)** —
  for high-energy contacts, use `solref="-1000 0", solimp="0.99 0.99 0.01"`
  and `cone="elliptic"`. The whip came from "energy invisible to you,
  stored in the contact spring."
- **[Mochan mocap tutorial](https://mochan.org/posts/mujoco_mocap_1/)** —
  use a `mocap` body + `weld` equality to drive a robot smoothly without
  the kp-tuning problem entirely.
- **[LEAP Hand paper](https://arxiv.org/pdf/2309.06440)** — the
  "Pour Tea" task they cite uses a weld equality between cup and
  fingertip, toggled active once fingers close. That's the industry
  pattern for grasp-stability in non-grasp-research tasks.

This was the most valuable hour of the session.

### 8. Pour task v2 — mocap + weld architecture

Refactored `pour_main_v2.py`:
- 4-DoF carrier deleted entirely; replaced with a single mocap body
  driving a freejoint carrier via `weld` equality (`solref="0.002 1"`).
- Cup-to-thumb-tip rigid grasp via second `weld` equality, activated
  after the LEAP cage closes.
- All ball + cup geoms got `solref="0.005 1"` and `solimp="0.99 0.999..."`
  — critically damped contacts, no elastic energy storage.
- LEAP fingers reverted to menagerie defaults (`kp=3, kv=0.01`).
- Cup body got `gravcomp="1"` so gravity is automatically compensated.

**The ball-whip problem vanished.** Balls now pour smoothly. But a
new problem appeared: the cup's swing arc during tilt put balls in
unpredictable XY landing zones, and the cup's offset from the mocap
pivot meant aiming required dynamic mocap-pose compensation that I
didn't have time to tune. Multi-seed sweeps showed: 9/10 balls poured
out, but they landed at scattered XY (some seeds threw balls 1–5 m
off-target).

### 9. Stack task — last differentiated swing

Ported the same mocap+weld architecture to pick-and-stack. The picture
in my head was: LEAP picks up 4 colored cubes from a table, stacks
them into a tower; on the 3rd cube a deliberate off-center release
triggers a failure-detection-and-recovery loop. This dimension —
**failure recovery** — is a Figure AI / Boston Dynamics / 1X favorite
narrative, and would have demonstrated sense → plan → act → detect →
re-plan in a clean closed loop.

Architecture: same as pour v2 (mocap-weld carrier + per-cube grasp
welds activatable independently). Flipped the LEAP 180° around X via
the carrier body's quat so fingers point down (so LEAP can approach
cubes on a table from above). Spike confirmed the orientation
geometry — fingers ~ 16 cm below palm, palm 10 cm below mocap, cube
center should align with palm minus 22 mm.

**Failure mode**: the grasp weld is activated mid-trajectory with
`relpos` set to the cube's current (cube_pos − palm_pos). Even with
stiff `solref="0.002 1"`, the weld engagement transferred enough
acceleration that some cubes detached during transport — they ended up
several meters off-table, on the floor. Cubes that *did* travel
correctly didn't always land at the stack XY (off by 2–6 cm). I
suspect the issue is the carrier-weld lag interacting with the
grasp-weld in series, but didn't have the time to fully diagnose.

### 10. The ship decision

At T-minus ~ 17 hours, with three differentiated tasks all hitting
principled failures and time pressure mounting, I made the call:

**Ship the working cube stabilization, with strengthened
perturbations so the closed-loop control is *visible*.**

Changes from the original cube version:
- Perturbation force 0.8 N → 4 N (5× stronger)
- Perturbation duration 0.15 s → 0.4 s (3× longer)
- Perturbation period 3 s → 5 s (gives eye time to follow each one)
- Total: 18 perturbations across 90 s
- Control law saturation: drift threshold raised 5 mm → 8 mm, range
  raised 15 mm → 22 mm (since 4 N produces ~ 15–20 mm drift, not 5 mm)
- Per-joint tighten delta: scaled up ~ 1.5× to match the larger drift

The result: a video where you can **see** the cube get visibly shoved
on every perturbation, and **see** the fingers tighten to pull it
back. Closed-loop control made legible.

## Lessons that apply to any LEAP entry

* **LEAP is for grasping, not key-pressing.** Don't fight its design.
* **Measure before placing.** A 10-minute calibration script saves
  hours of guessing.
* **Position-actuator PD chains stack badly.** Three serial PD stages
  (carrier → hand → object) store and release elastic energy as
  impulsive kicks on every setpoint change. Use mocap + weld for
  kinematic carriers.
* **Damp contact springs.** `solref` second arg = 1 (critically
  damped) is mandatory in multi-body sims; default presets like
  "bouncy ball" are bug magnets.
* **Read the maintainer discussions before reinventing.** The MuJoCo
  team has written down most of the answers; I just needed to ask.
* **A clear visual beats a clean number.** A 2.80 mm RMSE that the
  viewer can't see and a 5 mm drift that the viewer can't see score
  the same on Presentation: zero.
* **Ship the artifact you can stand behind, not the artifact you
  wanted.** The journey is the product.

— Claude
