# Build Log — Engineering decisions for the xylophone submission (v2)

First-person notes from the AI agent (Claude Opus 4.7, via Claude Code)
on the architectural decisions behind this submission.

## 1. Why pivot from cube-stabilize to music

The first submission (`submissions/claude-dex-stabilize`) was a closed-loop
cube disturbance-rejection demo. Reviewing it against the official 8-axis
rubric and against the strongest competing submission (SlipZero — Franka
arm + LEAP friction-margin), two gaps stood out:

1. **Presentation** rubric: cube drift is < 15 mm, visually unimpressive
   without reading the README. A musical task is recognizable by ear.
2. **Innovation** rubric: stabilization is the most-attempted task on
   the LEAP leaderboard. Music synthesis is structurally different —
   *discrete event scheduling × contact-based event detection*.

Music keeps the LEAP hand at the center of the visual frame while
moving onto a control-problem axis no leaderboard entry occupies.

## 2. v1 was 3 notes; v2 is 8 — and the difference is everything

The first iteration parked the LEAP hand stationary and used three
fingers (index, middle, ring) on three bars (C, D, E). It "worked"
(100 % accuracy across 30 multi-seed runs) but the "songs" — Hot Cross
Buns, "Mary Had a Little Lamb", "Three Blind Mice" — were ear-test
failures: Mary and Three Blind Mice both genuinely need notes outside
C–D–E, and I had silently truncated them. The audio sounded like an
alarm clock, not music.

v2 fixes this by going to the full **C-major scale (C5..C6)** on 8 bars
and adding a **mocap wrist** that slides laterally in Y. The index
finger does all strikes; the wrist positions it. This lets the demo
play real recognizable tunes (Twinkle Twinkle, Ode to Joy) with the
correct notes, not stripped-down approximations.

## 3. The mocap wrist pattern

The LEAP root attaches directly under a mocap body's frame:

```python
wrist = world.add_body(name="wrist", mocap=True, pos=[0.0, y0, 0.30])
mount = wrist.add_frame(pos=[0, 0, 0])
host.attach(hand, prefix="hand_", frame=mount)
```

Per-step the controller writes `data.mocap_pos[wrist_mocap_id] =
[0, target_y, 0.30]` and the entire LEAP follows kinematically — no
weld equality, no carrier body, no constraint solving for the wrist.

This is cleaner than the *carrier + freejoint + weld* pattern (used in
the cube-stabilize submission's `cup_ref` body) because here we want
*the LEAP to follow exactly*, not to interact through a soft
constraint. Mocap is the right tool when you want kinematic input.

## 4. The wrist trajectory: closed-form, not a state machine

`wrist_y_at(t, schedule)` is a pure function: given current time and
the full song schedule, return the wrist Y position. It does linear
interpolation between consecutive notes' bar Y positions, finishing
the slide `SLIDE_LEAD_S = 0.10 s` before each strike instant.

The first v2 implementation had a control-flow bug: the loop
short-circuited at `if t < a["strike_t"]` on the first iteration,
returning the first note's Y *forever*. The fix (re-shipped here) is
to only act inside the matching pair `a["strike_t"] <= t < b["strike_t"]`
and `continue` otherwise. Lesson: closed-form trajectory functions
are wonderful but exhaustively test the edge cases.

## 5. The big debug: parked fingers dragging through bars

After the mocap pattern compiled, the first run reported 142 strikes
on 72 scheduled notes — almost 2× per beat — and only 12 correct
matches. Inspecting strike events showed *all* contacts on bar C
regardless of which note was scheduled. With the wrist-trajectory bug
the wrist never left C's position, so every strike landed on C.

After the trajectory fix, the count came down — but it was still
inflated. Root cause: the **parked middle / ring / thumb fingertips
were at z ≈ 0.455 m, below the bar plane z = 0.480 m**. As the wrist
slid laterally, those fingertips dragged across bars and triggered
false-positive contacts.

Two possible fixes:
1. Choose parking poses that lift the tips above the bar plane —
   tried, but the LEAP joint envelope doesn't reach high enough
   without weird thumb gymnastics.
2. **Disable collision on all non-index hand geoms.** The finger
   geoms still render (their visual class already had `contype=0`);
   only the *collision* class geoms get `contype = conaffinity = 0`
   set after compile.

Fix #2 is one loop over `model.geom_*`. Done.

## 6. Debounce tuned to the beat

Each scheduled strike caused 2 rising-edge events ~ 210 ms apart: the
initial impact and the spring-hinge rebound. The first debounce window
was 0.18 s — too short. Bumped to **0.30 s**, which is longer than the
rebound but shorter than the beat at 130 bpm (0.46 s). At very fast
tempos the rebound and the next intended strike start to compete; that
shows up as the 130-bpm accuracy drop in the difficulty sweep.

## 7. Touch-sensor site-sphere pitfall (carried over from v1)

The MuJoCo touch sensor counts contact normal forces whose contact
positions lie inside the site's bounding sphere. Default site size is
**1 mm**. With 40-mm-wide bars the sensor reads 0 even with active
contact. Fixed by setting the bar site radius to
`max(bar_half_x, bar_half_y) + 5 mm` so the entire bar surface lies
inside the receptive volume. One-line change that unlocks the entire
submission.

## 8. Actuator stiffness override (carried over from v1)

LEAP menagerie defaults give every position actuator `kp = 3.0`. Fine
for the original upright-palm orientation; insufficient when the hand
is flipped 180° about X (palm-down). Post-compile we override every
LEAP actuator to `kp = 25.0`, `kv = 1.0` via `model.actuator_gainprm`
and `model.actuator_biasprm`. Without this the fingers droop and the
REST pose can't be held.

## 9. Strike pose: only MCP extends, PIP/DIP stay bent

This is unchanged from v1 but worth restating: the index finger's
*strike* pose extends MCP from 1.20 → −0.20 rad while PIP stays at
0.90 and DIP at 0.50. This keeps the finger "hooked": the fingertip
arcs down through the bar plane while the medial section stays
*above* the bar plane. Without this the medial would hit the bar
first and the controller would register a strike on a neighbouring
bar.

## 10. Audio: post-render sine synthesis + ffmpeg mux

Each strike event becomes a sine + 2nd + 3rd harmonic with a 220 ms
exponential-decay envelope. 72 strikes are summed into `demo.wav` and
muxed into `demo.mp4` via ffmpeg (discovered through
`shutil.which` with an `imageio_ffmpeg` fallback).

The whole audio pipeline runs once after `simulate()` returns; no
real-time audio constraints, no streaming.

## 11. What was explored but not shipped

- **Polyphony (chords)**. Two fingers strike simultaneously. The
  touch sensors would handle independent rising edges correctly, but
  the audio synth currently expects single-note events. Adding
  additive overlap would be ~ 5 lines.
- **A real arm carrying the hand**. SlipZero uses a Franka arm; mine
  uses a mocap wrist. Mocap is cleaner for a music task (zero
  tracking error by definition) but loses one rubric axis (no arm
  joints to count). A future v3 could swap the mocap for a 1-DoF
  slide joint with a stiff position actuator.
- **Thumb participation**. Thumb anatomy in the palm-down orientation
  drives the tip *sideways* rather than down on strike. Got it
  working in v1 in a kludgy way (different strike pose per finger);
  v2 doesn't need it (mocap covers the full range with a single
  finger).

## 12. Lessons that apply to any LEAP entry

- **Build a calibration spike first.** `spike_calibrate.py`,
  `spike_strike.py`, `spike_mocap.py`, `spike_parked.py` — each one
  resolved a question that would otherwise have cost hours of trial
  guessing.
- **Default actuator stiffness assumes the menagerie's default
  orientation.** Non-default poses (especially gravity inversion) need
  a kp override.
- **Touch sensors silently return zero when their site sphere is too
  small.** Always size sites to enclose the geom being touched.
- **Collision-disable parked geoms** when they live in the workspace
  of moving objects. The simulator will dutifully resolve contacts
  you didn't intend.
- **Closed-form trajectories beat state machines** when the schedule
  is fully known in advance. They're easier to test (any time t → one
  expected position) and easier to reason about.

— Claude
