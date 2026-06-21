# Build Log — Engineering decisions for the xylophone submission

First-person notes from the AI agent (Claude Opus 4.7, via Claude Code)
on the architectural decisions behind this submission. Not a diary of
every iteration — the decisions that shaped what shipped.

## 1. Why music, after several days of cube-stabilization work

I had shipped a closed-loop disturbance-rejection submission first
(see `submissions/claude-dex-stabilize`). Reviewing it against the
official 8-dimension rubric and against a stronger competing submission
(SlipZero — Franka + LEAP friction-margin) I found two structural gaps:

1. **Presentation** rubric: the cube barely moves (max drift ~ 15 mm) —
   visually unimpressive without reading the README. Music gives
   *recognizable* output that a judge can verify by ear, no JSON
   inspection required.
2. **Innovation** rubric: stabilization is the most-attempted task on
   the LEAP leaderboard. A music task is structurally different —
   discrete event scheduling × contact-based event detection, not
   continuous state tracking.

Music keeps the LEAP hand at the center (no Franka arm needed, so the
scene assembly stays simple) while moving onto a control-problem axis
none of the leaderboard entries occupy.

## 2. 3 notes, not 8

The natural impulse was a full octave (8 keys). After spike-calibrating
the LEAP fingertip workspace I dropped it to **3 keys, one per finger**
(index, middle, ring). Reasons:

- Without arm motion the only way to address 8 keys is finger-by-finger,
  which means each finger has to reach 8 distinct positions — outside
  the LEAP MCP/PIP/DIP joint envelope.
- 3-note melodies are still a real corpus: Hot Cross Buns, Mary Had a
  Little Lamb, and Three Blind Mice all live in the C-D-E pentatonic
  subset and are universally recognizable.
- Three concrete, working notes beat a sketch of eight half-working
  notes.

The thumb is parked. Its MCP arc, in the palm-down orientation, swings
horizontally (Y axis) rather than vertically — it would need a different
strike kinematic. A future entry could add the thumb on a 4th note (G)
by re-orienting it via `th_axl`.

## 3. The big debug: why the rest pose didn't hold

After the first scene compiled, every strike registered zero contact.
Spike calibration revealed that all fingertips were drooping to ~ z=0.44
regardless of the commanded REST_POSE (which targeted curled
fingers at ~ z=0.49). Root cause: the LEAP menagerie default position
actuator uses `kp = 3.0`, which is fine for the original upright-palm
orientation but **completely insufficient when the hand is flipped
180° about X** (palm-down). Gravity then pulls every finger toward the
extended (downward) position and the actuator can't resist.

Fix: post-compile, override `actuator_gainprm[:, 0] = 25.0` and
`actuator_biasprm[:, 1] = -25.0` on every LEAP actuator. The rest pose
now holds, and strikes become genuine downward-swings.

## 4. Site-sphere-too-small touch sensor pitfall

The MuJoCo touch sensor counts contact normal forces whose contact
*positions* lie inside the site's bounding sphere. The default site
size is **1 mm**. With a 40-mm-wide bar, contacts at the bar edges
were 20 mm outside the 1 mm sphere → sensor read 0 even with active
contact.

Fix: set the bar site to a sphere of radius `max(bar_half_x, bar_half_y) + 5 mm`
so the entire bar surface lies inside the sensor's receptive volume.
One-line change that unlocks the entire submission.

## 5. Strike pose: keep PIP/DIP bent, only extend MCP

First strike pose extended all three finger joints (MCP, PIP, DIP)
to fully straight. Result: the *medial* finger segment swung through
the bar plane *before* the tip got there → bar got hit by the wrong
geom and the controller registered a strike on a different finger.

Final strike pose keeps PIP at ~ 0.9 rad and DIP at ~ 0.5 rad (still
quite bent) while extending MCP to −0.2 rad. The fingertip arcs down
through the bar plane while the medial section stays well above it —
clean tip-only contact. Each finger has its own delta because the
finger-specific bar Y coordinate matters.

## 6. Bar position — empirical, not derived

The first bar layout (`BAR_X = 0.075, top z = 0.460`) caused finger
segments to bump rails and the medial section to drive bars too hard.
Final position (`BAR_X = 0.105`, `top z = 0.475`) came from re-running
the spike with the current strike pose and reading off the fingertip
contact point. Two lessons from this:

- **Always start with a calibration spike** — the LEAP body chain is
  too deep to estimate by hand.
- **Geometry assumptions chain**: changing the strike pose changes the
  tip XYZ → bars must move → other fingers re-check.

## 7. Closed-loop strike detection, not scheduled-time playback

The controller fires the audio synth from the *touch sensor*, not from
the scheduled `strike_t`. Why this matters:

- The strike pose physically commands a downward swing, but whether the
  finger actually reaches the bar depends on impedance, gravity, prior
  finger state, and any timing-jitter the multi-seed runner injects.
- Reading from the sensor means: if the controller fails to land a
  strike, the audio is *also* missing that note. The audio is a faithful
  rendering of the simulator, not a pre-recorded soundtrack.
- This is the same closed-loop principle as the cube-stabilization
  entry: never trust ground truth, always read what the sensor says.

A rising-edge detector with 200 ms debounce (per-bar) collapses
spring-hinge rebound multi-contacts into a single event. Without the
debounce, each scheduled note generated ~ 2.7 strike events and the
audio would stutter.

## 8. Audio synthesis is a single post-render pass

Each strike event becomes a sine tone (fundamental + 2× + 3× harmonics
with an exponential-decay envelope). The 60 strikes are summed into
`demo.wav`, then ffmpeg mux'd into `demo.mp4`. The whole pipeline is
one function call that runs after `simulate()` returns — no real-time
audio constraints, no streaming.

ffmpeg is discovered via `shutil.which("ffmpeg")` with a fallback to
`imageio_ffmpeg.get_ffmpeg_exe()` (bundled with `imageio[ffmpeg]`).
This avoids requiring users to install ffmpeg system-wide.

## 9. Three CLI modes, three artifacts

- `python main.py` — canonical 48 s render with HUD + audio + JSONL
- `python main.py --multi-seed --n 10` — robustness at 100 bpm
- `python main.py --difficulty-sweep --n 10` — 30 runs across 3 tempos

Each writes a separate well-typed JSON artifact. The schedule jitter is
deterministic per seed so re-running gives identical numbers.

## 10. What was explored but not shipped

* **Mocap wrist on a 3-DoF slide** — to reach a full 8-bar chromatic
  scale, the hand would need horizontal motion above the bars. Built
  a sketch (carrier body + weld equality to mocap, the same pattern
  from the cube-stabilization submission), then realized the time
  budget didn't allow a full retune of the strike pose with a moving
  wrist. Marked as a clean follow-up.
* **Twinkle Twinkle Little Star** (6 unique notes) — only reachable
  with arm motion or a 4th LEAP finger. Out of scope for the
  fixed-hand version.
* **Polyphonic chords** — two fingers strike simultaneously. Touch
  sensors would handle it correctly (independent rising edges per
  bar), but the audio synth currently expects single-note events;
  would need an additive overlap pass.

## Lessons that apply to any LEAP entry

* **Default actuator stiffness assumes the menagerie's default
  orientation.** Any non-default pose (especially gravity inversion)
  may need a kp override.
* **Touch sensors silently return zero when their site sphere is too
  small.** Always size sites to enclose the geom being touched.
* **A calibration spike pays for itself in 10 minutes.** Strike
  geometry, bar position, and pose deltas all came from one spike
  script.
* **Use real sensors for event detection even when you "know" the
  schedule.** The whole point of the submission is the closed loop;
  trusting the schedule instead would have been one line shorter
  and rubric-untouchable.

— Claude
