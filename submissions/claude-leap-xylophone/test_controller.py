"""Unit tests for the LEAP xylophone v2 submission (8-bar mocap-wrist version).

Run with:  python -m pytest submissions/claude-leap-xylophone/test_controller.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pytest
import mujoco

import main as M


def test_scene_compiles_with_mocap_wrist_and_8_bars():
    model = M.build_scene()
    assert model.nu == 16, f"LEAP should expose 16 actuators, got {model.nu}"
    assert model.nmocap == 1, f"need 1 mocap wrist, got {model.nmocap}"
    # 16 finger joints + 8 bar hinges = 24
    assert model.njnt == 24, f"expected 24 joints, got {model.njnt}"


def test_sensors_include_per_bar_touch_and_angle_and_leap_jointpos():
    model = M.build_scene()
    smap = M.get_sensor_map(model)
    for note in M.NOTE_ORDER:
        assert f"bar_{note}_touch" in smap, f"missing touch sensor for bar {note}"
        assert f"bar_{note}_angle" in smap, f"missing angle sensor for bar {note}"
    for j in ["if_mcp", "if_pip", "if_dip",
              "mf_mcp", "mf_pip", "mf_dip",
              "rf_mcp", "rf_pip", "rf_dip",
              "th_cmc", "th_axl", "th_mcp", "th_ipl"]:
        assert f"hand_{j}_sensor" in smap, f"missing LEAP joint sensor: {j}"
    # 8 bars × 2 sensors + 16 LEAP joint sensors = 32
    assert model.nsensor >= 16 + 16


def test_note_frequencies_are_monotonic_and_distinct():
    fs = [M.NOTE_FREQS[n] for n in M.NOTE_ORDER]
    assert fs == sorted(fs), f"NOTE_FREQS must increase along NOTE_ORDER"
    assert len(set(fs)) == len(fs), "all note frequencies must be distinct"


def test_wrist_y_offset_is_consistent_per_note_index():
    # bar_y at note_index i should equal BAR_Y_FIRST + i * BAR_Y_SPACING.
    for i in range(8):
        expected = M.BAR_Y_FIRST + i * M.BAR_Y_SPACING
        assert abs(M.bar_y(i) - expected) < 1e-9
    # wrist_y_for_note(i) places the index fingertip at bar_y(i)
    for i in range(8):
        assert abs(M.wrist_y_for_note(i) - (M.bar_y(i) - M.WRIST_TIP_Y_OFFSET)) < 1e-9


def test_strike_pose_only_changes_index_finger():
    for joint, rest_val in M.REST_POSE.items():
        strike_val = M.STRIKE_POSE[joint]
        if joint.startswith("if_"):
            assert strike_val != rest_val or joint == "if_rot", \
                f"index joint {joint} should change in STRIKE_POSE"
        else:
            assert strike_val == rest_val, \
                f"non-index joint {joint} must NOT change in STRIKE_POSE"


def test_wrist_y_at_returns_first_note_before_start():
    schedule, _pieces, _total = M.schedule_for_tempo(110.0, 0)
    y0 = M.wrist_y_for_note(schedule[0]["note_idx"])
    assert M.wrist_y_at(0.0, schedule) == y0
    assert M.wrist_y_at(schedule[0]["strike_t"] - 0.05, schedule) == y0


def test_wrist_y_at_returns_last_note_after_end():
    schedule, _pieces, _total = M.schedule_for_tempo(110.0, 0)
    yN = M.wrist_y_for_note(schedule[-1]["note_idx"])
    assert M.wrist_y_at(schedule[-1]["strike_t"] + 5.0, schedule) == yN


def test_wrist_y_at_interpolates_between_consecutive_notes():
    schedule, _pieces, _total = M.schedule_for_tempo(110.0, 0)
    # find first pair (a, b) with different note_idx
    a, b = None, None
    for i in range(len(schedule) - 1):
        if schedule[i]["note_idx"] != schedule[i + 1]["note_idx"]:
            a, b = schedule[i], schedule[i + 1]
            break
    assert a is not None, "expected at least one pitch change in the schedule"
    ya = M.wrist_y_for_note(a["note_idx"])
    yb = M.wrist_y_for_note(b["note_idx"])
    # at the slide_end moment the wrist must already be at yb
    y_at_strike = M.wrist_y_at(b["strike_t"] - M.SLIDE_LEAD_S, schedule)
    assert abs(y_at_strike - yb) < 1e-9, \
        f"wrist must finish slide by strike_t - SLIDE_LEAD_S (was {y_at_strike} expected {yb})"


def test_schedule_jitter_is_seed_dependent_and_deterministic():
    s0, _p0, _t0 = M.schedule_for_tempo(110.0, 0)
    s1, _p1, _t1 = M.schedule_for_tempo(110.0, 12345)
    s2, _p2, _t2 = M.schedule_for_tempo(110.0, 12345)
    assert any(a["strike_t"] != b["strike_t"] for a, b in zip(s0, s1))
    assert all(a["strike_t"] == b["strike_t"] for a, b in zip(s1, s2))


def test_concert_program_has_four_named_pieces():
    s, pieces, total_t = M.schedule_for_tempo(110.0, 0)
    assert len(pieces) == 4
    titles = [p["title"] for p in pieces]
    assert any("Mary" in t for t in titles)
    assert any("Twinkle" in t for t in titles)
    assert any("Joy" in t for t in titles)
    assert any("Birthday" in t for t in titles)
    # total_t includes intro + outro
    assert total_t > pieces[-1]["end_t"]


def test_intro_and_outro_durations_are_present():
    assert M.INTRO_DURATION_S > 0
    assert M.OUTRO_DURATION_S > 0
    assert M.INTER_PIECE_S > 0
    s, pieces, _ = M.schedule_for_tempo(110.0, 0)
    assert pieces[0]["start_t"] >= M.INTRO_DURATION_S - 0.01


def test_short_simulation_runs_without_error(monkeypatch):
    # patch CONCERT_PROGRAM down to a single short piece for speed
    monkeypatch.setattr(M, "CONCERT_PROGRAM",
                        [("Spike", [("C", 1), ("E", 1), ("G", 1)], 130.0)])
    r = M.simulate(seed=42, tempo_bpm=130.0,
                   render_video=False, write_jsonl=False)
    assert r.scheduled_n == 3
    assert r.struck_n >= 1


def test_audio_synthesis_produces_a_wav(tmp_path):
    events = [
        {"t": 0.1, "note": "C", "velocity": 0.7},
        {"t": 0.5, "note": "G", "velocity": 0.9},
        {"t": 1.0, "note": "c", "velocity": 0.5},
    ]
    out = tmp_path / "out.wav"
    M.synth_audio(events, total_duration_s=1.5, out_wav=out)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_non_index_finger_collisions_disabled():
    """The mf/rf/th finger collision geoms must be non-collidable so the
    sliding wrist doesn't drag them across the bars."""
    model = M.build_scene()
    bad = []
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if (bname.startswith("hand_") and
                not bname.startswith("hand_palm") and
                not bname.startswith("hand_if_")):
            if model.geom_contype[gid] != 0 or model.geom_conaffinity[gid] != 0:
                bad.append((bname, gname))
    assert not bad, f"non-index finger geoms still collidable: {bad}"
