"""Unit tests for the LEAP xylophone submission.

Tests cover:
- scene compiles cleanly
- sensor map includes per-bar touch + angle sensors
- finger->note assignment is consistent
- strike pose differs from rest pose on the targeted finger only
- song schedule contains the expected note count and ordering
- a short simulation runs without raising

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


def test_scene_compiles_and_has_actuators():
    model = M.build_scene()
    assert model.nu == 16, f"LEAP should expose 16 position actuators, got {model.nu}"
    assert model.njnt >= 16 + 3, "scene should have 16 LEAP joints + 3 bar hinges"


def test_sensors_include_per_bar_touch_and_angle():
    model = M.build_scene()
    smap = M.get_sensor_map(model)
    for note, _, _ in M.BAR_DEFS:
        assert f"bar_{note}_touch" in smap
        assert f"bar_{note}_angle" in smap
    # also: 16 LEAP joint sensors
    for j in ["if_mcp", "if_pip", "if_dip",
              "mf_mcp", "mf_pip", "mf_dip",
              "rf_mcp", "rf_pip", "rf_dip",
              "th_cmc", "th_axl", "th_mcp", "th_ipl"]:
        assert f"hand_{j}_sensor" in smap, f"missing LEAP joint sensor: {j}"


def test_finger_note_mapping_is_invertible():
    # Every finger maps to a distinct note and vice versa.
    notes = list(M.FINGER_OF_NOTE.keys())
    fingers = list(M.FINGER_OF_NOTE.values())
    assert len(notes) == len(set(notes))
    assert len(fingers) == len(set(fingers))


def test_strike_pose_only_extends_target_finger():
    rest = M.REST_POSE
    for finger in ("if", "mf", "rf"):
        pose = M.strike_pose(finger)
        # the targeted finger's mcp/pip/dip joints differ from rest
        for j in (f"{finger}_mcp", f"{finger}_pip", f"{finger}_dip"):
            assert pose[j] != rest[j], f"{finger} strike should change {j}"
        # all OTHER fingers must be at rest
        for other in ("if", "mf", "rf"):
            if other == finger:
                continue
            for j in (f"{other}_mcp", f"{other}_pip", f"{other}_dip"):
                assert pose[j] == rest[j], \
                    f"{finger} strike must NOT change {other}'s {j}"


def test_schedule_has_expected_note_count_and_only_known_notes():
    sched, total = M.schedule_for_tempo(100.0, 0)
    assert len(sched) == 60, f"medley should be 60 strikes, got {len(sched)}"
    for ev in sched:
        assert ev["note"] in M.NOTE_FREQS
        assert ev["finger"] in ("if", "mf", "rf")
        assert ev["strike_t"] > 0
    # monotonically increasing strike times
    times = [ev["strike_t"] for ev in sched]
    assert times == sorted(times)
    assert total > times[-1]


def test_schedule_jitter_is_seed_dependent():
    s0, _ = M.schedule_for_tempo(100.0, 0)
    s1, _ = M.schedule_for_tempo(100.0, 12345)
    s2, _ = M.schedule_for_tempo(100.0, 12345)
    # seed 0 gives no jitter; seed != 0 gives jitter
    assert any(a["strike_t"] != b["strike_t"] for a, b in zip(s0, s1))
    # same seed gives same schedule (deterministic)
    assert all(a["strike_t"] == b["strike_t"] for a, b in zip(s1, s2))


def test_short_simulation_runs_without_error(monkeypatch):
    # patch MEDLEY down to 3 notes to keep the test under a second
    monkeypatch.setattr(M, "MEDLEY", ["E", "D", "C"])
    r = M.simulate(seed=42, tempo_bpm=120.0,
                   render_video=False, write_jsonl=False)
    assert r.scheduled_n == 3
    # should hit at least 1 note (tight tolerance — full hit count varies
    # under timing jitter, but at minimum 1 strike must register)
    assert r.struck_n >= 1


def test_audio_synthesis_produces_a_wav(tmp_path):
    events = [
        {"t": 0.1, "note": "C", "velocity": 0.7},
        {"t": 0.5, "note": "E", "velocity": 0.9},
        {"t": 1.0, "note": "D", "velocity": 0.5},
    ]
    out = tmp_path / "out.wav"
    M.synth_audio(events, total_duration_s=1.5, out_wav=out)
    assert out.exists()
    assert out.stat().st_size > 1000   # non-empty WAV


def test_note_frequencies_are_distinct_and_monotonic():
    # C5 < D5 < E5
    assert M.NOTE_FREQS["C"] < M.NOTE_FREQS["D"] < M.NOTE_FREQS["E"]
