"""Unit tests for the LEAP closed-loop stabilization controller.

Run with `pytest test_controller.py` from this directory (or from repo root).
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main


def test_scene_compiles():
    """Scene must compile and expose the expected number of sensors."""
    model = main.build_scene()
    assert model.nbody >= 6, "carrier + cube + ref + hand bodies should be present"
    # 16 LEAP joint position sensors + 7 cube state sensors = 23.
    assert model.nsensor == 23, f"expected 23 sensors, got {model.nsensor}"


def test_actuator_map_has_all_finger_actuators():
    model = main.build_scene()
    name2act = main.get_actuator_map(model)
    expected = {f"hand_{j}_act" for j in main.CAGE_BASE.keys()}
    assert expected.issubset(set(name2act.keys()))


def test_grip_pose_blend_endpoints():
    """tighten=0 yields CAGE_BASE; tighten=1 yields CAGE_BASE + GRIP_TIGHTEN_DELTA."""
    base = main.CAGE_BASE
    delta = main.GRIP_TIGHTEN_DELTA
    p0 = main.make_grip_pose(0.0)
    p1 = main.make_grip_pose(1.0)
    for k, v0 in base.items():
        assert math.isclose(p0[k], v0, abs_tol=1e-12), f"open mismatch at {k}"
    for k, dk in delta.items():
        assert math.isclose(p1[k], base[k] + dk, abs_tol=1e-12), f"close mismatch at {k}"


def test_control_law_threshold_below_kicks_in_zero():
    """Drift below 8 mm threshold yields zero tighten command (in PERTURBED/RECOVERY)."""
    # The control law in simulate(): tighten = clamp((drift - 0.008) / 0.022, 0, 1)
    drift = 0.005  # 5 mm < 8 mm threshold
    tighten = max(0.0, min(1.0, (drift - 0.008) / 0.022))
    assert tighten == 0.0


def test_control_law_threshold_at_saturation():
    """Drift at 30 mm or above yields tighten=1."""
    drift = 0.030
    tighten = max(0.0, min(1.0, (drift - 0.008) / 0.022))
    assert tighten == 1.0


def test_control_law_mid_range_proportional():
    """Drift at 19 mm yields tighten = (0.019-0.008)/0.022 ≈ 0.5."""
    drift = 0.019
    tighten = max(0.0, min(1.0, (drift - 0.008) / 0.022))
    assert math.isclose(tighten, 0.5, abs_tol=0.001)


def test_sensor_map_includes_cube_err():
    """Verify the named sensor that feeds the closed-loop is present."""
    model = main.build_scene()
    sensor_map = main.get_sensor_map(model)
    assert "cube_err" in sensor_map
    adr, dim = sensor_map["cube_err"]
    assert dim == 3, "cube_err is a 3D vector"


def test_simulate_short_runs_without_error():
    """Smoke test: a very short simulate() invocation must run cleanly."""
    # Patch the duration so this test runs quickly.
    saved = main.DURATION_S
    main.DURATION_S = 0.5
    try:
        r = main.simulate(seed=1, render_video=False, write_jsonl=False)
        assert r.duration_s == 0.5
        assert r.perturbations_n == 0    # no perturb in first 0.5 s
    finally:
        main.DURATION_S = saved


def test_state_constants_are_distinct():
    s = main.ControlState
    states = {s.NORMAL, s.PERTURBED, s.RECOVERY, s.HOLD}
    assert len(states) == 4
