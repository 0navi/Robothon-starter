"""Spike: check parked finger positions in REST_POSE vs bar Z plane."""
from pathlib import Path
import sys
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from main import (build_scene, REST_POSE, apply_pose,
                  get_actuator_map, BAR_Z_TOP, BAR_HALF_Z, DT,
                  wrist_y_for_note)

model = build_scene()
data = mujoco.MjData(model)
name2act = get_actuator_map(model)

# wrist at note 0 (C)
wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
wrist_mocap = int(model.body_mocapid[wrist_bid])
data.mocap_pos[wrist_mocap] = [0.0, wrist_y_for_note(0), 0.30]

apply_pose(data, name2act, REST_POSE)
for _ in range(int(0.5 / DT)):
    apply_pose(data, name2act, REST_POSE)
    mujoco.mj_step(model, data)

bar_top = BAR_Z_TOP + BAR_HALF_Z
print(f"bar top z = {bar_top:.3f}")
print(f"bar y range (top of bar plane): {bar_top}\n")

tips = ["hand_if_tip", "hand_mf_tip", "hand_rf_tip", "hand_th_tip"]
for tname in tips:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, tname)
    p = data.geom_xpos[gid]
    clearance = p[2] - bar_top
    print(f"  {tname}: x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}  "
          f"clearance={clearance*1000:+.1f} mm")

# also check medial / proximal segments for index (these would be lower than tip in curled pose)
extras = ["hand_if_px_visual", "hand_if_md_visual",
          "hand_mf_px_visual", "hand_mf_md_visual",
          "hand_rf_px_visual", "hand_rf_md_visual",
          "hand_th_bs_visual", "hand_th_px_visual", "hand_th_ds_visual"]
print("\n--- segment geoms ---")
for tname in extras:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, tname)
    if gid < 0:
        continue
    p = data.geom_xpos[gid]
    clearance = p[2] - bar_top
    print(f"  {tname}: y={p[1]:+.4f}  z={p[2]:+.4f}  clr={clearance*1000:+.1f} mm")

# now check contacts
print(f"\nncon: {data.ncon}")
for i in range(min(data.ncon, 20)):
    c = data.contact[i]
    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "?"
    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "?"
    if "bar" in g1 or "bar" in g2:
        print(f"  bar contact: {g1} <-> {g2}  dist={c.dist:+.4f}")
