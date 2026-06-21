"""Debug spike: hold strike pose for index finger, track tip+bar contact."""
from pathlib import Path
import sys
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from main import (build_scene, REST_POSE, strike_pose, apply_pose,
                  get_actuator_map, get_sensor_map, read_sensor, BAR_DEFS, DT)

model = build_scene()
data = mujoco.MjData(model)
name2act = get_actuator_map(model)
sensor_map = get_sensor_map(model)

# settle to rest
apply_pose(data, name2act, REST_POSE)
for _ in range(int(0.5 / DT)):
    apply_pose(data, name2act, REST_POSE)
    mujoco.mj_step(model, data)

if_tip_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hand_if_tip")
print(f"after REST settle: if_tip={data.geom_xpos[if_tip_gid]}")

# now hold strike pose for index for 1.0s
target = strike_pose("if")
print(f"strike target keys: {target}")

n_steps = int(1.5 / DT)
for s in range(n_steps):
    apply_pose(data, name2act, target)
    mujoco.mj_step(model, data)
    if s % 50 == 0:
        touches = {n: float(read_sensor(data, sensor_map, f"bar_{n}_touch")[0])
                   for n, _, _ in BAR_DEFS}
        angles = {n: float(read_sensor(data, sensor_map, f"bar_{n}_angle")[0])
                  for n, _, _ in BAR_DEFS}
        tip = data.geom_xpos[if_tip_gid]
        # also dump if_mcp ctrl & qpos
        if_mcp_act = name2act.get("hand_if_mcp_act", -1)
        if_mcp_qpos_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hand_if_mcp")
        if if_mcp_qpos_id >= 0:
            jadr = model.jnt_qposadr[if_mcp_qpos_id]
            qval = data.qpos[jadr]
        else:
            qval = float('nan')
        print(f"t={s*DT:5.2f}  if_mcp ctrl={data.ctrl[if_mcp_act]:+.3f} "
              f"qpos={qval:+.3f}  tip=({tip[0]:+.3f},{tip[1]:+.3f},{tip[2]:+.3f})  "
              f"touch={touches}  angle E={angles['E']:+.3f}")

# also: any contacts between if_tip and bar_E geom?
print(f"\nfinal contacts: ncon={data.ncon}")
for i in range(min(data.ncon, 20)):
    c = data.contact[i]
    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "?"
    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "?"
    print(f"  contact {i}: {g1}  <->  {g2}   dist={c.dist:+.4f}  pos={c.pos}")
