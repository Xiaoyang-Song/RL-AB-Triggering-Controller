import argparse
import numpy as np
import pickle
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
gp_path = os.path.abspath(os.path.join(current_dir, "GP_pred"))
sys.path.append(gp_path)

from ford_ped_backend_single import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Testing injury risk functions for airbag triggering."
    )

    # Demographic / model inputs
    parser.add_argument("--height", type=float, default=1.65)
    parser.add_argument("--sex", type=str, default="F")
    parser.add_argument("--bmi", type=float, default=27.13)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--orientation", type=float, default=-90.0)
    parser.add_argument("--vstiffness", type=float, default=0.956)
    parser.add_argument("--vtype", type=str, default="SUV")
    parser.add_argument("--age", type=float, default=30.0)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="test")

    return parser.parse_args()

def compute_joint_injury_risk(service, row, args):
    """
    Compute current predicted joint injury risk from the current state row.
    Assumes the backend model can use current ego/pedestrian speeds as proxies.
    """
    collision_v_ego = float(row["ego_vel_ms"]) * 3.6
    collision_v_ped = float(row["walker_vel_ms"]) * 3.6

    data = np.array([
        0,
        args.height,
        args.sex,
        float(args.bmi),
        float(args.offset),
        float(args.orientation),
        float(collision_v_ped),
        float(args.vstiffness),
        args.vtype,
        float(args.age),
        float(collision_v_ego),
    ], dtype=object)

    ir = service.predict_injury(data)
    return float(pjoint(ir))


args = parse_args()
np.random.seed(args.seed)

service = ford_ped_calc_service()

v_list = np.arange(10, 70, 2)
iir_list = []

for v in v_list:
    row = {
        "walker_vel_ms": 1.2,
        "ego_vel_ms": v / 3.6,
    }
    iir = compute_joint_injury_risk(service, row, args)
    iir_list.append(iir)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(v_list, iir_list, marker='o')
plt.title("Predicted Joint Injury Risk vs. Ego Vehicle Speed")
plt.xlabel("Ego Vehicle Speed (km/h)")
plt.ylabel("Predicted Joint Injury Risk")
plt.grid()
plt.savefig(f"figure/pjoint_vs_speed_{args.tag}.png")