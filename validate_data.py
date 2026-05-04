import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import os
import sys
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_dir = os.path.dirname(os.path.abspath(__file__))
gp_path = os.path.abspath(os.path.join(current_dir, "GP_pred"))
sys.path.append(gp_path)

data_path = "data/aggregated/rl_trajectories.pkl"
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


with open(data_path, "rb") as f:
    rl_trajectories = pickle.load(f)


all_trajectories = []
# rl_trajectories = []

if not os.path.exists(args.root_folder):
    raise FileNotFoundError(f"Root folder not found: {args.root_folder}")

traj_names = sorted(
    [x for x in os.listdir(args.root_folder) if os.path.isdir(os.path.join(args.root_folder, x))],
    key=lambda x: int(x) if x.isdigit() else x
)

print(f"number of trajectories found: {len(traj_names)}")

example_traj = rl_trajectories[2]
print(example_traj.head())
print(example_traj.tail())