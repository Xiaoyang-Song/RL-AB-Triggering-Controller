import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import os

# --- Load trajectories ---
with open('data/aggregated/rl_trajectories.pkl', 'rb') as f:
    rl_trajectories = pickle.load(f)

# --- Split by trajectory ---
num_traj = len(rl_trajectories)

# Visualize trajectory 
for traj in rl_trajectories[0:3]:
    df = pd.DataFrame(traj)
    # print(df.head())
    # print out last 5 rows to check collision status
    print(df.tail())
    # break