import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# --- Hyperparameters ---
gamma = 0.99
lr = 1e-3
num_epochs = 10
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define Q-network ---
class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)  # outputs Q-values for both actions

q_net = QNetwork().to(device)
optimizer = optim.Adam(q_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# --- Prepare training data from trajectories ---
with open('data/aggregated/rl_trajectories.pkl', 'rb') as f:
    rl_trajectories = pickle.load(f)
states_list = []
actions_list = []
rewards_list = []
next_states_list = []

for traj in rl_trajectories:
    states = traj[["walker_vel_ms","ego_vel_ms","dx","dy"]].values
    actions = traj["action_trigger"].values
    rewards = traj["reward"].values
    
    for t in range(len(traj)-1):
        states_list.append(states[t])
        actions_list.append(actions[t])
        rewards_list.append(rewards[t])
        next_states_list.append(states[t+1])

# Convert to tensors
states_tensor = torch.tensor(np.array(states_list), dtype=torch.float32).to(device)
actions_tensor = torch.tensor(np.array(actions_list), dtype=torch.int64).to(device)
rewards_tensor = torch.tensor(np.array(rewards_list), dtype=torch.float32).to(device)
next_states_tensor = torch.tensor(np.array(next_states_list), dtype=torch.float32).to(device)

# --- Offline Q-learning training ---
dataset_size = len(states_tensor)
print("Training on", dataset_size, "samples")

for epoch in range(num_epochs):
    perm = np.random.permutation(dataset_size)
    for i in range(0, dataset_size, batch_size):
        idx = perm[i:i+batch_size]
        
        s_batch = states_tensor[idx]
        a_batch = actions_tensor[idx]
        r_batch = rewards_tensor[idx]
        s_next_batch = next_states_tensor[idx]
        
        # Current Q-values
        q_values = q_net(s_batch)  # shape: [batch, 2]
        q_a = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            q_next = q_net(s_next_batch)
            q_next_max, _ = q_next.max(dim=1)
            target = r_batch + gamma * q_next_max
        
        # Loss & optimization
        loss = loss_fn(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}")

# --- Example Q-value prediction ---
state_example = torch.tensor([50, 40, 1.2, 0.5], dtype=torch.float32).unsqueeze(0).to(device)
q_values_example = q_net(state_example)
print("Q-values for actions 0 and 1:", q_values_example.detach().cpu().numpy())