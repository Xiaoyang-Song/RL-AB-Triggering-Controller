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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Hyperparameters ---
gamma = 0.99
lr = 1e-3
num_epochs = 200
batch_size = 128
val_ratio = 0.1
seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --- Define Q-network ---
class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

q_net = QNetwork().to(device)
optimizer = optim.Adam(q_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# --- Load trajectories ---
with open('data/aggregated/rl_trajectories.pkl', 'rb') as f:
    rl_trajectories = pickle.load(f)

# --- Split by trajectory ---
num_traj = len(rl_trajectories)
indices = np.random.permutation(num_traj)
split = int((1 - val_ratio) * num_traj)

train_idx = indices[:split]
val_idx = indices[split:]

train_traj = [rl_trajectories[i] for i in train_idx]
val_traj = [rl_trajectories[i] for i in val_idx]

print(f"Train trajectories: {len(train_traj)}")
print(f"Validation trajectories: {len(val_traj)}")

# --- Function to convert trajectories to tensors ---
def build_dataset(trajectories):
    states_list, actions_list, rewards_list, next_states_list = [], [], [], []
    
    for traj in trajectories:
        states = traj[["walker_vel_ms","ego_vel_ms","dx","dy"]].values
        actions = traj["action_trigger"].values
        rewards = traj["reward"].values
        
        for t in range(len(traj)-1):
            states_list.append(states[t])
            actions_list.append(actions[t])
            rewards_list.append(rewards[t])
            next_states_list.append(states[t+1])

    return (
        torch.tensor(np.array(states_list), dtype=torch.float32).to(device),
        torch.tensor(np.array(actions_list), dtype=torch.int64).to(device),
        torch.tensor(np.array(rewards_list), dtype=torch.float32).to(device),
        torch.tensor(np.array(next_states_list), dtype=torch.float32).to(device),
    )

train_states, train_actions, train_rewards, train_next_states = build_dataset(train_traj)
val_states, val_actions, val_rewards, val_next_states = build_dataset(val_traj)

print("Training samples:", len(train_states))
print("Validation samples:", len(val_states))

# --- Training ---
train_losses = []
val_losses = []

for epoch in tqdm(range(num_epochs)):

    # ---- Training ----
    q_net.train()
    perm = torch.randperm(len(train_states))
    batch_losses = []

    for i in range(0, len(train_states), batch_size):
        idx = perm[i:i+batch_size]

        s_batch = train_states[idx]
        a_batch = train_actions[idx]
        r_batch = train_rewards[idx]
        s_next_batch = train_next_states[idx]

        q_values = q_net(s_batch)
        q_a = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = q_net(s_next_batch)
            q_next_max = q_next.max(dim=1)[0]
            target = r_batch + gamma * q_next_max

        loss = loss_fn(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)

    # ---- Validation ----
    q_net.eval()
    with torch.no_grad():
        q_values = q_net(val_states)
        q_a = q_values.gather(1, val_actions.unsqueeze(1)).squeeze(1)

        q_next = q_net(val_next_states)
        q_next_max = q_next.max(dim=1)[0]
        target = val_rewards + gamma * q_next_max

        val_loss = loss_fn(q_a, target).item()
        val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

checkpoint_path = os.path.join('checkpoints', 'model', 'q_net.pth')
torch.save(q_net.state_dict(), checkpoint_path)

# --- Plot ---
plt.figure()
plt.plot(train_losses, label="Train Loss")
# plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Offline Q-Learning: Training Loss")
plt.legend()
# plt.show()
plt.savefig("Q_loss.png")

# --- Example Q prediction ---
state_example = torch.tensor([50, 40, 1.2, 0.5], dtype=torch.float32).unsqueeze(0).to(device)
q_values_example = q_net(state_example)
print("Q-values:", q_values_example.detach().cpu().numpy())