import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================
# Hyperparameters
# =========================
gamma = 0.99
lr = 1e-3
num_epochs = 50
batch_size = 128
val_ratio = 0.1
seed = 42
hidden_dim = 64
target_update_freq = 5
checkpoint_path = os.path.join("checkpoints", "model", "q_net.pth")
loss_plot_path = "Q_loss.png"
data_path = "data/aggregated/rl_trajectories.pkl"

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================
# Q-network
# =========================
class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Utilities
# =========================
def set_seed(seed_value: int):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def build_dataset(trajectories):
    """
    Convert list of trajectories into transition tuples:
    (state, action, reward, next_state, done)

    Important:
    - Includes terminal step
    - For terminal step, next_state is zeros
    - done = 1 for terminal step
    """
    states_list = []
    actions_list = []
    rewards_list = []
    next_states_list = []
    dones_list = []

    for traj in trajectories:
        if len(traj) == 0:
            continue

        states = traj[["walker_vel_ms", "ego_vel_ms", "dx", "dy"]].values.astype(np.float32)
        actions = traj["action_trigger"].values.astype(np.int64)
        rewards = traj["reward"].values.astype(np.float32)

        T = len(traj)

        for t in range(T):
            states_list.append(states[t])
            actions_list.append(actions[t])
            rewards_list.append(rewards[t])

            if t == T - 1:
                next_states_list.append(np.zeros_like(states[t], dtype=np.float32))
                dones_list.append(1.0)
            else:
                next_states_list.append(states[t + 1])
                dones_list.append(0.0)

    states = torch.tensor(np.array(states_list), dtype=torch.float32)
    actions = torch.tensor(np.array(actions_list), dtype=torch.int64)
    rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states_list), dtype=torch.float32)
    dones = torch.tensor(np.array(dones_list), dtype=torch.float32)

    return states, actions, rewards, next_states, dones


def normalize_states(train_states, train_next_states, val_states, val_next_states):
    """
    Standardize states using train-set statistics only.
    """
    mean = train_states.mean(dim=0, keepdim=True)
    std = train_states.std(dim=0, keepdim=True).clamp_min(1e-6)

    train_states = (train_states - mean) / std
    train_next_states = (train_next_states - mean) / std
    val_states = (val_states - mean) / std
    val_next_states = (val_next_states - mean) / std

    return train_states, train_next_states, val_states, val_next_states, mean, std


def evaluate_policy(q_net, trajectories, state_mean, state_std, eta=0.0):
    """
    Simple policy-level summary, not just Bellman loss.
    Assumes:
    - action 1 = trigger
    - terminal reward is on the last row
    - high-risk missed trigger often has negative terminal reward
    """
    q_net.eval()

    total_steps = 0
    trigger_steps = 0
    terminal_count = 0
    terminal_trigger_count = 0
    terminal_no_trigger_count = 0

    with torch.no_grad():
        for traj in trajectories:
            if len(traj) == 0:
                continue

            states = traj[["walker_vel_ms", "ego_vel_ms", "dx", "dy"]].values.astype(np.float32)
            states = torch.tensor(states, dtype=torch.float32, device=device)
            states = (states - state_mean.to(device)) / state_std.to(device)

            q_values = q_net(states)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

            total_steps += len(actions)
            trigger_steps += (actions == 1).sum()

            terminal_action = actions[-1]
            terminal_count += 1
            if terminal_action == 1:
                terminal_trigger_count += 1
            else:
                terminal_no_trigger_count += 1

    metrics = {
        "trigger_rate_all_steps": trigger_steps / max(total_steps, 1),
        "terminal_trigger_rate": terminal_trigger_count / max(terminal_count, 1),
        "terminal_no_trigger_rate": terminal_no_trigger_count / max(terminal_count, 1),
    }
    return metrics


# =========================
# Main
# =========================
def main():
    set_seed(seed)

    # ---- Load trajectories ----
    with open(data_path, "rb") as f:
        rl_trajectories = pickle.load(f)

    num_traj = len(rl_trajectories)
    indices = np.random.permutation(num_traj)
    split = int((1 - val_ratio) * num_traj)

    train_idx = indices[:split]
    val_idx = indices[split:]

    train_traj = [rl_trajectories[i] for i in train_idx]
    val_traj = [rl_trajectories[i] for i in val_idx]

    print(f"Train trajectories: {len(train_traj)}")
    print(f"Validation trajectories: {len(val_traj)}")

    # ---- Build datasets ----
    train_states, train_actions, train_rewards, train_next_states, train_dones = build_dataset(train_traj)
    val_states, val_actions, val_rewards, val_next_states, val_dones = build_dataset(val_traj)

    print("Training samples:", len(train_states))
    print("Validation samples:", len(val_states))

    # ---- Normalize ----
    train_states, train_next_states, val_states, val_next_states, state_mean, state_std = normalize_states(
        train_states, train_next_states, val_states, val_next_states
    )

    # Move to device
    train_states = train_states.to(device)
    train_actions = train_actions.to(device)
    train_rewards = train_rewards.to(device)
    train_next_states = train_next_states.to(device)
    train_dones = train_dones.to(device)

    val_states = val_states.to(device)
    val_actions = val_actions.to(device)
    val_rewards = val_rewards.to(device)
    val_next_states = val_next_states.to(device)
    val_dones = val_dones.to(device)

    # ---- Networks ----
    q_net = QNetwork(hidden_dim=hidden_dim).to(device)
    target_net = QNetwork(hidden_dim=hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    # =========================
    # Training loop
    # =========================
    for epoch in tqdm(range(num_epochs), desc="Training"):
        q_net.train()
        perm = torch.randperm(len(train_states), device=device)
        batch_losses = []

        for i in range(0, len(train_states), batch_size):
            idx = perm[i:i + batch_size]

            s_batch = train_states[idx]
            a_batch = train_actions[idx]
            r_batch = train_rewards[idx]
            s_next_batch = train_next_states[idx]
            done_batch = train_dones[idx]

            q_values = q_net(s_batch)
            q_a = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = target_net(s_next_batch)
                q_next_max = q_next.max(dim=1)[0]
                target = r_batch + gamma * (1.0 - done_batch) * q_next_max

            loss = loss_fn(q_a, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        train_losses.append(train_loss)

        # ---- Validation ----
        q_net.eval()
        with torch.no_grad():
            q_values = q_net(val_states)
            q_a = q_values.gather(1, val_actions.unsqueeze(1)).squeeze(1)

            q_next = target_net(val_next_states)
            q_next_max = q_next.max(dim=1)[0]
            target = val_rewards + gamma * (1.0 - val_dones) * q_next_max

            val_loss = loss_fn(q_a, target).item()
            val_losses.append(val_loss)

        # ---- Update target network ----
        if (epoch + 1) % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        # ---- Policy summary ----
        val_metrics = evaluate_policy(q_net, val_traj, state_mean, state_std)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Trigger Rate: {val_metrics['trigger_rate_all_steps']:.4f} | "
            f"Val Terminal Trigger Rate: {val_metrics['terminal_trigger_rate']:.4f}"
        )

    # =========================
    # Save model + normalization stats
    # =========================
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": q_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "state_mean": state_mean.cpu(),
            "state_std": state_std.cpu(),
            "hyperparameters": {
                "gamma": gamma,
                "lr": lr,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "val_ratio": val_ratio,
                "seed": seed,
                "hidden_dim": hidden_dim,
                "target_update_freq": target_update_freq,
            },
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to: {checkpoint_path}")

    # =========================
    # Plot losses
    # =========================
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Offline Q-Learning Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    print(f"Saved loss plot to: {loss_plot_path}")

    # =========================
    # Example Q prediction
    # =========================
    state_example = torch.tensor([[50.0, 40.0, 1.2, 0.5]], dtype=torch.float32).to(device)
    state_example = (state_example - state_mean.to(device)) / state_std.to(device)

    q_net.eval()
    with torch.no_grad():
        q_values_example = q_net(state_example)

    print("Example Q-values:", q_values_example.cpu().numpy())
    print("Recommended action:", int(torch.argmax(q_values_example, dim=1).item()))


if __name__ == "__main__":
    main()