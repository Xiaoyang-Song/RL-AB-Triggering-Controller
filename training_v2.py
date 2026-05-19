import os
import random
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================
# Argparse
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Offline Q-Learning for airbag triggering.")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/aggregated")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/model")
    parser.add_argument("--loss_plot_dir", type=str, default="plots")

    # Reward / eta parameters (used only for resolving filenames)
    parser.add_argument("--b1", type=float, default=2.0)
    parser.add_argument("--c1", type=float, default=1.5)
    parser.add_argument("--b2", type=float, default=1.0)
    parser.add_argument("--c2", type=float, default=3.0)
    parser.add_argument("--c3", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.2)

    # Training hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--target_update_freq", type=int, default=5)
    parser.add_argument("--lr_step_size", type=int, default=100)
    parser.add_argument("--lr_gamma", type=float, default=0.5)

    # Input uncertainty during training
    parser.add_argument("--train_noise_std", type=float, default=0.0,
                        help="Std dev of Gaussian noise added to normalized states each batch "
                             "(0 = disabled). Encourages robustness to sensor uncertainty.")

    return parser.parse_args()


def build_param_suffix(args):
    base = f"b1{args.b1}_c1{args.c1}_b2{args.b2}_c2{args.c2}_c3{args.c3}_eta{args.eta}"
    full = base + (f"_noise{args.train_noise_std}" if args.train_noise_std > 0.0 else "")
    return base, full


def build_paths(args):
    base_suffix, full_suffix = build_param_suffix(args)
    data_path = os.path.join(args.data_dir, f"rl_trajectories_{base_suffix}.pkl")
    checkpoint_path = os.path.join(args.checkpoint_dir, f"q_net_{full_suffix}.pth")
    loss_plot_path = os.path.join(args.loss_plot_dir, f"Q_loss_{full_suffix}.png")
    return data_path, checkpoint_path, loss_plot_path


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
    (state, action, reward, next_state, done, eventual_collision, high_risk)

    eventual_collision and high_risk are trajectory-level labels
    broadcast to every step within the trajectory.
    """
    states_list = []
    actions_list = []
    rewards_list = []
    next_states_list = []
    dones_list = []
    eventual_collision_list = []
    high_risk_list = []

    for traj in trajectories:
        if len(traj) == 0:
            continue

        states = traj[["walker_vel_ms", "ego_vel_ms", "dx", "dy"]].values.astype(np.float32)
        actions = traj["action_trigger"].values.astype(np.int64)
        rewards = traj["reward"].values.astype(np.float32)
        T = len(traj)

        # Trajectory-level ground truth — constant across all steps
        ec = int(traj["eventual_collision"].iloc[-1])
        hr = int(traj["high_risk"].iloc[-1])

        for t in range(T):
            states_list.append(states[t])
            actions_list.append(actions[t])
            rewards_list.append(rewards[t])
            eventual_collision_list.append(ec)
            high_risk_list.append(hr)

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
    eventual_collision = torch.tensor(eventual_collision_list, dtype=torch.int64)
    high_risk = torch.tensor(high_risk_list, dtype=torch.int64)

    return states, actions, rewards, next_states, dones, eventual_collision, high_risk


def normalize_states(train_states, train_next_states, val_states, val_next_states):
    """Standardize states using train-set statistics only."""
    mean = train_states.mean(dim=0, keepdim=True)
    std = train_states.std(dim=0, keepdim=True).clamp_min(1e-6)

    train_states = (train_states - mean) / std
    train_next_states = (train_next_states - mean) / std
    val_states = (val_states - mean) / std
    val_next_states = (val_next_states - mean) / std

    return train_states, train_next_states, val_states, val_next_states, mean, std


def evaluate_errors(q_net, trajectories, state_mean, state_std):
    """
    Type-I  (false positive): model triggers when it shouldn't
             — low injury risk collision, OR no collision at all
    Type-II (false negative): model never triggers when it should
             — high injury risk collision with no trigger fired

    Decision is whether the model triggers at ANY point in the trajectory,
    which mirrors real deployment (airbag fires on first trigger action).
    """
    q_net.eval()

    type1_count, type1_denom = 0, 0
    type2_count, type2_denom = 0, 0

    with torch.no_grad():
        for traj in trajectories:
            if len(traj) == 0:
                continue

            eventual_collision = bool(traj["eventual_collision"].iloc[-1])
            high_risk = bool(traj["high_risk"].iloc[-1])

            states = traj[["walker_vel_ms", "ego_vel_ms", "dx", "dy"]].values.astype(np.float32)
            states = torch.tensor(states, dtype=torch.float32, device=device)
            states = (states - state_mean.to(device)) / state_std.to(device)

            q_values = q_net(states)
            triggered = bool((torch.argmax(q_values, dim=1) == 1).any().item())

            should_trigger = eventual_collision and high_risk

            if should_trigger:
                type2_denom += 1
                if not triggered:
                    type2_count += 1
            else:
                type1_denom += 1
                if triggered:
                    type1_count += 1

    return {
        "type1_error": type1_count / max(type1_denom, 1),
        "type2_error": type2_count / max(type2_denom, 1),
        "type1_count": type1_count,
        "type1_denom": type1_denom,
        "type2_count": type2_count,
        "type2_denom": type2_denom,
    }


# =========================
# Main
# =========================
def main():
    args = parse_args()
    set_seed(args.seed)

    data_path, checkpoint_path, loss_plot_path = build_paths(args)
    print(f"Loading data from: {data_path}")

    # ---- Load trajectories ----
    with open(data_path, "rb") as f:
        rl_trajectories = pickle.load(f)

    num_traj = len(rl_trajectories)
    indices = np.random.permutation(num_traj)
    split = int((1 - args.val_ratio) * num_traj)

    train_traj = [rl_trajectories[i] for i in indices[:split]]
    val_traj = [rl_trajectories[i] for i in indices[split:]]

    print(f"Train trajectories: {len(train_traj)}")
    print(f"Validation trajectories: {len(val_traj)}")

    # ---- Build datasets ----
    (train_states, train_actions, train_rewards,
     train_next_states, train_dones,
     train_ec, train_hr) = build_dataset(train_traj)

    (val_states, val_actions, val_rewards,
     val_next_states, val_dones,
     val_ec, val_hr) = build_dataset(val_traj)

    print("Training samples:", len(train_states))
    print("Validation samples:", len(val_states))

    # ---- Keep raw copies for noise-before-normalization ----
    train_states_raw      = train_states.clone()
    train_next_states_raw = train_next_states.clone()

    # ---- Normalize ----
    (train_states, train_next_states,
     val_states, val_next_states,
     state_mean, state_std) = normalize_states(
        train_states, train_next_states, val_states, val_next_states
    )

    # ---- Move to device ----
    def to_device(*tensors):
        return [t.to(device) for t in tensors]

    (train_states, train_actions, train_rewards,
     train_next_states, train_dones,
     train_ec, train_hr) = to_device(
        train_states, train_actions, train_rewards,
        train_next_states, train_dones,
        train_ec, train_hr
    )

    if args.train_noise_std > 0.0:
        train_states_raw, train_next_states_raw = to_device(
            train_states_raw, train_next_states_raw
        )

    state_mean_d = state_mean.to(device)
    state_std_d  = state_std.to(device)

    (val_states, val_actions, val_rewards,
     val_next_states, val_dones,
     val_ec, val_hr) = to_device(
        val_states, val_actions, val_rewards,
        val_next_states, val_dones,
        val_ec, val_hr
    )

    # ---- Networks ----
    q_net = QNetwork(hidden_dim=args.hidden_dim).to(device)
    target_net = QNetwork(hidden_dim=args.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    loss_fn = nn.MSELoss()

    # ---- Tracking ----
    train_losses = []
    val_losses = []
    train_type1_errors = []
    train_type2_errors = []
    val_type1_errors = []
    val_type2_errors = []

    # =========================
    # Training loop
    # =========================
    for epoch in tqdm(range(args.num_epochs), desc="Training"):
        q_net.train()
        perm = torch.randperm(len(train_states), device=device)
        batch_losses = []

        for i in range(0, len(train_states), args.batch_size):
            idx = perm[i:i + args.batch_size]

            a_batch      = train_actions[idx]
            r_batch      = train_rewards[idx]
            done_batch   = train_dones[idx]

            if args.train_noise_std > 0.0:
                # Multiplicative noise on raw states, then normalize — matches testing
                eps   = torch.randn_like(train_states_raw[idx])      * args.train_noise_std
                eps_n = torch.randn_like(train_next_states_raw[idx]) * args.train_noise_std
                s_batch      = (train_states_raw[idx]      * (1.0 + eps)   - state_mean_d) / state_std_d
                s_next_batch = (train_next_states_raw[idx] * (1.0 + eps_n) - state_mean_d) / state_std_d
            else:
                s_batch      = train_states[idx]
                s_next_batch = train_next_states[idx]

            q_values = q_net(s_batch)
            q_a = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = target_net(s_next_batch)
                q_next_max = q_next.max(dim=1)[0]
                target = r_batch + args.gamma * (1.0 - done_batch) * q_next_max

            loss = loss_fn(q_a, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        train_losses.append(train_loss)

        # ---- Validation loss ----
        q_net.eval()
        with torch.no_grad():
            q_values = q_net(val_states)
            q_a = q_values.gather(1, val_actions.unsqueeze(1)).squeeze(1)
            q_next = target_net(val_next_states)
            q_next_max = q_next.max(dim=1)[0]
            target = val_rewards + args.gamma * (1.0 - val_dones) * q_next_max
            val_loss = loss_fn(q_a, target).item()
            val_losses.append(val_loss)

        # ---- Update target network ----
        if (epoch + 1) % args.target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        # ---- Type-I / Type-II errors on train and val ----
        train_errors = evaluate_errors(q_net, train_traj, state_mean, state_std)
        val_errors = evaluate_errors(q_net, val_traj, state_mean, state_std)

        train_type1_errors.append(train_errors["type1_error"])
        train_type2_errors.append(train_errors["type2_error"])
        val_type1_errors.append(val_errors["type1_error"])
        val_type2_errors.append(val_errors["type2_error"])

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Train Type-I: {train_errors['type1_error']:.4f} "
            f"({train_errors['type1_count']}/{train_errors['type1_denom']}) | "
            f"Train Type-II: {train_errors['type2_error']:.4f} "
            f"({train_errors['type2_count']}/{train_errors['type2_denom']}) | "
            f"Val Type-I: {val_errors['type1_error']:.4f} "
            f"({val_errors['type1_count']}/{val_errors['type1_denom']}) | "
            f"Val Type-II: {val_errors['type2_error']:.4f} "
            f"({val_errors['type2_count']}/{val_errors['type2_denom']}) | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        scheduler.step()

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
            "scheduler_state_dict": scheduler.state_dict(),
            "train_noise_std": args.train_noise_std,
            "hyperparameters": vars(args),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to: {checkpoint_path}")

    # =========================
    # Plot losses and errors
    # =========================
    os.makedirs(args.loss_plot_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Offline Q-Learning Loss")
    ax1.legend()

    ax2.plot(train_type1_errors, label="Train Type-I (false trigger)", color="orange",  linestyle="--")
    ax2.plot(train_type2_errors, label="Train Type-II (missed trigger)", color="red",   linestyle="--")
    ax2.plot(val_type1_errors,   label="Val Type-I (false trigger)",    color="orange")
    ax2.plot(val_type2_errors,   label="Val Type-II (missed trigger)",  color="red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Error Rate")
    ax2.set_title("Type-I and Type-II Errors")
    ax2.set_ylim(0, 1)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close(fig)
    print(f"Saved loss plot to: {loss_plot_path}")



if __name__ == "__main__":
    main()