import os
import sys
import random
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Redirected stdout (e.g. SLURM --output) is fully buffered by default, so
# epoch-level prints wouldn't show up until the buffer fills or the run ends.
# Line-buffer it so `tail -f` on the log actually shows live progress.
sys.stdout.reconfigure(line_buffering=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    # On shared HPC login nodes with many cores, torch's default of using
    # every visible core for tiny per-trajectory ops causes severe thread
    # contention. Cap it — this only affects CPU fallback runs.
    torch.set_num_threads(min(8, os.cpu_count() or 8))
print("Device:", device)


# =========================
# Argparse
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline Monte-Carlo policy gradient (REINFORCE + baseline) for airbag triggering."
    )

    # Data — same aggregated trajectories used by the Q-learning pipeline
    parser.add_argument("--data_dir", type=str, default="data/aggregated")
    parser.add_argument("--checkpoint_dir", type=str, default="policy_gradient/checkpoints/model")
    parser.add_argument("--loss_plot_dir", type=str, default="policy_gradient/plots")

    # Reward / eta parameters (used only for resolving filenames).
    # Defaults to the "all 5.0" reward configuration (b1=c1=b2=c2=c3=5.0, eta=0.15).
    parser.add_argument("--b1", type=float, default=5.0)
    parser.add_argument("--c1", type=float, default=5.0)
    parser.add_argument("--b2", type=float, default=5.0)
    parser.add_argument("--c2", type=float, default=5.0)
    parser.add_argument("--c3", type=float, default=5.0)
    parser.add_argument("--eta", type=float, default=0.15)

    # Training hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr_step_size", type=int, default=100)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Entropy bonus weight — keeps the policy from collapsing "
                             "given how rare the trigger action is in the offline data.")
    parser.add_argument("--normalize_advantage", action="store_true", default=True)
    parser.add_argument("--advantage_clip", type=float, default=10.0,
                        help="Clamp normalized advantage to [-x, x]. With trigger actions "
                             "at ~0.4%% of transitions, a minibatch's return distribution is "
                             "mostly near-zero — normalizing by that tiny std can blow up the "
                             "handful of large-return outliers into extreme z-scores, which "
                             "runs away the actor loss. Set <=0 to disable.")
    parser.add_argument("--grad_clip_norm", type=float, default=5.0,
                        help="Max gradient norm for actor and critic updates. Set <=0 to disable.")

    return parser.parse_args()


def build_param_suffix(args):
    return f"b1{args.b1}_c1{args.c1}_b2{args.b2}_c2{args.c2}_c3{args.c3}_eta{args.eta}"


def build_paths(args):
    suffix = build_param_suffix(args)
    data_path = os.path.join(args.data_dir, f"rl_trajectories_{suffix}.pkl")
    checkpoint_path = os.path.join(args.checkpoint_dir, f"pg_net_{suffix}.pth")
    loss_plot_path = os.path.join(args.loss_plot_dir, f"PG_loss_{suffix}.png")
    return data_path, checkpoint_path, loss_plot_path


# =========================
# Networks
# =========================
class PolicyNetwork(nn.Module):
    """Outputs action logits over {wait, trigger}."""

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
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


class ValueNetwork(nn.Module):
    """State-value baseline V(s), used to form the REINFORCE advantage."""

    def __init__(self, state_dim=4, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =========================
# Utilities
# =========================
def set_seed(seed_value: int):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def compute_returns_to_go(rewards, gamma):
    """Discounted Monte-Carlo return-to-go for one trajectory."""
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def build_dataset(trajectories, gamma):
    """
    Convert list of trajectories into (state, action, return_to_go,
    eventual_collision, high_risk) tuples. Rewards in this dataset are
    zero everywhere except the terminal step (first trigger, or collision),
    so return_to_go collapses to a discounted terminal reward — this is
    exact Monte-Carlo credit assignment, no bootstrapping needed.
    """
    states_list = []
    actions_list = []
    returns_list = []
    eventual_collision_list = []
    high_risk_list = []

    for traj in trajectories:
        if len(traj) == 0:
            continue

        states = traj[["walker_vel_ms", "ego_vel_ms", "dx", "dy"]].values.astype(np.float32)
        actions = traj["action_trigger"].values.astype(np.int64)
        rewards = traj["reward"].values.astype(np.float32)
        returns = compute_returns_to_go(rewards, gamma)
        T = len(traj)

        ec = int(traj["eventual_collision"].iloc[-1])
        hr = int(traj["high_risk"].iloc[-1])

        for t in range(T):
            states_list.append(states[t])
            actions_list.append(actions[t])
            returns_list.append(returns[t])
            eventual_collision_list.append(ec)
            high_risk_list.append(hr)

    states = torch.tensor(np.array(states_list), dtype=torch.float32)
    actions = torch.tensor(np.array(actions_list), dtype=torch.int64)
    returns = torch.tensor(np.array(returns_list), dtype=torch.float32)
    eventual_collision = torch.tensor(eventual_collision_list, dtype=torch.int64)
    high_risk = torch.tensor(high_risk_list, dtype=torch.int64)

    return states, actions, returns, eventual_collision, high_risk


def normalize_states(train_states, val_states):
    """Standardize states using train-set statistics only."""
    mean = train_states.mean(dim=0, keepdim=True)
    std = train_states.std(dim=0, keepdim=True).clamp_min(1e-6)

    train_states = (train_states - mean) / std
    val_states = (val_states - mean) / std

    return train_states, val_states, mean, std


def evaluate_errors(policy_net, trajectories, state_mean, state_std):
    """
    Same definition used by the Q-learning pipeline, so results are
    directly comparable:

    Type-I  (false positive): model triggers when it shouldn't
             — low injury risk collision, OR no collision at all
    Type-II (false negative): model never triggers when it should
             — high injury risk collision with no trigger fired

    Decision uses the greedy (argmax-probability) action, mirroring
    deployment where the airbag fires on the first trigger action.
    """
    policy_net.eval()

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

            logits = policy_net(states)
            triggered = bool((torch.argmax(logits, dim=1) == 1).any().item())

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
    train_states, train_actions, train_returns, train_ec, train_hr = build_dataset(train_traj, args.gamma)
    val_states, val_actions, val_returns, val_ec, val_hr = build_dataset(val_traj, args.gamma)

    print("Training samples:", len(train_states))
    print("Validation samples:", len(val_states))

    # ---- Normalize ----
    train_states, val_states, state_mean, state_std = normalize_states(train_states, val_states)

    # ---- Move to device ----
    def to_device(*tensors):
        return [t.to(device) for t in tensors]

    train_states, train_actions, train_returns = to_device(train_states, train_actions, train_returns)
    val_states, val_actions, val_returns = to_device(val_states, val_actions, val_returns)

    # ---- Networks ----
    policy_net = PolicyNetwork(hidden_dim=args.hidden_dim).to(device)
    value_net = ValueNetwork(hidden_dim=args.hidden_dim).to(device)

    actor_optimizer = optim.Adam(policy_net.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(value_net.parameters(), lr=args.critic_lr)
    actor_scheduler = optim.lr_scheduler.StepLR(actor_optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    critic_scheduler = optim.lr_scheduler.StepLR(critic_optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    value_loss_fn = nn.MSELoss()

    # ---- Tracking ----
    train_policy_losses, train_value_losses = [], []
    val_value_losses = []
    train_type1_errors, train_type2_errors = [], []
    val_type1_errors, val_type2_errors = [], []

    # =========================
    # Training loop
    # =========================
    for epoch in tqdm(range(args.num_epochs), desc="Training"):
        policy_net.train()
        value_net.train()
        perm = torch.randperm(len(train_states), device=device)
        policy_losses, value_losses = [], []

        for i in range(0, len(train_states), args.batch_size):
            idx = perm[i:i + args.batch_size]

            s_batch = train_states[idx]
            a_batch = train_actions[idx]
            g_batch = train_returns[idx]

            # ---- Critic: regress V(s) toward the Monte-Carlo return ----
            values = value_net(s_batch)
            value_loss = value_loss_fn(values, g_batch)

            critic_optimizer.zero_grad()
            value_loss.backward()
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(value_net.parameters(), args.grad_clip_norm)
            critic_optimizer.step()

            # ---- Actor: REINFORCE with baseline ----
            with torch.no_grad():
                baseline = value_net(s_batch)
                advantage = g_batch - baseline
                if args.normalize_advantage and advantage.numel() > 1:
                    advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
                if args.advantage_clip > 0:
                    advantage = advantage.clamp(-args.advantage_clip, args.advantage_clip)

            logits = policy_net(s_batch)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(a_batch)

            policy_loss = -(log_probs * advantage).mean() - args.entropy_coef * dist.entropy().mean()

            actor_optimizer.zero_grad()
            policy_loss.backward()
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(policy_net.parameters(), args.grad_clip_norm)
            actor_optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        train_policy_loss = float(np.mean(policy_losses))
        train_value_loss = float(np.mean(value_losses))
        train_policy_losses.append(train_policy_loss)
        train_value_losses.append(train_value_loss)

        # ---- Validation critic loss (policy loss needs on-policy sampling, so we
        #      only track value-fit quality + downstream Type-I/II error on val) ----
        policy_net.eval()
        value_net.eval()
        with torch.no_grad():
            val_values = value_net(val_states)
            val_value_loss = value_loss_fn(val_values, val_returns).item()
            val_value_losses.append(val_value_loss)

        # ---- Type-I / Type-II errors on train and val ----
        train_errors = evaluate_errors(policy_net, train_traj, state_mean, state_std)
        val_errors = evaluate_errors(policy_net, val_traj, state_mean, state_std)

        train_type1_errors.append(train_errors["type1_error"])
        train_type2_errors.append(train_errors["type2_error"])
        val_type1_errors.append(val_errors["type1_error"])
        val_type2_errors.append(val_errors["type2_error"])

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"Policy Loss: {train_policy_loss:.6f} | "
            f"Value Loss: {train_value_loss:.6f} | "
            f"Val Value Loss: {val_value_loss:.6f} | "
            f"Train Type-I: {train_errors['type1_error']:.4f} "
            f"({train_errors['type1_count']}/{train_errors['type1_denom']}) | "
            f"Train Type-II: {train_errors['type2_error']:.4f} "
            f"({train_errors['type2_count']}/{train_errors['type2_denom']}) | "
            f"Val Type-I: {val_errors['type1_error']:.4f} "
            f"({val_errors['type1_count']}/{val_errors['type1_denom']}) | "
            f"Val Type-II: {val_errors['type2_error']:.4f} "
            f"({val_errors['type2_count']}/{val_errors['type2_denom']}) | "
            f"Actor LR: {actor_scheduler.get_last_lr()[0]:.6f}"
        )

        actor_scheduler.step()
        critic_scheduler.step()

    # =========================
    # Save model + normalization stats
    # =========================
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy_net.state_dict(),
            "value_state_dict": value_net.state_dict(),
            "state_mean": state_mean.cpu(),
            "state_std": state_std.cpu(),
            "actor_scheduler_state_dict": actor_scheduler.state_dict(),
            "critic_scheduler_state_dict": critic_scheduler.state_dict(),
            "hyperparameters": vars(args),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to: {checkpoint_path}")

    # =========================
    # Plot losses and errors
    # =========================
    os.makedirs(args.loss_plot_dir, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11))

    ax1.plot(train_policy_losses, label="Train Policy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("REINFORCE Loss")
    ax1.set_title("Offline Policy Gradient — Actor Loss")
    ax1.legend()

    ax2.plot(train_value_losses, label="Train Value Loss")
    ax2.plot(val_value_losses, label="Val Value Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Baseline Critic Loss")
    ax2.legend()

    ax3.plot(train_type1_errors, label="Train Type-I (false trigger)", color="orange", linestyle="--")
    ax3.plot(train_type2_errors, label="Train Type-II (missed trigger)", color="red", linestyle="--")
    ax3.plot(val_type1_errors, label="Val Type-I (false trigger)", color="orange")
    ax3.plot(val_type2_errors, label="Val Type-II (missed trigger)", color="red")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Error Rate")
    ax3.set_title("Type-I and Type-II Errors")
    ax3.set_ylim(0, 1)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close(fig)
    print(f"Saved loss plot to: {loss_plot_path}")


if __name__ == "__main__":
    main()
