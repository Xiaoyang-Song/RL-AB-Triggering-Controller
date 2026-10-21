#!/bin/bash
#SBATCH --account=sunwbgt0
#SBATCH --job-name=RL-AB-RewardSweep-3
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=5:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/checkpoints/logs/reward_sweep_j3.log

# =========================
# Reward sweep job 3/5 — raise-C3-and-C2
# Goal: reduce BOTH Type-I and Type-II, not just trade one off for the other.
# Baseline (B1=B2=C1=C2=C3=5.0, from gforce_sweep_allfive/j1):
#   Type-I 12.96% (449/3464), Type-II 4.62% (67/1450)
#
# Reward-term roles (see assign_terminal_reward in data/transform_reward.py):
#   C3 - penalizes trigger + no collision      (the no-collision Type-I case)
#   C1 - penalizes trigger + low-risk collision (the other Type-I source)
#   C2 - penalizes no-trigger + high-risk miss   (drives Type-II)
#   B1 - rewards a correct high-risk trigger
#
# Hyperparameters held at the gforce_sweep_allfive winning config
# (lr=0.002, hidden_dim=256, target_update_freq=5) so any change in error
# rates is attributable to the reward weights, not the optimizer settings.
# =========================
B1=5.0
C1=5.0
B2=5.0
C2=8.0
C3=8.0
ETA=0.20

TRIGGER_PROB=0.005

LR=0.002
HIDDEN_DIM=256
TARGET_UPDATE_FREQ=5
LR_STEP_SIZE=100
LR_GAMMA=0.5
NUM_EPOCHS=1000
BATCH_SIZE=20000

TAG=j3
CKPT_DIR=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/checkpoints/model/reward_sweep/$TAG
PLOT_DIR=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/plots/reward_sweep/$TAG
RESULTS_DIR=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/results/reward_sweep/$TAG
mkdir -p "$CKPT_DIR" "$PLOT_DIR" "$RESULTS_DIR"

# =========================
# Pipeline — reward weights change the dataset's reward column, so the 5D
# dataset must be regenerated from scratch for each config.
# =========================
python /nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/data/transform_reward.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --trigger_prob $TRIGGER_PROB \
    --include_gforce

python /nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/training_v2.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce \
    --num_epochs $NUM_EPOCHS --lr $LR --hidden_dim $HIDDEN_DIM \
    --batch_size $BATCH_SIZE --target_update_freq $TARGET_UPDATE_FREQ \
    --lr_step_size $LR_STEP_SIZE --lr_gamma $LR_GAMMA \
    --checkpoint_dir "$CKPT_DIR" --loss_plot_dir "$PLOT_DIR"

# Final-epoch checkpoint
python /nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/testing_v2.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce \
    --checkpoint_dir "$CKPT_DIR" --output_dir "$RESULTS_DIR"

python /nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/visualize.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce \
    --results_dir "$RESULTS_DIR"

# Best checkpoint (lowest val Type-I+Type-II error)
python /nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/testing_v2.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce --use_best \
    --checkpoint_dir "$CKPT_DIR" --output_dir "$RESULTS_DIR"

python /nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/visualize.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce --use_best \
    --results_dir "$RESULTS_DIR"
