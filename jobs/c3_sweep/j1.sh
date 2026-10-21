#!/bin/bash
#SBATCH --account=sunwbgt0
#SBATCH --job-name=RL-AB-C3Sweep-1
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=5:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/checkpoints/logs/c3_sweep_j1.log

# =========================
# C3 sweep job 1/1 — C3=12.0
# Reduces Type-I error on no-collision trajectories specifically: C3 is the
# only reward term that penalizes "trigger + no collision" (see
# assign_terminal_reward in data/transform_reward.py). Baseline C3=5.0 is the
# already-completed gforce_sweep_allfive/j1 run (Type-I 12.96%, Type-II
# 4.62%) — compare against that.
#
# All other reward terms and hyperparameters are held at the winning config
# from jobs/gforce_sweep_allfive/ (lr=0.002, hidden_dim=256,
# target_update_freq=5), so any change in error rates here is attributable
# to C3 alone.
# =========================
B1=5.0
C1=5.0
B2=5.0
C2=5.0
C3=12.0
ETA=0.20

TRIGGER_PROB=0.005

LR=0.002
HIDDEN_DIM=256
TARGET_UPDATE_FREQ=5
LR_STEP_SIZE=100
LR_GAMMA=0.5
NUM_EPOCHS=1000
BATCH_SIZE=20000

TAG=j1
CKPT_DIR=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/checkpoints/model/c3_sweep/$TAG
PLOT_DIR=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/plots/c3_sweep/$TAG
RESULTS_DIR=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/results/c3_sweep/$TAG
mkdir -p "$CKPT_DIR" "$PLOT_DIR" "$RESULTS_DIR"

# =========================
# Pipeline — this C3 value changes the reward column, so the 5D dataset
# must be regenerated from scratch (unlike earlier sweeps, which reused an
# existing dataset via data/add_gforce.py).
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
