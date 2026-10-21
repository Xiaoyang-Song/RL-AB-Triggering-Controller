#!/bin/bash
#SBATCH --account=sunwbgt0
#SBATCH --job-name=RL-AB-GFSweep-9
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=3:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/checkpoints/logs/gforce_sweep_j9.log

# =========================
# Hyperparameter sweep job 9/9 — mid-lr_gentler-decay
# Diagnosis: with gforce as a 5th state dim, the baseline LR=5e-3 /
# hidden_dim=256 / target_update_freq=5 combo (tuned for the 4D state)
# caused a mid-training instability spike in val loss that never fully
# recovered by epoch 1000 (see plots/Q_loss_..._eta0.2_gforce.png).
# This sweep varies LR, network width, target-network sync frequency,
# and LR decay schedule to find a combo that trains as cleanly as the
# 4D baseline (plots/Q_loss_..._eta0.2.png).
# =========================
B1=5.0
C1=6.0
B2=5.0
C2=5.0
C3=5.0
ETA=0.20

LR=0.002
HIDDEN_DIM=256
TARGET_UPDATE_FREQ=5
LR_STEP_SIZE=200
LR_GAMMA=0.7
NUM_EPOCHS=1000
BATCH_SIZE=20000

# Each sweep job gets its own checkpoint/plot/results subfolder so the
# 10 runs never overwrite each other or the existing gforce checkpoint.
TAG=j9
CKPT_DIR=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/checkpoints/model/sweep/$TAG
PLOT_DIR=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/plots/sweep/$TAG
RESULTS_DIR=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/results/sweep/$TAG
mkdir -p "$CKPT_DIR" "$PLOT_DIR" "$RESULTS_DIR"

# =========================
# Pipeline
# Uses the existing gforce-augmented dataset (data/add_gforce.py output) —
# no need to rerun data/transform_reward.py.
# =========================
python /nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/training_v2.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce \
    --num_epochs $NUM_EPOCHS --lr $LR --hidden_dim $HIDDEN_DIM \
    --batch_size $BATCH_SIZE --target_update_freq $TARGET_UPDATE_FREQ \
    --lr_step_size $LR_STEP_SIZE --lr_gamma $LR_GAMMA \
    --checkpoint_dir "$CKPT_DIR" --loss_plot_dir "$PLOT_DIR"

# Evaluate the best-val-loss checkpoint (not the final epoch) — this is
# the whole point of the sweep, since the final epoch is exactly what
# was unreliable in the first gforce run.
python /nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/testing_v2.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce --use_best \
    --checkpoint_dir "$CKPT_DIR" --output_dir "$RESULTS_DIR"

python /nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/visualize.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce --use_best \
    --results_dir "$RESULTS_DIR"
