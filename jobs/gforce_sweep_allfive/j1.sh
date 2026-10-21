#!/bin/bash
#SBATCH --account=sunwbgt0
#SBATCH --job-name=RL-AB-GFSweep5-1
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=3:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/checkpoints/logs/gforce_sweep_allfive_j1.log

# =========================
# Hyperparameter sweep job 1/9 — mid-lr_baseline-arch
# Same sweep grid as jobs/gforce_sweep/ (b1=5/c1=6 config), run here against
# the all-5.0 reward config instead (b1=b2=c1=c2=c3=5.0). Uses the fixed
# training_v2.py, which now selects "_best" by lowest val Type-I+Type-II
# error rather than lowest val MSE loss.
# =========================
B1=5.0
C1=5.0
B2=5.0
C2=5.0
C3=5.0
ETA=0.20

LR=0.002
HIDDEN_DIM=256
TARGET_UPDATE_FREQ=5
LR_STEP_SIZE=100
LR_GAMMA=0.5
NUM_EPOCHS=1000
BATCH_SIZE=20000

# Each sweep job gets its own checkpoint/plot/results subfolder so the
# 10 runs never overwrite each other or the existing gforce checkpoint.
TAG=j1
CKPT_DIR=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/checkpoints/model/sweep_allfive/$TAG
PLOT_DIR=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/plots/sweep_allfive/$TAG
RESULTS_DIR=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/results/sweep_allfive/$TAG
mkdir -p "$CKPT_DIR" "$PLOT_DIR" "$RESULTS_DIR"

# =========================
# Pipeline
# Uses the existing gforce-augmented dataset
# (data/aggregated/rl_trajectories_b1${B1}_c1${C1}_b2${B2}_c2${C2}_c3${C3}_eta${ETA}_gforce.pkl)
# built via data/add_gforce.py — no need to rerun data/transform_reward.py.
# =========================
python /nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/training_v2.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce \
    --num_epochs $NUM_EPOCHS --lr $LR --hidden_dim $HIDDEN_DIM \
    --batch_size $BATCH_SIZE --target_update_freq $TARGET_UPDATE_FREQ \
    --lr_step_size $LR_STEP_SIZE --lr_gamma $LR_GAMMA \
    --checkpoint_dir "$CKPT_DIR" --loss_plot_dir "$PLOT_DIR"

# Evaluate the best-val-Type-I/II checkpoint (not the final epoch).
python /nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/testing_v2.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce --use_best \
    --checkpoint_dir "$CKPT_DIR" --output_dir "$RESULTS_DIR"

python /nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/visualize.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce --use_best \
    --results_dir "$RESULTS_DIR"
