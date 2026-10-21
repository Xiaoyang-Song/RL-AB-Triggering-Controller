#!/bin/bash
#SBATCH --account=sunwbgt0
#SBATCH --job-name=RL-AB-GForce-AllFive
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=5:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/checkpoints/logs/gforce_j1.log


# =========================
# Reward parameters: all five set to 5.0 (uniform cost/reward, no asymmetry)
# =========================
B1=5.0
C1=5.0
B2=5.0
C2=5.0
C3=5.0
ETA=0.20

# =========================
# Training hyperparameters
# Same baseline values used in jobs/gforce/j0.sh. The jobs/gforce_sweep/
# sweep (run for the b1=5/c1=6 config) hadn't settled on a winner yet when
# this job was written — once summarize_gforce_sweep.py names a clear best
# config, swap these four values in here too.
# =========================
NUM_EPOCHS=1000
LR=5e-3
HIDDEN_DIM=256
BATCH_SIZE=20000
TARGET_UPDATE_FREQ=5
LR_STEP_SIZE=100
LR_GAMMA=0.5

# =========================
# Pipeline
#
# No data/transform_reward.py step here: the 5D dataset
# (data/aggregated/rl_trajectories_b1${B1}_c1${C1}_b2${B2}_c2${C2}_c3${C3}_eta${ETA}_gforce.pkl)
# was already built from the existing 4D dataset via data/add_gforce.py,
# which just attaches each trajectory's constant gforce value without
# rerunning injury-risk simulation.
#
# training_v2.py now saves TWO checkpoints: the final-epoch one, and
# "_best" (lowest val Type-I+Type-II error, not lowest MSE loss — see
# jobs/gforce_sweep/ for why loss-based selection picks an undertrained,
# trigger-happy network). Both get evaluated below for comparison.
# =========================
python training_v2.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce \
    --num_epochs $NUM_EPOCHS --lr $LR --hidden_dim $HIDDEN_DIM \
    --batch_size $BATCH_SIZE --target_update_freq $TARGET_UPDATE_FREQ \
    --lr_step_size $LR_STEP_SIZE --lr_gamma $LR_GAMMA

# Final-epoch checkpoint
python testing_v2.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce

python visualize.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce

# Best checkpoint (lowest val Type-I+Type-II error)
python testing_v2.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce --use_best

python visualize.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce --use_best
