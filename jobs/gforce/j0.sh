#!/bin/bash
#SBATCH --account=sunwbgt0
#SBATCH --job-name=RL-AB-GForce
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=5:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/checkpoints/logs/gforce_j0.log


# =========================
# Shared reward parameters
# (same config as jobs/sensitivity/j1.sh, now with gforce as a 5th state dim)
# =========================
B1=5.0
C1=6.0
B2=5.0
C2=5.0
C3=5.0
ETA=0.20

# =========================
# Training
# =========================
NUM_EPOCHS=1000
LR=5e-3
HIDDEN_DIM=256
BATCH_SIZE=20000

# =========================
# Pipeline
#
# No data/transform_reward.py step here: the 5D dataset
# (data/aggregated/rl_trajectories_b1${B1}_c1${C1}_b2${B2}_c2${C2}_c3${C3}_eta${ETA}_gforce.pkl)
# was already built from the existing 4D dataset via data/add_gforce.py,
# which just attaches each trajectory's constant gforce value without
# rerunning injury-risk simulation.
# =========================
python training_v2.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce \
    --num_epochs $NUM_EPOCHS --lr $LR --hidden_dim $HIDDEN_DIM --batch_size $BATCH_SIZE

python testing_v2.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce

python visualize.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --include_gforce
