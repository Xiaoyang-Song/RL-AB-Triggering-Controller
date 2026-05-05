#!/bin/bash
#SBATCH --account=sunwbgt0
#SBATCH --job-name=RL-AB-J3
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=5:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/checkpoints/logs/sensitivity_j3.log


# =========================
# Shared reward parameters
# =========================
B1=5.0
C1=8.0
B2=5.0
C2=5.0
C3=5.0
ETA=0.2
 
# =========================
# Data generation
# =========================
TRIGGER_PROB=0.005
 
# =========================
# Training
# =========================
NUM_EPOCHS=500
LR=5e-3
HIDDEN_DIM=128
BATCH_SIZE=16384
 
# =========================
# Pipeline
# =========================
python ./data/transform_reward.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --trigger_prob $TRIGGER_PROB
 
python training_v2.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
    --num_epochs $NUM_EPOCHS --lr $LR --hidden_dim $HIDDEN_DIM --batch_size $BATCH_SIZE
 
python testing_v2.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA
 
python visualize.py \
    --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA