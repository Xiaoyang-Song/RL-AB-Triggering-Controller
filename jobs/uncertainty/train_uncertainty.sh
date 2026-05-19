#!/bin/bash
#SBATCH --account=sunwbgt0
#SBATCH --job-name=RL-AB-UC-TRAIN
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=48:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/checkpoints/logs/train_uncertainty_j0.log



# =========================
# Model parameters
# =========================
B1=5.0
C1=6.0
B2=5.0
C2=5.0
C3=5.0
ETA=0.2

# =========================
# Training hyperparameters
# =========================
NUM_EPOCHS=1000
LR=5e-3
HIDDEN_DIM=256
BATCH_SIZE=20000

# =========================
# Uncertainty sweep
# =========================
TRAIN_NOISES=(0.01 0.05 0.1)
NUM_REPS=20
BASE_OUTPUT_DIR="results/train_uncertainty"

echo "Train-with-noise sweep: noises=${TRAIN_NOISES[*]}  reps=${NUM_REPS}  output=${BASE_OUTPUT_DIR}"
echo "-------------------------------------------------------------"

for NOISE in "${TRAIN_NOISES[@]}"; do
    NOISE_DIR="${BASE_OUTPUT_DIR}/noise${NOISE}"
    echo ""
    echo ">>> Training with noise std: ${NOISE}"

    python training_v2.py \
        --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
        --num_epochs $NUM_EPOCHS --lr $LR --hidden_dim $HIDDEN_DIM --batch_size $BATCH_SIZE \
        --train_noise_std ${NOISE}

    echo ""
    echo "    Training done. Running ${NUM_REPS} evaluation trials..."

    for i in $(seq 1 ${NUM_REPS}); do
        echo "  [noise=${NOISE}  rep=${i}/${NUM_REPS}]  seed=${i}"

        python testing_v2.py \
            --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
            --train_noise_std ${NOISE} \
            --uncertainty_analysis \
            --magnitude_variation ${NOISE} \
            --output_dir ${NOISE_DIR} \
            --seed ${i} \
            --label "rep${i}"
    done

    echo ""
    echo "  --- Summary for noise ${NOISE} ---"
    python summarize_uncertainty.py \
        --results_dir ${NOISE_DIR} \
        --pattern "rep" \
        --eta $ETA

done

echo ""
echo "-------------------------------------------------------------"
echo "All done. Results under: ${BASE_OUTPUT_DIR}/"
