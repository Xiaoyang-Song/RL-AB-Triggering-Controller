#!/bin/bash
#SBATCH --account=sunwbgt0
#SBATCH --job-name=RL-AB-UC
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=48:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/checkpoints/logs/uncertainty_j0.log



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
# Uncertainty analysis
# =========================
MAGNITUDES=(0.01 0.05 0.1 0.15 0.2)
NUM_REPS=20
BASE_OUTPUT_DIR="results/uncertainty"

echo "Uncertainty sweep: magnitudes=${MAGNITUDES[*]}  reps=${NUM_REPS}  base=${BASE_OUTPUT_DIR}"
echo "-------------------------------------------------------------"

for MAG in "${MAGNITUDES[@]}"; do
    MAG_DIR="${BASE_OUTPUT_DIR}/mag${MAG}"
    echo ""
    echo ">>> Magnitude: ${MAG}  ->  ${MAG_DIR}"

    for i in $(seq 1 ${NUM_REPS}); do
        LABEL="rep${i}"
        echo "  [mag=${MAG}  rep=${i}/${NUM_REPS}]  seed=${i}"

        python testing_v2.py \
            --b1 $B1 --c1 $C1 --b2 $B2 --c2 $C2 --c3 $C3 --eta $ETA \
            --uncertainty_analysis \
            --magnitude_variation ${MAG} \
            --output_dir ${MAG_DIR} \
            --seed ${i} \
            --label "${LABEL}"
    done

    echo ""
    echo "  --- Summary for magnitude ${MAG} ---"
    python summarize_uncertainty.py \
        --results_dir ${MAG_DIR} \
        --pattern "rep" \
        --eta $ETA

done

echo ""
echo "-------------------------------------------------------------"
echo "All done. Results under: ${BASE_OUTPUT_DIR}/"
