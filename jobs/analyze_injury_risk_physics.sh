#!/bin/bash
#SBATCH --account=jhjin1
#SBATCH --job-name=RL-AB-InjuryRiskPhysics
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=1:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/checkpoints/logs/analyze_injury_risk_physics.log

cd /nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller

python analyze_injury_risk_physics.py \
    --results_csv results/sweep_allfive/j1/evaluation_results_b15.0_c15.0_b25.0_c25.0_c35.0_eta0.2_gforce_best.csv \
    --root_folder data/simulation/ \
    --eta 0.2 \
    --title "gforce sweep_allfive j1 (lr=0.002, hidden_dim=256) -- physics-estimated risk" \
    --out plots/sweep_allfive/j1/injury_risk_physics_curve.png
