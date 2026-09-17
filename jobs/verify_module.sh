#!/bin/bash
#SBATCH --account=sunwbgt0
#SBATCH --job-name=RL-AB-Verify
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=2:00:00
#SBATCH --output=/nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller/checkpoints/logs/verify_module.log

cd /nfs/turbo/coe-sunwbgt/xysong/RL-AB-Triggering-Controller
python3 verify_module.py
