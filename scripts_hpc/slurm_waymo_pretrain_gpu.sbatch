#!/bin/bash
#SBATCH --job-name=waymo_3d_pretrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=scripts_hpc/waymo_pretrain_%j.log

echo "=========================================================="
echo "Starting Slurm Job: $SLURM_JOB_ID on $SLURMD_NODENAME at $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================================="

module load cuda/11.8 2>/dev/null || module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true

PYTHON_BIN="$HOME/.conda/envs/emit-har/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(which python3)"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd):$PYTHONPATH"
mkdir -p checkpoints evaluation/results visualization/figures

$PYTHON_BIN experiments/train_waymo_pretrain.py --config configs/waymo_pretrain.yaml
