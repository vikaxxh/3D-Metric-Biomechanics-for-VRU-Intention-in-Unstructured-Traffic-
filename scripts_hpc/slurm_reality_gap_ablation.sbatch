#!/bin/bash
#SBATCH --job-name=reality_gap_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=scripts_hpc/reality_gap_eval_%j.log

echo "=========================================================="
echo "Starting Evaluation Job: $SLURM_JOB_ID on $SLURMD_NODENAME at $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================================="

module load cuda/11.8 2>/dev/null || module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true

PYTHON_BIN="$HOME/.conda/envs/emit-har/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(which python3)"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd):$PYTHONPATH"
mkdir -p evaluation/results visualization/figures

$PYTHON_BIN experiments/evaluate_reality_gap.py --config configs/ablation_lane_dropout.yaml
$PYTHON_BIN visualization/plot_reality_gap_curves.py
$PYTHON_BIN visualization/plot_biomech_skeletons.py

echo "Evaluation and Figure Generation Finished at $(date)"
