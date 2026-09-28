#!/bin/bash
#SBATCH --job-name=scenario_c_ablations
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=scripts_hpc/scenario_c_ablations_%j.log

echo "=========================================================="
echo "Starting Scenario C Slurm Job on $SLURMD_NODENAME at $(date)"
echo "=========================================================="

module load cuda/11.8 2>/dev/null || module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true

PYTHON_BIN="$HOME/.conda/envs/emit-har/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(which python3)"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd):$PYTHONPATH"

$PYTHON_BIN experiments/evaluate_clothing_pothole_ablation.py --config configs/ablation_clothing_potholes.yaml
$PYTHON_BIN evaluation/icra_benchmark_comparison.py

echo "Scenario C Slurm Job Completed at $(date)"
