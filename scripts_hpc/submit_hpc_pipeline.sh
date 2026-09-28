#!/bin/bash
# ==============================================================================
# Master HPC Submission Script with Queue Selection & Dependency Chaining
# ==============================================================================
# Available Queues (MANIT / HPC):
#   - dgx / gpu   : Dedicated AI nodes (Fastest GPU, Recommended)
#   - gpu_shared  : Multi-user shared GPU queue
#   - short       : High Priority (Walltime limit <= 12h)
#   - workq       : Standard Default queue (Walltime <= 12h)
#   - medium      : Standard Priority (Walltime <= 24h)
#
# Usage:
#   ./scripts_hpc/submit_hpc_pipeline.sh [pbs|slurm|auto] [queue_name]
# Examples:
#   ./scripts_hpc/submit_hpc_pipeline.sh pbs dgx          # Dedicated DGX AI GPU (Recommended)
#   ./scripts_hpc/submit_hpc_pipeline.sh pbs gpu_shared   # Shared GPU queue
#   ./scripts_hpc/submit_hpc_pipeline.sh pbs short        # High-priority short queue
#   ./scripts_hpc/submit_hpc_pipeline.sh slurm gpu        # Slurm cluster
# ==============================================================================

MODE="${1:-auto}"
REQ_QUEUE="${2:-dgx}"

# Auto-detect scheduler
if [ "$MODE" = "auto" ]; then
    if command -v qsub &>/dev/null; then
        MODE="pbs"
    elif command -v sbatch &>/dev/null; then
        MODE="slurm"
    else
        echo "[Notice] Neither qsub nor sbatch detected locally. Defaulting to PBS script preparation."
        MODE="pbs"
    fi
fi

echo "=========================================================="
echo " Submitting 3D Metric Biomechanics VRU Pipeline to HPC"
echo " Scheduler Mode : $MODE"
echo " Target Queue   : $REQ_QUEUE"
echo "=========================================================="

Q_FLAG=""
if [ -n "$REQ_QUEUE" ]; then
    if [ "$MODE" = "pbs" ]; then
        Q_FLAG="-q $REQ_QUEUE"
    elif [ "$MODE" = "slurm" ]; then
        Q_FLAG="-p $REQ_QUEUE"
    fi
fi

if [ "$MODE" = "pbs" ]; then
    echo "Submitting Stage 1 (Waymo 3D Pre-training) to [$REQ_QUEUE]..."
    JOB1=$(qsub $Q_FLAG scripts_hpc/pbs_waymo_pretrain_gpu.pbs)
    echo "  -> Submitted Stage 1 Job ID: $JOB1"

    echo "Submitting Stage 2 (IDD-PeD Transfer) dependent on Stage 1..."
    JOB2=$(qsub $Q_FLAG -W depend=afterok:$JOB1 scripts_hpc/pbs_idd_ped_transfer_gpu.pbs)
    echo "  -> Submitted Stage 2 Job ID: $JOB2"

    echo "Submitting Stage 3 (Reality Gap Ablation) dependent on Stage 2..."
    JOB3=$(qsub $Q_FLAG -W depend=afterok:$JOB2 scripts_hpc/pbs_reality_gap_ablation.pbs)
    echo "  -> Submitted Stage 3 Job ID: $JOB3"

    echo "Submitting Stage 4 (Scenario C: Clothing & Pothole Ablation) dependent on Stage 3..."
    JOB4=$(qsub $Q_FLAG -W depend=afterok:$JOB3 scripts_hpc/pbs_clothing_potholes_ablation.pbs)
    echo "  -> Submitted Stage 4 Job ID: $JOB4"

    echo "=========================================================="
    echo " All 4 stages queued on PBS Pro / Torque in queue [$REQ_QUEUE]!"
    echo " Check status with: qstat -u $USER"
    echo "=========================================================="

elif [ "$MODE" = "slurm" ]; then
    echo "Submitting Stage 1 (Waymo 3D Pre-training) to [$REQ_QUEUE]..."
    JOB1=$(sbatch $Q_FLAG --parsable scripts_hpc/slurm_waymo_pretrain_gpu.sbatch)
    echo "  -> Submitted Stage 1 Job ID: $JOB1"

    echo "Submitting Stage 2 (IDD-PeD Transfer) dependent on Stage 1..."
    JOB2=$(sbatch $Q_FLAG --parsable --dependency=afterok:$JOB1 scripts_hpc/slurm_idd_ped_transfer_gpu.sbatch)
    echo "  -> Submitted Stage 2 Job ID: $JOB2"

    echo "Submitting Stage 3 (Reality Gap Ablation) dependent on Stage 2..."
    JOB3=$(sbatch $Q_FLAG --parsable --dependency=afterok:$JOB2 scripts_hpc/slurm_reality_gap_ablation.sbatch)
    echo "  -> Submitted Stage 3 Job ID: $JOB3"

    echo "Submitting Stage 4 (Scenario C: Clothing & Pothole Ablation) dependent on Stage 3..."
    JOB4=$(sbatch $Q_FLAG --parsable --dependency=afterok:$JOB3 scripts_hpc/slurm_clothing_potholes_ablation.sbatch)
    echo "  -> Submitted Stage 4 Job ID: $JOB4"

    echo "=========================================================="
    echo " All 4 stages queued on Slurm in partition [$REQ_QUEUE]!"
    echo " Check status with: squeue -u $USER"
    echo "=========================================================="
fi
