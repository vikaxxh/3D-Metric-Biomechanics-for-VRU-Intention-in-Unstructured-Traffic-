"""
Master Benchmark Runner
=======================
Executes the full end-to-end research pipeline for:
"Bridging the Reality Gap: 3D Metric Biomechanics for VRU Intention in Unstructured Traffic"
Target Venues: IEEE IROS 2027 / IEEE ICRA 2027 / IEEE VTC
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.synthetic_benchmark_gen import generate_benchmark_dataset
from experiments.train_waymo_pretrain import train_waymo_stage1
from experiments.train_idd_transfer import train_idd_stage2
from experiments.evaluate_reality_gap import run_reality_gap_evaluation
from visualization.plot_reality_gap_curves import plot_reality_gap_curves
from visualization.plot_biomech_skeletons import plot_3d_skeleton_comparison


def main():
    start_time = time.time()
    print("=" * 80)
    print("  BRIDGING THE REALITY GAP: 3D METRIC BIOMECHANICS FOR VRU INTENTION")
    print("                 IN UNSTRUCTURED TRAFFIC (WAYMO + IDD-PeD)")
    print("=" * 80)

    # 1. Dataset Verification
    data_dir = "data/benchmark_data"
    if not os.path.exists(os.path.join(data_dir, "waymo", "train")):
        print("\n[Step 1] Synthesizing physically consistent Waymo 3D and IDD-PeD datasets...")
        generate_benchmark_dataset(output_root=data_dir, num_train=120, num_val=30, num_test=50)
    else:
        print("\n[Step 1] Verified dataset existence in data/benchmark_data.")

    # 2. Stage 1: Waymo 3D Pre-training
    print("\n[Step 2] Executing Stage 1: Waymo 3D Metric Biomechanics Pre-training...")
    train_waymo_stage1("configs/waymo_pretrain.yaml")

    # 3. Stage 2: IDD-PeD Cross-Domain Transfer
    print("\n[Step 3] Executing Stage 2: Cross-Domain Reality Gap Transfer to IDD-PeD...")
    train_idd_stage2("configs/idd_ped_transfer.yaml")

    # 4. Stage 3: Reality Gap Ablation Evaluation
    print("\n[Step 4] Executing Stage 3: Controlled Infrastructure Dropout Ablation...")
    results = run_reality_gap_evaluation("configs/ablation_lane_dropout.yaml")

    # 5. Visualizations & Paper Figures
    print("\n[Step 5] Generating Publication-Grade Figures (PNG & PDF)...")
    plot_reality_gap_curves()
    plot_3d_skeleton_comparison()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"  ALL BENCHMARKS AND EXPERIMENTS COMPLETED SUCCESSFULLY IN {elapsed:.1f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
