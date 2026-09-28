"""
Publication Figure Generator: Reality Gap, Stress-Tests & Early Horizon Curves
=============================================================================
Strictly requires authentic JSON evaluation artifacts:
- evaluation/results/reality_gap_ablation.json
- evaluation/results/clothing_pothole_ablation.json

Zero hardcoded fallback lists or simulated curve generation.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import matplotlib.pyplot as plt
import numpy as np


def apply_ieee_style():
    """Configures matplotlib with IEEE Transactions publication styling."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 13,
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_reality_gap_curves(
    results_json: str = "evaluation/results/reality_gap_ablation.json",
    output_dir: str = "visualization/figures"
):
    """Figure 1: Authentic performance degradation across lane dropout levels."""
    if not os.path.exists(results_json):
        print(f"[Notice] Skipping Figure 1: {results_json} not found. Run experiments/evaluate_reality_gap.py first.")
        return

    with open(results_json, "r") as f:
        data = json.load(f)

    lane_pct = [int(x * 100) for x in data["lane_levels"]]
    models = data["models"]

    apply_ieee_style()
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    color_vis = "#d95f02"   # Vermilion/Orange
    color_pose = "#7570b3"  # Slate Purple
    color_prop = "#1b9e77"  # Emerald Teal

    # 1. Macro F1-Score Plot
    ax1 = axes[0]
    ax1.plot(lane_pct, models["Visual_Context_Only"]["f1"], "o--", color=color_vis, label="Visual Context Only")
    ax1.plot(lane_pct, models["2D_Pose_Baseline"]["f1"], "^--", color=color_pose, label="2D Pose (MMPose)")
    ax1.plot(lane_pct, models["BioMetric_VRU_Proposed"]["f1"], "s-", color=color_prop, label="BioMetric-VRU (Ours)")
    ax1.set_xlabel("Road Infrastructure Availability (%)\n(Structured Western $\\rightarrow$ Unstructured India)")
    ax1.set_ylabel("Intention Macro $F_1$ Score (%)")
    ax1.set_title("(a) Intention Prediction $F_1$ Resilience")
    ax1.set_xlim(105, -5)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", framealpha=0.9)

    # 2. Accuracy Plot
    ax2 = axes[1]
    ax2.plot(lane_pct, models["Visual_Context_Only"]["acc"], "o--", color=color_vis, label="Visual Context Only")
    ax2.plot(lane_pct, models["2D_Pose_Baseline"]["acc"], "^--", color=color_pose, label="2D Pose")
    ax2.plot(lane_pct, models["BioMetric_VRU_Proposed"]["acc"], "s-", color=color_prop, label="BioMetric-VRU (Ours)")
    ax2.set_xlabel("Road Infrastructure Availability (%)\n(Structured Western $\\rightarrow$ Unstructured India)")
    ax2.set_ylabel("Classification Accuracy (%)")
    ax2.set_title("(b) Intention Classification Accuracy")
    ax2.set_xlim(105, -5)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. AUC-ROC Plot
    ax3 = axes[2]
    ax3.plot(lane_pct, models["Visual_Context_Only"]["auc"], "o--", color=color_vis, label="Visual Context Only")
    ax3.plot(lane_pct, models["2D_Pose_Baseline"]["auc"], "^--", color=color_pose, label="2D Pose")
    ax3.plot(lane_pct, models["BioMetric_VRU_Proposed"]["auc"], "s-", color=color_prop, label="BioMetric-VRU (Ours)")
    ax3.set_xlabel("Road Infrastructure Availability (%)\n(Structured Western $\\rightarrow$ Unstructured India)")
    ax3.set_ylabel("AUC-ROC Score")
    ax3.set_title("(c) Discriminative Separation (AUC)")
    ax3.set_xlim(105, -5)
    ax3.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    out_png = os.path.join(output_dir, "fig_reality_gap_ablation.png")
    out_pdf = os.path.join(output_dir, "fig_reality_gap_ablation.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"[Figure 1 Saved] Reality Gap Curves:\n  - {out_png}\n  - {out_pdf}")


def plot_clothing_and_pothole_ablation(
    results_json: str = "evaluation/results/clothing_pothole_ablation.json",
    output_dir: str = "visualization/figures"
):
    """
    Figure 2: Stress-Test Protocol validation from authentic model evaluation.
    Panel (a): F1 vs. Lower-Limb Occlusion (Gated vs. Ungated).
    Panel (b): Learned α_upper Weight vs. Occlusion Level.
    Panel (c): F1 vs. Road Roughness Pitch Shocks (Intrinsic Torso vs. Camera Frame).
    """
    if not os.path.exists(results_json):
        print(f"[Notice] Skipping Figure 2: {results_json} not found. Run experiments/evaluate_clothing_pothole_ablation.py first.")
        return

    with open(results_json, "r") as f:
        data = json.load(f)

    occ_rates = [int(x * 100) for x in data["clothing_occlusion_experiment"]["occlusion_rates"]]
    f1_prop_occ = data["clothing_occlusion_experiment"]["proposed_upper_gated_f1"]
    f1_base_occ = data["clothing_occlusion_experiment"]["baseline_unweighted_f1"]
    alpha_weights = data["clothing_occlusion_experiment"]["alpha_upper_weights"]

    pitch_deg = data["pothole_pitch_experiment"]["pitch_angles_deg"]
    f1_prop_pitch = data["pothole_pitch_experiment"]["proposed_intrinsic_torso_f1"]
    f1_base_pitch = data["pothole_pitch_experiment"]["baseline_camera_frame_f1"]

    apply_ieee_style()
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    color_prop = "#1b9e77" # Emerald
    color_base = "#d95f02" # Red/Orange
    color_gate = "#2b83ba" # Deep Blue

    # 1. Attire Occlusion Resilience (F1 Score)
    ax1 = axes[0]
    ax1.plot(occ_rates, f1_base_occ, "o--", color=color_base, label="Baseline (Ungated)")
    ax1.plot(occ_rates, f1_prop_occ, "s-", color=color_prop, label="Proposed (Gated)")
    ax1.fill_between(occ_rates, f1_base_occ, f1_prop_occ, color=color_prop, alpha=0.15, label="Attire Resilience Gap")
    ax1.set_xlabel("Lower-Limb Occlusion (%)\n(Pants $\\rightarrow$ Flowing Saree/Kurta)")
    ax1.set_ylabel("Macro $F_1$ Score (%)")
    ax1.set_title("(a) Traditional Attire Occlusion F1")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", framealpha=0.9)

    # 2. Dynamic Gating Behavior
    ax2 = axes[1]
    ax2.plot(occ_rates, alpha_weights, "D-", color=color_gate, linewidth=2.5, label="Learned $\\alpha_{\\text{upper}}$ Weight")
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Balanced (50/50)")
    ax2.set_xlabel("Lower-Limb Occlusion (%)\n(0% = Full Stride $\\rightarrow$ 100% = Masked)")
    ax2.set_ylabel("Upper-Body Reliance Weight ($\\alpha_{\\text{upper}}$)")
    ax2.set_title("(b) Postural Gate Attention Shift")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", framealpha=0.9)

    # 3. Pothole / Vehicle Pitch Shock Invariance
    ax3 = axes[2]
    ax3.plot(pitch_deg, f1_base_pitch, "^--", color=color_base, label="Camera-Frame Baseline")
    ax3.plot(pitch_deg, f1_prop_pitch, "s-", color=color_prop, label="Intrinsic Torso Frame (Ours)")
    ax3.fill_between(pitch_deg, f1_base_pitch, f1_prop_pitch, color=color_prop, alpha=0.15, label="Pitch Invariance Margin")
    ax3.set_xlabel(r"Vehicle Pitch Perturbation (Degrees)" "\n" r"(0$^\circ$ Smooth $\rightarrow$ 12$^\circ$ Pothole/Speed Bump)")
    ax3.set_ylabel("Macro $F_1$ Score (%)")
    ax3.set_title("(c) Road Roughness Invariance")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()

    out_png = os.path.join(output_dir, "fig_stress_test_ablations.png")
    out_pdf = os.path.join(output_dir, "fig_stress_test_ablations.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"[Figure 2 Saved] Stress-Test Protocol Curves:\n  - {out_png}\n  - {out_pdf}")


def plot_all_figures():
    """Generates figures only when real results exist."""
    plot_reality_gap_curves()
    plot_clothing_and_pothole_ablation()


if __name__ == "__main__":
    plot_all_figures()
