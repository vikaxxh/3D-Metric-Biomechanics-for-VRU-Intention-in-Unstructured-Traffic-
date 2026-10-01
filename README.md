# Bridging the Reality Gap: 3D Metric Biomechanics for VRU Intention in Unstructured Traffic

Official implementation and benchmark evaluation harness for pedestrian crossing intention prediction in unstructured urban environments (e.g., IDD-PeD), using 3D metric human biomechanical priors distilled from LiDAR-annotated data (Waymo Open Dataset).

---

## 1. Research Motivation & Core Contributions

Existing vision-centric models for Vulnerable Road User (VRU) intention (e.g., trained on PIE or JAAD) rely heavily on structured Western traffic assumptions: designated zebra crosswalks, painted lane markers, strict signal compliance, and rigid Western attire. 

When transferred to unstructured mixed traffic in the Global South (e.g., Indian traffic recorded in **IDD-PeD**), these models experience catastrophic failure modes:
1. **Infrastructure Dropout**: Absence of painted lanes and traffic lights; pedestrians cross undesignated at arbitrary points.
2. **Attire Occlusion**: Flowing traditional clothing (saris, kurtas, dhotis) obscures lower-limb keypoints (knees, ankles).
3. **Camera Pitch Shocks**: Road irregularities (potholes, speed bumps, unpaved surfaces) induce severe vehicle pitch and roll, corrupting 2D monocular joint trajectories with non-human vertical velocity spikes.
4. **Monocular Scale Ambiguity**: Lack of metric 3D depth leaves distances and approach speeds uncalibrated.

### Core Contributions
* **Cross-Modal 3D LiDAR $\rightarrow$ Monocular Transfer**: Leverages calibrated Waymo 3D LiDAR skeletons as a metric kinematic teacher and distills invariant physical representations into uncalibrated monocular IDD-PeD sequences via Multi-Kernel Maximum Mean Discrepancy (MK-MMD) and anthropometric invariants.
* **Physiologically Grounded Postural Gating**: Decouples the human skeleton into an upper-body sub-graph (head, neck, torso) and lower-body sub-graph (pelvis, legs). Grounded in Anticipatory Postural Adjustments (APAs for forward CoM destabilization; Bouisset & Do 2008) and gaze-locomotion steering synergies (head-first path alignment; Hollands et al. 2001, 2002), dynamically shifting reliance to upper posture when lower limbs are occluded by traditional attire.
* **Camera-Invariant Intrinsic Torso Normalization**: Gram-Schmidt orthonormalization of the pedestrian spine and shoulder vectors, rendering 3D articulation mathematically invariant to vehicle pitch vibrations from potholes and speed bumps.
* **Unstructured Traffic Stress-Test Protocol**: Empirical robustness benchmarking under progressive lower-limb attire masking (0% to 100%), road roughness pitch perturbations ($0^\circ$ to $12^\circ$), and complete road infrastructure dropout.

---

## 2. Directory Layout

```text
├── checkpoints/              # Model weights (waymo_pretrain_best.pt, idd_transfer_best.pt)
├── configs/                  # Experiment YAML configuration files
│   ├── default_config.yaml
│   ├── waymo_pretrain.yaml
│   ├── idd_ped_transfer.yaml
│   ├── ablation_clothing_potholes.yaml
│   └── ablation_lane_dropout.yaml
├── data/
│   ├── raw_idd_ped/          # Official cloned IDD-PeD repo (IIIT Hyderabad)
│   ├── real_data/idd_ped/    # 100% genuine extracted video-disjoint IDD-PeD tracks
│   │   ├── train/            # gp_set_0002, 0004, 0006, 0007 (1,357 tracks, 92k frames)
│   │   └── test/             # gp_set_0003, 0005, 0008, 0009 (1,132 tracks, 58k frames)
│   ├── joint_taxonomy.py     # Canonical 15-joint kinematic tree & Winter's bone lengths
│   ├── biomechanics_features.py # Metric velocity, CoM trajectory, acceleration features
│   ├── idd_ped_adapter.py    # Real IDD-PeD data parser (MMPose 17 COCO -> Canonical 15)
│   └── dataset_loader.py     # PyTorch DataLoader with temporal sequence standardization
├── models/
│   ├── biolift_former.py     # 2D-to-3D Spatio-temporal lifting transformer
│   ├── clothing_occlusion_gater.py # Upper vs. lower body anticipatory postural gater
│   ├── ego_motion_compensator.py   # Intrinsic torso Gram-Schmidt normalizer
│   ├── cross_domain_bridge.py      # Teacher-student cross-domain coordination
│   ├── context_encoder.py    # Visual context and infrastructure dropout encoder
│   └── vr_intention_classifier.py # Multi-task classifier with architectural switches
├── losses/
│   ├── biomech_physics_loss.py # Anthropometric bone prior, rigid invariance, smoothness
│   └── domain_alignment_loss.py # Multi-Kernel MMD and domain discriminator
├── evaluation/
│   ├── icra_benchmark_comparison.py # Head-to-head comparison vs. ICRA 2025 baselines
│   ├── metrics.py            # Accuracy, Macro F1, AUC-ROC, bone error
│   └── results/              # Authentic JSON results and LaTeX tables
├── visualization/
│   ├── plot_reality_gap_curves.py # IEEE publication figures (PNG and vector PDF)
│   └── figures/              # Generated publication figures
├── scripts_hpc/              # HPC PBS Pro and Slurm submission pipeline
│   ├── submit_hpc_pipeline.sh # Master submitter with dependency chaining (-W depend=afterok)
│   ├── pbs_waymo_pretrain_gpu.pbs
│   ├── pbs_idd_ped_transfer_gpu.pbs
│   ├── pbs_reality_gap_ablation.pbs
│   └── pbs_clothing_potholes_ablation.pbs
└── tools/
    ├── audit_real_idd_ped.py # Forensic data auditor & split independence verifier
    └── ingest_real_idd_ped.py # Real data extractor matching official video frames
```

---

## 3. Dataset Audit & Official Splits

A forensic data audit verifies **100% real human-annotated video sequences** with **zero data leakage**:

| Split | Group Sets | Videos | Tracks | Frames | Crossing | Not Crossing | Pose Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | `gp_set_0002, 0004, 0006, 0007` | 14 | 1,357 | 92,576 | 229 (16.9%) | 1,128 (83.1%) | 100% |
| **Test** | `gp_set_0003, 0005, 0008, 0009` | 13 | 1,132 | 58,492 | 183 (16.2%) | 949 (83.8%) | 100% |

* **Video Overlap**: 0 (Strictly video-disjoint).
* **Track ID Overlap**: 0 (Zero identity contamination).

---

## 4. How to Run

### Step 1: Verify the Dataset
```bash
python3 tools/audit_real_idd_ped.py
```

### Step 2: Run Locally (Smoke Test or Evaluation)
```bash
# Evaluate genuine clothing & pothole ablations (requires checkpoint)
python3 experiments/evaluate_clothing_pothole_ablation.py

# Evaluate infrastructure dropout reality gap
python3 experiments/evaluate_reality_gap.py

# Generate IEEE publication figures from real evaluation artifacts
python3 visualization/plot_reality_gap_curves.py
```

### Step 3: Run Full Training on HPC Cluster (PBS Pro / Torque)
```bash
# Submit all 4 stages sequentially with automatic dependency chaining
./scripts_hpc/submit_hpc_pipeline.sh pbs dgx
```
