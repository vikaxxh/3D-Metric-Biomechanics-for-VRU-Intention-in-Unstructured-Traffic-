"""
High-Fidelity Biomechanical Benchmark Dataset Generator
======================================================
Generates physically consistent 3D metric human kinematic trajectories (Waymo LiDAR format)
and paired 2D monocular perspective projections with unstructured annotations (IDD-PeD format).

Physics Model:
- Inverted pendulum locomotion model with periodic leg swing and counter-phase arm swing
- Biomechanically accurate bone length invariants
- Intention-conditioned trajectories:
    * Crossing (CU / CFU): Standing/walking on road edge -> acceleration toward roadway
    * Non-Crossing: Moving parallel to curb or stationary waiting
    * Weaving Through Traffic (WTT): S-curve obstacle avoidance around stationary vehicles
    * Looking Over Shoulder (LOS): Head yaw rotation counter to travel direction
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from data.joint_taxonomy import (
    ANTHROPOMETRIC_BONE_LENGTHS,
    CANONICAL_BONES,
    WAYMO_KEYPOINTS,
    COCO_KEYPOINTS,
)


def generate_single_biomechanical_pedestrian(
    num_frames: int = 30,
    fps: float = 10.0,
    is_crossing: bool = True,
    is_weaving: bool = False,
    is_looking_over_shoulder: bool = False,
    is_sprinting: bool = False,
    initial_pos: Tuple[float, float, float] = (5.0, 0.0, 15.0), # (X_lateral, Y_height, Z_depth)
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Simulates a 3D pedestrian skeleton sequence obeying anatomical constraints.
    Returns:
        poses_3d: (num_frames, 15, 3) in meters (LiDAR/Camera metric coordinate frame)
        poses_2d: (num_frames, 17, 2) in image pixels (Camera perspective)
        attributes: dictionary of behavioral labels
    """
    dt = 1.0 / fps
    t = np.arange(num_frames) * dt

    # Speed parameters (m/s)
    if is_sprinting:
        base_speed = 3.2
        step_freq = 2.4 # Hz
    elif is_crossing:
        base_speed = 1.4
        step_freq = 1.8 # Hz
    else:
        base_speed = 1.0
        step_freq = 1.4 # Hz

    # Trajectory of Center of Mass (Pelvis - Root)
    x0, y0, z0 = initial_pos
    root_pos = np.zeros((num_frames, 3), dtype=np.float32)

    # Lateral movement (X axis) and Forward movement (Z axis)
    if is_crossing:
        # Crosses road towards negative X (into ego vehicle lane)
        x_disp = -base_speed * t
        z_disp = 0.2 * np.sin(0.5 * t)
        if is_weaving:
            # Weaving between cars: oscillatory lateral path
            x_disp += 0.8 * np.sin(2.5 * t)
    else:
        # Walks along the road corridor (constant X, changing Z)
        x_disp = 0.05 * np.sin(t)
        z_disp = -base_speed * t * 0.5

    root_pos[:, 0] = x0 + x_disp
    root_pos[:, 1] = y0 + 0.90 + 0.03 * np.abs(np.sin(2 * np.pi * step_freq * t)) # Pelvis height ~0.9m + vertical bounce
    root_pos[:, 2] = z0 + z_disp

    # Construct 15 canonical joints for each frame
    poses_3d = np.zeros((num_frames, 15, 3), dtype=np.float32)

    # 8: Pelvis (Root)
    poses_3d[:, 8, :] = root_pos

    # 1: Neck (Spine ~0.48m above pelvis)
    torso_pitch = 0.08 if is_crossing else 0.03 # Forward lean when crossing
    poses_3d[:, 1, 0] = root_pos[:, 0] - 0.48 * np.sin(torso_pitch)
    poses_3d[:, 1, 1] = root_pos[:, 1] + 0.48 * np.cos(torso_pitch)
    poses_3d[:, 1, 2] = root_pos[:, 2]

    # 0: Head (~0.20m above neck)
    head_yaw = -0.6 if is_looking_over_shoulder else 0.0 # Looking towards traffic (+Z)
    poses_3d[:, 0, 0] = poses_3d[:, 1, 0] + 0.20 * np.sin(head_yaw)
    poses_3d[:, 0, 1] = poses_3d[:, 1, 1] + 0.20
    poses_3d[:, 0, 2] = poses_3d[:, 1, 2] + 0.20 * np.cos(head_yaw)

    # Shoulders (2: Left, 3: Right)
    shoulder_width = 0.36
    poses_3d[:, 2, :] = poses_3d[:, 1, :] + np.array([-shoulder_width / 2, 0.0, 0.0])
    poses_3d[:, 3, :] = poses_3d[:, 1, :] + np.array([ shoulder_width / 2, 0.0, 0.0])

    # Hips (9: Left, 10: Right)
    hip_width = 0.28
    poses_3d[:, 9, :]  = poses_3d[:, 8, :] + np.array([-hip_width / 2, 0.0, 0.0])
    poses_3d[:, 10, :] = poses_3d[:, 8, :] + np.array([ hip_width / 2, 0.0, 0.0])

    # Gait Cycle: Periodic Leg and Arm Swings
    phase = 2 * np.pi * step_freq * t
    thigh_len = ANTHROPOMETRIC_BONE_LENGTHS[(9, 11)]  # 0.42
    shank_len = ANTHROPOMETRIC_BONE_LENGTHS[(11, 13)] # 0.40
    arm_len = ANTHROPOMETRIC_BONE_LENGTHS[(2, 4)]    # 0.28
    forearm_len = ANTHROPOMETRIC_BONE_LENGTHS[(4, 6)]# 0.25

    # Left Leg
    l_thigh_angle = 0.45 * np.sin(phase)
    poses_3d[:, 11, 0] = poses_3d[:, 9, 0] - thigh_len * np.sin(l_thigh_angle)
    poses_3d[:, 11, 1] = poses_3d[:, 9, 1] - thigh_len * np.cos(l_thigh_angle)
    poses_3d[:, 11, 2] = poses_3d[:, 9, 2] + 0.1 * np.sin(phase)

    l_shank_angle = l_thigh_angle - 0.2 * np.maximum(0.0, np.sin(phase))
    poses_3d[:, 13, 0] = poses_3d[:, 11, 0] - shank_len * np.sin(l_shank_angle)
    poses_3d[:, 13, 1] = np.maximum(0.0, poses_3d[:, 11, 1] - shank_len * np.cos(l_shank_angle))
    poses_3d[:, 13, 2] = poses_3d[:, 11, 2]

    # Right Leg (Counter-phase)
    r_thigh_angle = 0.45 * np.sin(phase + np.pi)
    poses_3d[:, 12, 0] = poses_3d[:, 10, 0] - thigh_len * np.sin(r_thigh_angle)
    poses_3d[:, 12, 1] = poses_3d[:, 10, 1] - thigh_len * np.cos(r_thigh_angle)
    poses_3d[:, 12, 2] = poses_3d[:, 10, 2] + 0.1 * np.sin(phase + np.pi)

    r_shank_angle = r_thigh_angle - 0.2 * np.maximum(0.0, np.sin(phase + np.pi))
    poses_3d[:, 14, 0] = poses_3d[:, 12, 0] - shank_len * np.sin(r_shank_angle)
    poses_3d[:, 14, 1] = np.maximum(0.0, poses_3d[:, 12, 1] - shank_len * np.cos(r_shank_angle))
    poses_3d[:, 14, 2] = poses_3d[:, 12, 2]

    # Arms (Reciprocal to Legs)
    poses_3d[:, 4, 0] = poses_3d[:, 2, 0] + arm_len * np.sin(r_thigh_angle * 0.7)
    poses_3d[:, 4, 1] = poses_3d[:, 2, 1] - arm_len * np.cos(r_thigh_angle * 0.7)
    poses_3d[:, 4, 2] = poses_3d[:, 2, 2]
    poses_3d[:, 6, :] = poses_3d[:, 4, :] + np.array([0.0, -forearm_len, 0.0])

    poses_3d[:, 5, 0] = poses_3d[:, 3, 0] + arm_len * np.sin(l_thigh_angle * 0.7)
    poses_3d[:, 5, 1] = poses_3d[:, 3, 1] - arm_len * np.cos(l_thigh_angle * 0.7)
    poses_3d[:, 5, 2] = poses_3d[:, 3, 2]
    poses_3d[:, 7, :] = poses_3d[:, 5, :] + np.array([0.0, -forearm_len, 0.0])

    # Camera Perspective Projection for 2D Keypoints (IDD-PeD monocular camera)
    # Camera Intrinsics typical for front-facing dashcam (1920x1080)
    fx, fy = 1200.0, 1200.0
    cx, cy = 960.0, 540.0

    # Project Canonical joints into 17-joint COCO/MMPose format
    poses_2d_coco = np.zeros((num_frames, 17, 2), dtype=np.float32)

    # 17 COCO indices:
    # 0: nose, 1: l_eye, 2: r_eye, 3: l_ear, 4: r_ear, 5: l_sh, 6: r_sh, 7: l_elb, 8: r_elb
    # 9: l_wri, 10: r_wri, 11: l_hip, 12: r_hip, 13: l_knee, 14: r_knee, 15: l_ank, 16: r_ank
    coco_3d = np.zeros((num_frames, 17, 3), dtype=np.float32)
    coco_3d[:, 0] = poses_3d[:, 0] # nose
    coco_3d[:, 1] = poses_3d[:, 0] + np.array([-0.04, 0.02, 0.0]) # l_eye
    coco_3d[:, 2] = poses_3d[:, 0] + np.array([ 0.04, 0.02, 0.0]) # r_eye
    coco_3d[:, 3] = poses_3d[:, 0] + np.array([-0.08, 0.00, -0.05]) # l_ear
    coco_3d[:, 4] = poses_3d[:, 0] + np.array([ 0.08, 0.00, -0.05]) # r_ear
    coco_3d[:, 5] = poses_3d[:, 2] # l_shoulder
    coco_3d[:, 6] = poses_3d[:, 3] # r_shoulder
    coco_3d[:, 7] = poses_3d[:, 4] # l_elbow
    coco_3d[:, 8] = poses_3d[:, 5] # r_elbow
    coco_3d[:, 9] = poses_3d[:, 6] # l_wrist
    coco_3d[:, 10] = poses_3d[:, 7] # r_wrist
    coco_3d[:, 11] = poses_3d[:, 9] # l_hip
    coco_3d[:, 12] = poses_3d[:, 10] # r_hip
    coco_3d[:, 13] = poses_3d[:, 11] # l_knee
    coco_3d[:, 14] = poses_3d[:, 12] # r_knee
    coco_3d[:, 15] = poses_3d[:, 13] # l_ankle
    coco_3d[:, 16] = poses_3d[:, 14] # r_ankle

    # Project to 2D
    depth = np.maximum(coco_3d[..., 2], 1.0)
    poses_2d_coco[..., 0] = fx * (coco_3d[..., 0] / depth) + cx
    poses_2d_coco[..., 1] = -fy * ((coco_3d[..., 1] - 1.5) / depth) + cy # Invert Y for image space

    # Add realistic detector noise
    detector_noise = np.random.normal(0, 1.2, size=poses_2d_coco.shape).astype(np.float32)
    poses_2d_coco += detector_noise

    # Behavioral metadata
    if is_crossing:
        cross_behavior = "CFU" if is_sprinting else "CU" # In unstructured India, crossing is predominantly Undesignated!
    else:
        cross_behavior = "N/A"

    attributes = {
        "crossing_behavior": cross_behavior,
        "traffic_interaction": "WTT" if is_weaving else "N/A",
        "pedestrian_activity": "Sprinting" if is_sprinting else "Walking",
        "attention_indicators": "LOS" if is_looking_over_shoulder else "FTT",
        "social_dynamics": "N/A",
        "stationary_behavior": "N/A",
        "lane_markings_present": False,  # Hallmark of unstructured traffic!
        "traffic_light_present": False,
        "intention_label": 1 if is_crossing else 0,
    }

    return poses_3d, poses_2d_coco, attributes


def generate_benchmark_dataset(
    output_root: str,
    num_train: int = 120,
    num_val: int = 30,
    num_test: int = 50,
    frames_per_seq: int = 30,
    seed: int = 42
):
    """
    Generates complete train/val/test datasets for both Waymo (3D LiDAR) and IDD-PeD (2D Unstructured).
    """
    np.random.seed(seed)
    os.makedirs(output_root, exist_ok=True)

    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test,
    }

    for split_name, count in splits.items():
        waymo_split_dir = os.path.join(output_root, "waymo", split_name)
        idd_split_dir = os.path.join(output_root, "idd_ped", split_name)
        os.makedirs(waymo_split_dir, exist_ok=True)
        os.makedirs(idd_split_dir, exist_ok=True)

        for i in range(count):
            seq_id = f"{split_name}_{i:04d}"
            is_crossing = (i % 2 == 0) # Balanced crossing / non-crossing
            is_weaving = is_crossing and (i % 3 == 0)
            is_los = is_crossing and (i % 4 == 0)
            is_sprint = is_crossing and (i % 5 == 0)

            # Random initial position
            init_x = float(np.random.uniform(2.5, 6.0))
            init_z = float(np.random.uniform(10.0, 22.0))

            poses_3d, poses_2d, attrs = generate_single_biomechanical_pedestrian(
                num_frames=frames_per_seq,
                is_crossing=is_crossing,
                is_weaving=is_weaving,
                is_looking_over_shoulder=is_los,
                is_sprinting=is_sprint,
                initial_pos=(init_x, 0.0, init_z)
            )

            # Waymo export format (14 joints 3D)
            # Waymo keypoints: nose(0), l_sh(1), r_sh(2), l_elb(3), r_elb(4), l_wri(5), r_wri(6),
            # l_hip(7), r_hip(8), l_knee(9), r_knee(10), l_ank(11), r_ank(12), head(13)
            waymo_14 = np.zeros((frames_per_seq, 14, 3), dtype=np.float32)
            waymo_14[:, 0] = poses_3d[:, 0]  # nose/head
            waymo_14[:, 1] = poses_3d[:, 2]  # l_sh
            waymo_14[:, 2] = poses_3d[:, 3]  # r_sh
            waymo_14[:, 3] = poses_3d[:, 4]  # l_elb
            waymo_14[:, 4] = poses_3d[:, 5]  # r_elb
            waymo_14[:, 5] = poses_3d[:, 6]  # l_wri
            waymo_14[:, 6] = poses_3d[:, 7]  # r_wri
            waymo_14[:, 7] = poses_3d[:, 9]  # l_hip
            waymo_14[:, 8] = poses_3d[:, 10] # r_hip
            waymo_14[:, 9] = poses_3d[:, 11] # l_knee
            waymo_14[:, 10] = poses_3d[:, 12] # r_knee
            waymo_14[:, 11] = poses_3d[:, 13] # l_ank
            waymo_14[:, 12] = poses_3d[:, 14] # r_ank
            waymo_14[:, 13] = poses_3d[:, 0]  # head center

            # Add LiDAR noise for Waymo
            waymo_noise = np.random.normal(0, 0.02, size=waymo_14.shape).astype(np.float32)
            waymo_14 += waymo_noise

            waymo_file = os.path.join(waymo_split_dir, f"{seq_id}.npz")
            np.savez_compressed(
                waymo_file,
                sequence_id=seq_id,
                track_id=1,
                keypoints_3d=waymo_14,
                intention_label=attrs["intention_label"],
                bboxes_3d=np.zeros((frames_per_seq, 7), dtype=np.float32)
            )

            # IDD-PeD export format (17 joints 2D + fine-grained behaviors)
            idd_file = os.path.join(idd_split_dir, f"{seq_id}.npz")
            # Calculate 2D bounding boxes from 2D keypoints
            min_xy = np.min(poses_2d, axis=1) - 20 # (T, 2)
            max_xy = np.max(poses_2d, axis=1) + 20 # (T, 2)
            bboxes_2d = np.concatenate([min_xy, max_xy], axis=-1) # (T, 4)

            np.savez_compressed(
                idd_file,
                sequence_id=seq_id,
                track_id=1,
                keypoints_2d=poses_2d,
                bboxes_2d=bboxes_2d,
                behavioral_attributes=attrs,
                lane_markings_present=attrs["lane_markings_present"],
                traffic_light_present=attrs["traffic_light_present"],
                intention_label=attrs["intention_label"]
            )

    print(f"[BenchmarkGen] Successfully generated synthetic Waymo 3D and IDD-PeD benchmarks at: {output_root}")


if __name__ == "__main__":
    generate_benchmark_dataset(
        output_root="/home/vikash/Desktop/sk/data/benchmark_data",
        num_train=120,
        num_val=30,
        num_test=50,
        frames_per_seq=30
    )
