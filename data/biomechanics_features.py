"""
Physics-Informed Metric Biomechanics Feature Extraction
======================================================
Computes metric kinematic and biomechanical quantities from 3D joint coordinate sequences:
- Metric velocities and accelerations (m/s, m/s^2)
- Anthropometric bone vectors and angles
- Center of Mass (CoM) trajectory and momentum
- Torso orientation (pitch, roll, yaw)
- Gait cadence and step periodicity
"""

import numpy as np
from typing import Dict, Tuple, Optional
from data.joint_taxonomy import CANONICAL_BONES, NUM_CANONICAL_JOINTS

# Anthropometric body segment mass fractions (Winter, 2009)
SEGMENT_MASS_FRACTIONS = {
    "head": 0.081,
    "torso": 0.497,
    "upper_arms": 0.056,  # 0.028 each
    "forearms": 0.032,    # 0.016 each
    "thighs": 0.200,      # 0.100 each
    "shanks_feet": 0.134  # 0.067 each
}


def compute_metric_kinematics(
    poses_3d: np.ndarray,
    fps: float = 10.0,
    smoothing_window: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes metric joint velocities and accelerations using central finite differences.
    Args:
        poses_3d: shape (T, J, 3) where coordinates are in meters
        fps: frame rate of sensor capture
        smoothing_window: odd integer for moving average smoothing
    Returns:
        velocities: shape (T, J, 3) in m/s
        accelerations: shape (T, J, 3) in m/s^2
    """
    T, J, D = poses_3d.shape
    dt = 1.0 / fps

    # Optional temporal smoothing
    if smoothing_window > 1 and T >= smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = np.empty_like(poses_3d)
        for j in range(J):
            for d in range(D):
                smoothed[:, j, d] = np.convolve(poses_3d[:, j, d], kernel, mode='same')
    else:
        smoothed = poses_3d

    # Velocity: central difference, forward/backward at boundaries
    velocities = np.zeros_like(smoothed)
    if T > 1:
        velocities[0] = (smoothed[1] - smoothed[0]) / dt
        velocities[-1] = (smoothed[-1] - smoothed[-2]) / dt
        if T > 2:
            velocities[1:-1] = (smoothed[2:] - smoothed[:-2]) / (2.0 * dt)

    # Acceleration
    accelerations = np.zeros_like(velocities)
    if T > 1:
        accelerations[0] = (velocities[1] - velocities[0]) / dt
        accelerations[-1] = (velocities[-1] - velocities[-2]) / dt
        if T > 2:
            accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2.0 * dt)

    return velocities, accelerations


def compute_bone_vectors_and_lengths(
    poses_3d: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 3D unit bone orientation vectors and bone segment lengths.
    Args:
        poses_3d: shape (T, J, 3)
    Returns:
        bone_units: shape (T, B, 3) where B = len(CANONICAL_BONES)
        bone_lengths: shape (T, B) in meters
    """
    T, J, _ = poses_3d.shape
    B = len(CANONICAL_BONES)
    bone_units = np.zeros((T, B, 3), dtype=np.float32)
    bone_lengths = np.zeros((T, B), dtype=np.float32)

    for b_idx, (p, c) in enumerate(CANONICAL_BONES):
        vec = poses_3d[:, c, :] - poses_3d[:, p, :]
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        # Avoid division by zero
        norm_safe = np.maximum(norm, 1e-6)
        bone_units[:, b_idx, :] = vec / norm_safe
        bone_lengths[:, b_idx] = norm[..., 0]

    return bone_units, bone_lengths


def compute_center_of_mass(poses_3d: np.ndarray) -> np.ndarray:
    """
    Calculates whole-body Center of Mass (CoM) trajectory based on
    anthropometric segment mass fractions.
    Args:
        poses_3d: shape (T, 15, 3)
    Returns:
        com_trajectory: shape (T, 3)
    """
    # Key indices:
    # 0: Head, 1: Neck, 2: L-Shoulder, 3: R-Shoulder, 4: L-Elbow, 5: R-Elbow
    # 6: L-Wrist, 7: R-Wrist, 8: Pelvis, 9: L-Hip, 10: R-Hip, 11: L-Knee, 12: R-Knee
    # 13: L-Ankle, 14: R-Ankle
    head_pos = poses_3d[:, 0, :]
    torso_pos = 0.5 * (poses_3d[:, 1, :] + poses_3d[:, 8, :])
    upper_arms = 0.5 * (
        0.5 * (poses_3d[:, 2, :] + poses_3d[:, 4, :]) +
        0.5 * (poses_3d[:, 3, :] + poses_3d[:, 5, :])
    )
    forearms = 0.5 * (
        0.5 * (poses_3d[:, 4, :] + poses_3d[:, 6, :]) +
        0.5 * (poses_3d[:, 5, :] + poses_3d[:, 7, :])
    )
    thighs = 0.5 * (
        0.5 * (poses_3d[:, 9, :] + poses_3d[:, 11, :]) +
        0.5 * (poses_3d[:, 10, :] + poses_3d[:, 12, :])
    )
    shanks = 0.5 * (
        0.5 * (poses_3d[:, 11, :] + poses_3d[:, 13, :]) +
        0.5 * (poses_3d[:, 12, :] + poses_3d[:, 14, :])
    )

    com = (
        SEGMENT_MASS_FRACTIONS["head"] * head_pos +
        SEGMENT_MASS_FRACTIONS["torso"] * torso_pos +
        SEGMENT_MASS_FRACTIONS["upper_arms"] * upper_arms +
        SEGMENT_MASS_FRACTIONS["forearms"] * forearms +
        SEGMENT_MASS_FRACTIONS["thighs"] * thighs +
        SEGMENT_MASS_FRACTIONS["shanks_feet"] * shanks
    )
    return com


def compute_torso_orientation(poses_3d: np.ndarray) -> np.ndarray:
    """
    Computes 3D orientation vectors of the torso:
    - Spine direction vector (Pelvis -> Neck)
    - Shoulder lateral vector (R-Shoulder -> L-Shoulder)
    - Normal vector (chest facing direction: spine x lateral)
    Args:
        poses_3d: shape (T, 15, 3)
    Returns:
        torso_features: shape (T, 9) containing [spine_dir (3), lateral_dir (3), normal_dir (3)]
    """
    T = poses_3d.shape[0]
    spine = poses_3d[:, 1, :] - poses_3d[:, 8, :]
    spine_norm = spine / np.maximum(np.linalg.norm(spine, axis=-1, keepdims=True), 1e-6)

    lateral = poses_3d[:, 2, :] - poses_3d[:, 3, :]
    lateral_norm = lateral / np.maximum(np.linalg.norm(lateral, axis=-1, keepdims=True), 1e-6)

    # Normal pointing forward out of chest
    normal = np.cross(spine_norm, lateral_norm)
    normal_norm = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)

    return np.concatenate([spine_norm, lateral_norm, normal_norm], axis=-1)


def compute_gait_cadence(poses_3d: np.ndarray, fps: float = 10.0) -> Tuple[float, float]:
    """
    Estimates gait step frequency (Hz) and stride length (meters) from ankle trajectories.
    Args:
        poses_3d: shape (T, 15, 3)
        fps: sampling frequency
    Returns:
        cadence_hz: step frequency in Hz
        stride_length: mean stride displacement in meters
    """
    T = poses_3d.shape[0]
    if T < 4:
        return 0.0, 0.0

    l_ankle = poses_3d[:, 13, :]
    r_ankle = poses_3d[:, 14, :]

    # Inter-ankle distance along ground plane (x, z)
    ankle_dist = np.linalg.norm(l_ankle[:, [0, 2]] - r_ankle[:, [0, 2]], axis=-1)

    # Stride length proxy: max inter-ankle distance
    stride_length = float(np.percentile(ankle_dist, 90))

    # Zero-crossing or peak counting for cadence
    centered = ankle_dist - np.mean(ankle_dist)
    zero_crossings = np.where(np.diff(np.sign(centered)))[0]
    num_steps = len(zero_crossings)
    duration_s = max(T / fps, 1e-3)
    cadence_hz = float(num_steps / (2.0 * duration_s))

    return cadence_hz, stride_length


def extract_full_biomechanical_feature_vector(
    poses_3d: np.ndarray,
    fps: float = 10.0
) -> Dict[str, np.ndarray]:
    """
    Aggregates all metric biomechanical features into a dictionary of arrays.
    Returns:
        dict with keys:
            'positions': (T, 15, 3)
            'velocities': (T, 15, 3)
            'accelerations': (T, 15, 3)
            'bone_units': (T, 14, 3)
            'bone_lengths': (T, 14)
            'com_trajectory': (T, 3)
            'com_velocity': (T, 3)
            'torso_frames': (T, 9)
            'cadence_stride': (2,)
    """
    velocities, accelerations = compute_metric_kinematics(poses_3d, fps=fps)
    bone_units, bone_lengths = compute_bone_vectors_and_lengths(poses_3d)
    com = compute_center_of_mass(poses_3d)

    dt = 1.0 / fps
    com_vel = np.zeros_like(com)
    if len(com) > 1:
        com_vel[1:] = (com[1:] - com[:-1]) / dt
        com_vel[0] = com_vel[1]

    torso_frames = compute_torso_orientation(poses_3d)
    cadence_hz, stride_len = compute_gait_cadence(poses_3d, fps=fps)

    return {
        "positions": poses_3d.astype(np.float32),
        "velocities": velocities.astype(np.float32),
        "accelerations": accelerations.astype(np.float32),
        "bone_units": bone_units.astype(np.float32),
        "bone_lengths": bone_lengths.astype(np.float32),
        "com_trajectory": com.astype(np.float32),
        "com_velocity": com_vel.astype(np.float32),
        "torso_frames": torso_frames.astype(np.float32),
        "cadence_stride": np.array([cadence_hz, stride_len], dtype=np.float32),
    }
