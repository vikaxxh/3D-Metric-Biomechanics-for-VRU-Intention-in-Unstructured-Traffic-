"""
Joint Taxonomy and Kinematic Skeleton Topology
==============================================
Standardizes keypoint representations across:
1. Waymo Open Dataset 3D Keypoints (14 joints)
2. IDD-PeD / MMPose COCO 2D Keypoints (17 joints)
3. Canonical Unified Biomechanical 3D/2D Skeleton (15 joints)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# Canonical Unified Biomechanical Skeleton (15 joints)
CANONICAL_JOINTS: List[str] = [
    "head",           # 0
    "neck",           # 1
    "left_shoulder",  # 2
    "right_shoulder", # 3
    "left_elbow",     # 4
    "right_elbow",    # 5
    "left_wrist",     # 6
    "right_wrist",    # 7
    "pelvis",         # 8 (Root / CoM proxy)
    "left_hip",       # 9
    "right_hip",      # 10
    "left_knee",      # 11
    "right_knee",     # 12
    "left_ankle",     # 13
    "right_ankle",    # 14
]

NUM_CANONICAL_JOINTS = len(CANONICAL_JOINTS)

# Kinematic Tree Bones: (Parent, Child)
CANONICAL_BONES: List[Tuple[int, int]] = [
    (0, 1),   # Head -> Neck
    (1, 2),   # Neck -> L Shoulder
    (2, 4),   # L Shoulder -> L Elbow
    (4, 6),   # L Elbow -> L Wrist
    (1, 3),   # Neck -> R Shoulder
    (3, 5),   # R Shoulder -> R Elbow
    (5, 7),   # R Elbow -> R Wrist
    (1, 8),   # Neck -> Pelvis (Spine)
    (8, 9),   # Pelvis -> L Hip
    (9, 11),  # L Hip -> L Knee
    (11, 13), # L Knee -> L Ankle
    (8, 10),  # Pelvis -> R Hip
    (10, 12), # R Hip -> R Knee
    (12, 14), # R Knee -> R Ankle
]

# Standard Anthropometric Bone Lengths (in meters) for an average 1.72m adult human
# Derived from Winter's Biomechanics and Motor Control of Human Movement
ANTHROPOMETRIC_BONE_LENGTHS: Dict[Tuple[int, int], float] = {
    (0, 1): 0.20,   # Head to Neck
    (1, 2): 0.18,   # Neck to Left Shoulder
    (2, 4): 0.28,   # Left Upper Arm
    (4, 6): 0.25,   # Left Forearm
    (1, 3): 0.18,   # Neck to Right Shoulder
    (3, 5): 0.28,   # Right Upper Arm
    (5, 7): 0.25,   # Right Forearm
    (1, 8): 0.48,   # Spine (Neck to Pelvis)
    (8, 9): 0.15,   # Pelvis to Left Hip
    (9, 11): 0.42,  # Left Femur (Thigh)
    (11, 13): 0.40, # Left Tibia (Shank)
    (8, 10): 0.15,  # Pelvis to Right Hip
    (10, 12): 0.42, # Right Femur (Thigh)
    (12, 14): 0.40, # Right Tibia (Shank)
}

# Waymo 14-Keypoint Names
WAYMO_KEYPOINTS = [
    "nose",           # 0
    "left_shoulder",  # 1
    "right_shoulder", # 2
    "left_elbow",     # 3
    "right_elbow",    # 4
    "left_wrist",     # 5
    "right_wrist",    # 6
    "left_hip",       # 7
    "right_hip",      # 8
    "left_knee",      # 9
    "right_knee",     # 10
    "left_ankle",     # 11
    "right_ankle",    # 12
    "head_center",    # 13
]

# MMPose / COCO 17-Keypoint Names (used in IDD-PeD)
COCO_KEYPOINTS = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]


def waymo_to_canonical(waymo_kpts: np.ndarray) -> np.ndarray:
    """
    Converts Waymo 14-keypoint array to Canonical 15-keypoint array.
    Args:
        waymo_kpts: shape (..., 14, D) where D is 2 or 3 (x,y,[z])
    Returns:
        canonical_kpts: shape (..., 15, D)
    """
    prefix_shape = waymo_kpts.shape[:-2]
    d = waymo_kpts.shape[-1]
    canonical = np.zeros((*prefix_shape, 15, d), dtype=waymo_kpts.dtype)

    # Direct mappings
    canonical[..., 0, :] = waymo_kpts[..., 13, :]  # Head
    # Neck = midpoint of left and right shoulder
    canonical[..., 1, :] = 0.5 * (waymo_kpts[..., 1, :] + waymo_kpts[..., 2, :])
    canonical[..., 2, :] = waymo_kpts[..., 1, :]   # L Shoulder
    canonical[..., 3, :] = waymo_kpts[..., 2, :]   # R Shoulder
    canonical[..., 4, :] = waymo_kpts[..., 3, :]   # L Elbow
    canonical[..., 5, :] = waymo_kpts[..., 4, :]   # R Elbow
    canonical[..., 6, :] = waymo_kpts[..., 5, :]   # L Wrist
    canonical[..., 7, :] = waymo_kpts[..., 6, :]   # R Wrist
    # Pelvis = midpoint of left and right hip
    canonical[..., 8, :] = 0.5 * (waymo_kpts[..., 7, :] + waymo_kpts[..., 8, :])
    canonical[..., 9, :] = waymo_kpts[..., 7, :]   # L Hip
    canonical[..., 10, :] = waymo_kpts[..., 8, :]  # R Hip
    canonical[..., 11, :] = waymo_kpts[..., 9, :]  # L Knee
    canonical[..., 12, :] = waymo_kpts[..., 10, :] # R Knee
    canonical[..., 13, :] = waymo_kpts[..., 11, :] # L Ankle
    canonical[..., 14, :] = waymo_kpts[..., 12, :] # R Ankle

    return canonical


def coco_to_canonical(coco_kpts: np.ndarray) -> np.ndarray:
    """
    Converts COCO/MMPose 17-keypoint array (IDD-PeD) to Canonical 15-keypoint array.
    Args:
        coco_kpts: shape (..., 17, D) where D is 2 or 3 (u, v, [confidence])
    Returns:
        canonical_kpts: shape (..., 15, D)
    """
    prefix_shape = coco_kpts.shape[:-2]
    d = coco_kpts.shape[-1]
    canonical = np.zeros((*prefix_shape, 15, d), dtype=coco_kpts.dtype)

    # Head is proxy by nose or ear-midpoint
    canonical[..., 0, :] = coco_kpts[..., 0, :]   # Head (Nose)
    # Neck = midpoint of shoulders
    canonical[..., 1, :] = 0.5 * (coco_kpts[..., 5, :] + coco_kpts[..., 6, :])
    canonical[..., 2, :] = coco_kpts[..., 5, :]   # L Shoulder
    canonical[..., 3, :] = coco_kpts[..., 6, :]   # R Shoulder
    canonical[..., 4, :] = coco_kpts[..., 7, :]   # L Elbow
    canonical[..., 5, :] = coco_kpts[..., 8, :]   # R Elbow
    canonical[..., 6, :] = coco_kpts[..., 9, :]   # L Wrist
    canonical[..., 7, :] = coco_kpts[..., 10, :]  # R Wrist
    # Pelvis = midpoint of hips
    canonical[..., 8, :] = 0.5 * (coco_kpts[..., 11, :] + coco_kpts[..., 12, :])
    canonical[..., 9, :] = coco_kpts[..., 11, :]  # L Hip
    canonical[..., 10, :] = coco_kpts[..., 12, :] # R Hip
    canonical[..., 11, :] = coco_kpts[..., 13, :] # L Knee
    canonical[..., 12, :] = coco_kpts[..., 14, :] # R Knee
    canonical[..., 13, :] = coco_kpts[..., 15, :] # L Ankle
    canonical[..., 14, :] = coco_kpts[..., 16, :] # R Ankle

    return canonical


def get_adjacency_matrix() -> np.ndarray:
    """
    Constructs normalized graph adjacency matrix for the 15-joint canonical skeleton
    with self-loops (for GCN / Spatio-Temporal Graph layers).
    """
    adj = np.eye(NUM_CANONICAL_JOINTS, dtype=np.float32)
    for p, c in CANONICAL_BONES:
        adj[p, c] = 1.0
        adj[c, p] = 1.0

    # Symmetric Laplacian normalization D^(-1/2) * A * D^(-1/2)
    degrees = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = np.diag(d_inv_sqrt)
    norm_adj = d_mat @ adj @ d_mat
    return norm_adj
