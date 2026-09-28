"""
Ego-Motion & Road Roughness Invariance Module
=============================================
Addresses Reviewer Critique on Indian Traffic Potholes & Vehicle Pitch:
When a vehicle drives over speed breakers, unpaved gravel, or potholes in India,
the monocular camera experiences severe pitch, roll, and vibration. This induces
false vertical velocities in 2D/3D joint detections.

This module computes an Intrinsic Pedestrian Coordinate Frame:
1. Root Pelvis Decoupling: Articulation is made strictly relative to pelvis root.
2. Orthogonal Torso Basis Alignment: Computes orthonormal basis [u_x, u_y, u_z] from
   spine and shoulder vectors.
3. Invariant Projection: Transforms joint coordinates into the pedestrian's intrinsic
   reference frame, rendering biomechanics mathematically invariant to camera pitch/roll.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntrinsicTorsoFrameNormalizer(nn.Module):
    """
    Transforms 3D keypoints into pedestrian-centric intrinsic reference frame.
    Removes camera pitch, roll, vehicle vibration, and road slope effects.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, poses_3d: torch.Tensor) -> dict:
        """
        Args:
            poses_3d: (B, T, 15, 3) in camera/metric coordinates
        Returns:
            dict containing:
                'intrinsic_poses_3d': (B, T, 15, 3) camera-motion-invariant poses
                'torso_rotation': (B, T, 3, 3) rotation matrices
                'pelvis_trajectory': (B, T, 3) global root path
        """
        B, T, J, _ = poses_3d.shape

        # 1. Pelvis Root Centering (Joint 8 = Pelvis)
        pelvis_pos = poses_3d[:, :, 8:9, :] # (B, T, 1, 3)
        rel_poses = poses_3d - pelvis_pos   # (B, T, 15, 3)

        # 2. Compute Orthonormal Torso Basis
        # Joint 1: Neck, Joint 8: Pelvis (Centered at 0)
        neck_pos = rel_poses[:, :, 1, :] # (B, T, 3)
        # Spine vector u_y points from Pelvis to Neck
        u_y = neck_pos / torch.clamp(torch.norm(neck_pos, dim=-1, keepdim=True), min=self.eps)

        # Joint 2: Left Shoulder, Joint 3: Right Shoulder
        l_sh = rel_poses[:, :, 2, :] # (B, T, 3)
        r_sh = rel_poses[:, :, 3, :] # (B, T, 3)
        lateral = l_sh - r_sh        # (B, T, 3)

        # Gram-Schmidt orthogonalization: ensure u_x is orthogonal to u_y
        proj = torch.sum(lateral * u_y, dim=-1, keepdim=True) * u_y
        u_x_raw = lateral - proj
        u_x = u_x_raw / torch.clamp(torch.norm(u_x_raw, dim=-1, keepdim=True), min=self.eps)

        # Forward normal u_z = u_x x u_y (pointing outward from chest)
        u_z = torch.cross(u_x, u_y, dim=-1)
        u_z = u_z / torch.clamp(torch.norm(u_z, dim=-1, keepdim=True), min=self.eps)

        # Construct Rotation Matrix R = [u_x, u_y, u_z]^T: shape (B, T, 3, 3)
        # Row 0: u_x, Row 1: u_y, Row 2: u_z
        R_torso = torch.stack([u_x, u_y, u_z], dim=-2) # (B, T, 3, 3)

        # 3. Rotate relative keypoints into Intrinsic Frame
        # rel_poses: (B, T, 15, 3)
        # intrinsic_poses = rel_poses @ R_torso^T
        intrinsic_poses = torch.einsum("btjd,btkd->btjk", rel_poses, R_torso)

        return {
            "intrinsic_poses_3d": intrinsic_poses,
            "torso_rotation": R_torso,
            "pelvis_trajectory": pelvis_pos.squeeze(2),
        }
