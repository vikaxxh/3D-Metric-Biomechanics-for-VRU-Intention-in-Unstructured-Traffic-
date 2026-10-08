"""
Ego-Motion & Road Roughness Invariance Module (Revised)
=====================================================
Addresses Committee Review:
1. Avoids the "Zero Trunk Pitch" paradox:
   While joint articulation is normalized into the intrinsic torso basis [u_x, u_y, u_z]
   to isolate limb articulation from camera rotation, we explicitly PRESERVE the torso
   inclination angles relative to the vertical axis (gravity proxy):
     - Trunk Forward Lean / Pitch: theta_pitch = arcsin(u_{y, z})  <-- Captures APA early cue!
     - Lateral Torso Roll: theta_roll = arcsin(u_{y, x})
     - Forward Torso Heading Vector: u_z
   These orientation features are output alongside the intrinsic skeleton so that
   Anticipatory Postural Adjustments (APA) remain fully available to the intention classifier!
2. Clarifies geometric scope:
   Invariance holds for the 3D coordinate basis; upstream 2D perspective distortions
   are handled via camera homography robustness testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntrinsicTorsoFrameNormalizer(nn.Module):
    """
    Decouples pedestrian limb articulation from global camera coordinate rotations
    while explicitly retaining trunk forward pitch (APA) and torso orientation.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, poses_3d: torch.Tensor) -> dict:
        """
        Args:
            poses_3d: (B, T, 15, 3) in camera/metric coordinates
                      Assumes camera coordinates with Z = depth, Y = vertical, X = lateral
        Returns:
            dict containing:
                'intrinsic_poses_3d': (B, T, 15, 3) intrinsic relative limb articulation
                'torso_orientation_features': (B, T, 6) explicit trunk posture:
                    [theta_pitch, theta_roll, u_y (3), u_z (1)]
                'torso_rotation': (B, T, 3, 3) rotation matrices
                'pelvis_trajectory': (B, T, 3) root translation path
        """
        B, T, J, _ = poses_3d.shape
        device = poses_3d.device

        # 1. Pelvis Root Centering (Joint 8 = Pelvis)
        pelvis_pos = poses_3d[:, :, 8:9, :] # (B, T, 1, 3)
        rel_poses = poses_3d - pelvis_pos   # (B, T, 15, 3)

        # 2. Compute Orthonormal Torso Basis
        # Joint 1: Neck, Joint 8: Pelvis (Centered at origin)
        neck_pos = rel_poses[:, :, 1, :] # (B, T, 3)
        spine_vec = neck_pos # vector from pelvis to neck
        spine_norm = torch.clamp(torch.norm(spine_vec, dim=-1, keepdim=True), min=self.eps)
        u_y = spine_vec / spine_norm # (B, T, 3): unit spine axis

        # 3. EXPLICIT TRUNK POSTURE (APA Lean) EXTRACTION BEFORE NORMALIZATION
        # In camera coordinates, depth is Z, vertical is Y, lateral is X.
        # Forward trunk lean (pitch toward street / depth):
        theta_pitch = torch.asin(torch.clamp(u_y[..., 2:3], -0.99, 0.99)) # (B, T, 1)
        # Lateral trunk tilt (roll):
        theta_roll = torch.asin(torch.clamp(u_y[..., 0:1], -0.99, 0.99))  # (B, T, 1)

        # Joint 2: Left Shoulder, Joint 3: Right Shoulder
        l_sh = rel_poses[:, :, 2, :] # (B, T, 3)
        r_sh = rel_poses[:, :, 3, :] # (B, T, 3)
        lateral = l_sh - r_sh        # (B, T, 3)

        # Gram-Schmidt orthogonalization: ensure u_x is orthogonal to u_y
        proj = torch.sum(lateral * u_y, dim=-1, keepdim=True) * u_y
        u_x_raw = lateral - proj
        u_x = u_x_raw / torch.clamp(torch.norm(u_x_raw, dim=-1, keepdim=True), min=self.eps)

        # Forward chest normal u_z = u_x x u_y
        u_z = torch.cross(u_x, u_y, dim=-1)
        u_z = u_z / torch.clamp(torch.norm(u_z, dim=-1, keepdim=True), min=self.eps)

        # Construct Rotation Matrix R = [u_x, u_y, u_z]^T: shape (B, T, 3, 3)
        R_torso = torch.stack([u_x, u_y, u_z], dim=-2) # (B, T, 3, 3)

        # 4. Intrinsic Relative Articulation (Limb flexion, knee angles)
        intrinsic_poses = torch.einsum("btjd,btkd->btjk", rel_poses, R_torso)

        # 5. Pack Torso Orientation Features to preserve APA signal
        # [theta_pitch (1), theta_roll (1), u_y (3), chest_forward_z (1)] = 6 dims
        torso_orientation_features = torch.cat([
            theta_pitch,
            theta_roll,
            u_y,
            u_z[..., 2:3]
        ], dim=-1) # (B, T, 6)

        return {
            "intrinsic_poses_3d": intrinsic_poses,
            "torso_orientation_features": torso_orientation_features,
            "torso_rotation": R_torso,
            "pelvis_trajectory": pelvis_pos.squeeze(2),
        }
