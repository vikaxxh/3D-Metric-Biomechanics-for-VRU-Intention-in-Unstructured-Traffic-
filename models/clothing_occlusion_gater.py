"""
Hierarchical Clothing & Occlusion Gating (Upper vs. Lower Body Decoupling)
========================================================================
Addresses Reviewer Critique on Traditional Indian Attire (Sarees, Kurtas, Dhotis):
In unstructured traffic, lower limbs (knees, ankles) are frequently hidden under
flowing traditional clothing or occluded by crawling two-wheelers.

Biomechanical Grounding:
Before stepping forward into traffic, human locomotive neuroscience dictates that:
1. Head Yaw (gaze orienting towards traffic flow) occurs 350-500ms before step onset.
2. Torso Pitch (forward gravitational CoM shift) occurs 200-350ms before step onset.
3. Lower limb stride follows afterwards.

This module dynamically gates kinematic attention between upper and lower body
sub-graphs based on joint visibility/confidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Joint Indices
# Upper: 0 (Head), 1 (Neck), 2 (L-Sh), 3 (R-Sh), 4 (L-Elb), 5 (R-Elb), 6 (L-Wri), 7 (R-Wri)
UPPER_BODY_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7]
# Lower: 8 (Pelvis), 9 (L-Hip), 10 (R-Hip), 11 (L-Knee), 12 (R-Knee), 13 (L-Ank), 14 (R-Ank)
LOWER_BODY_JOINTS = [8, 9, 10, 11, 12, 13, 14]


class ClothingOcclusionAwareGater(nn.Module):
    """
    Decoupled Upper-Lower Kinematic Stream with Dynamic Attire/Occlusion Gating.
    """
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # Upper body encoder (Head, Neck, Torso, Arms)
        self.upper_encoder = nn.Sequential(
            nn.Linear(len(UPPER_BODY_JOINTS) * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Lower body encoder (Pelvis, Hips, Legs, Stride)
        self.lower_encoder = nn.Sequential(
            nn.Linear(len(LOWER_BODY_JOINTS) * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Attire & Occlusion Gating Network
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        intrinsic_poses_3d: torch.Tensor,
        confidence_scores: torch.Tensor = None, # (B, T, 15) optional
        simulate_clothing_occlusion: bool = False,
    ) -> dict:
        """
        Args:
            intrinsic_poses_3d: (B, T, 15, 3) in intrinsic pedestrian frame
            confidence_scores: Optional (B, T, 15) joint detector confidence
            simulate_clothing_occlusion: if True, masks out lower body to test clothing invariance
        Returns:
            dict containing:
                'z_kinematic_fused': (B, T, embed_dim)
                'upper_weight': (B, T, 1) reliance on upper body
                'z_upper': upper body features
                'z_lower': lower body features
        """
        B, T, J, D = intrinsic_poses_3d.shape
        device = intrinsic_poses_3d.device

        # 1. Extract Upper & Lower Body Joint Subsets
        upper_joints = intrinsic_poses_3d[:, :, UPPER_BODY_JOINTS, :].reshape(B, T, -1)
        lower_joints = intrinsic_poses_3d[:, :, LOWER_BODY_JOINTS, :].reshape(B, T, -1)

        # If clothing occlusion occurs or is simulated, mask lower body
        if simulate_clothing_occlusion:
            lower_joints = torch.zeros_like(lower_joints)
        elif confidence_scores is not None:
            lower_conf = confidence_scores[:, :, LOWER_BODY_JOINTS].mean(dim=-1, keepdim=True) # (B, T, 1)
            lower_joints = lower_joints * lower_conf

        h_upper = self.upper_encoder(upper_joints) # (B, T, D)
        h_lower = self.lower_encoder(lower_joints) # (B, T, D)

        # 2. Dynamic Gating: Upper vs Lower Weight
        gate_input = torch.cat([h_upper, h_lower], dim=-1)
        w_upper = self.gate_mlp(gate_input) # in [0, 1]

        if simulate_clothing_occlusion:
            # Force high reliance on upper body
            w_upper = torch.clamp(w_upper + 0.4, max=0.95)

        # Fused kinematic representation
        h_fused = w_upper * h_upper + (1.0 - w_upper) * h_lower
        h_fused = self.out_norm(h_fused)

        return {
            "z_kinematic_fused": h_fused,
            "upper_weight": w_upper,
            "z_upper": h_upper,
            "z_lower": h_lower,
        }
