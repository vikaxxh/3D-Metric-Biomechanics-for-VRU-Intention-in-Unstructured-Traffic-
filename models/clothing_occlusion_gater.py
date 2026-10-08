"""
Hierarchical Clothing & Occlusion Gating (Upper vs. Lower Body Decoupling) - Revised
===================================================================================
Addresses Committee Review:
1. Gater now EXPLICITLY ingests detection confidence scores:
   Input: lower_confidence = c_lower in [0, 1].
   Formula: alpha_upper = Sigmoid( W_gate * [h_upper; h_lower; 1.0 - c_lower] )
   The gating network is explicitly conditioned on detection quality / missingness!
2. Ingests explicit trunk posture (APA lean):
   Upper body stream incorporates trunk pitch theta_pitch and orientation features.
3. Supports training-time Lower-Limb Dropout:
   During training, lower limbs are randomly masked with probability p_drop to force
   the gate to dynamically route attention to the upper-body kinetic chain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Joint Indices
UPPER_BODY_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7]
LOWER_BODY_JOINTS = [8, 9, 10, 11, 12, 13, 14]


class ClothingOcclusionAwareGater(nn.Module):
    """
    Decoupled Upper-Lower Kinematic Stream with Confidence-Conditioned Attire Gating.
    """
    def __init__(self, embed_dim: int = 128, trunk_posture_dim: int = 6):
        super().__init__()
        self.embed_dim = embed_dim

        # Upper body encoder: Upper joints (8 * 3 = 24) + Trunk posture (6) = 30
        self.upper_encoder = nn.Sequential(
            nn.Linear(len(UPPER_BODY_JOINTS) * 3 + trunk_posture_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Lower body encoder: Lower joints (7 * 3 = 21)
        self.lower_encoder = nn.Sequential(
            nn.Linear(len(LOWER_BODY_JOINTS) * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Confidence-Conditioned Gating MLP
        # Ingests [h_upper (embed_dim), h_lower (embed_dim), occlusion_signal (1)]
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        intrinsic_poses_3d: torch.Tensor,
        torso_orientation_features: torch.Tensor = None,
        confidence_scores: torch.Tensor = None, # (B, T, 15) joint confidence
        p_train_mask_lower: float = 0.0,        # Training-time lower limb dropout
        simulate_clothing_occlusion: bool = False,
    ) -> dict:
        """
        Args:
            intrinsic_poses_3d: (B, T, 15, 3) intrinsic relative limb articulation
            torso_orientation_features: Optional (B, T, 6) [theta_pitch, theta_roll, u_y, u_z]
            confidence_scores: Optional (B, T, 15) detection confidences from MMPose
            p_train_mask_lower: float probability of dropping lower limbs during training
            simulate_clothing_occlusion: if True, sets lower confidence to 0 (saree drape)
        Returns:
            dict containing:
                'z_kinematic_fused': (B, T, embed_dim)
                'upper_weight': (B, T, 1) alpha_upper
                'lower_confidence': (B, T, 1) computed lower body visibility
        """
        B, T, J, D = intrinsic_poses_3d.shape
        device = intrinsic_poses_3d.device

        # 1. Extract Upper & Lower Joint Coordinates
        upper_joints = intrinsic_poses_3d[:, :, UPPER_BODY_JOINTS, :].reshape(B, T, -1) # (B, T, 24)
        lower_joints = intrinsic_poses_3d[:, :, LOWER_BODY_JOINTS, :].reshape(B, T, -1) # (B, T, 21)

        # 2. Append Trunk Posture (APA lean) to Upper Body Stream
        if torso_orientation_features is not None:
            upper_input = torch.cat([upper_joints, torso_orientation_features], dim=-1) # (B, T, 30)
        else:
            dummy_posture = torch.zeros(B, T, 6, device=device)
            upper_input = torch.cat([upper_joints, dummy_posture], dim=-1)

        # 3. Determine Lower Body Occlusion / Confidence Metric
        if simulate_clothing_occlusion:
            lower_conf = torch.zeros(B, T, 1, device=device)
            lower_joints = torch.zeros_like(lower_joints)
        elif confidence_scores is not None:
            # Average confidence over lower body joints (hips, knees, ankles)
            lower_conf = confidence_scores[:, :, LOWER_BODY_JOINTS].mean(dim=-1, keepdim=True) # (B, T, 1)
            lower_joints = lower_joints * lower_conf
        else:
            lower_conf = torch.ones(B, T, 1, device=device)

        # Training-time lower limb masking
        if self.training and p_train_mask_lower > 0.0:
            mask = (torch.rand(B, 1, 1, device=device) > p_train_mask_lower).float()
            lower_joints = lower_joints * mask
            lower_conf = lower_conf * mask

        # 4. Feature Encoding
        h_upper = self.upper_encoder(upper_input) # (B, T, D)
        h_lower = self.lower_encoder(lower_joints) # (B, T, D)

        # 5. Gating: alpha_upper explicitly conditioned on [h_upper; h_lower; (1 - lower_conf)]
        occlusion_signal = 1.0 - lower_conf # 0 = fully visible, 1 = fully occluded
        gate_input = torch.cat([h_upper, h_lower, occlusion_signal], dim=-1) # (B, T, 2D + 1)
        alpha_upper = self.gate_mlp(gate_input) # (B, T, 1)

        # Fused kinematic representation
        h_fused = alpha_upper * h_upper + (1.0 - alpha_upper) * h_lower
        h_fused = self.out_norm(h_fused)

        return {
            "z_kinematic_fused": h_fused,
            "upper_weight": alpha_upper,
            "lower_confidence": lower_conf,
            "z_upper": h_upper,
            "z_lower": h_lower,
        }
