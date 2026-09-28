"""
VRU Intention and Biomechanical Behavior Classifier
===================================================
Multi-task prediction network:
1. Binary Crossing Intention (Crossing vs. Not Crossing)
2. Fine-Grained Behavioral Attributes (CU, CFU, WTT, LOS)
3. Time-to-Cross (TTC) Horizon Regression (seconds)
4. Dynamic Biomechanical Reliability Gating (Physical Anchor vs Context)
5. True Architectural Ablation Switches:
   - use_intrinsic_torso: Gram-Schmidt torso basis vs raw camera coordinates
   - use_clothing_gater: Dynamic upper/lower gating vs unweighted static baseline
   - disable_biomechanics: Visual Context Only ablation
   - disable_visual_context: Pose/Biomechanics Only ablation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cross_domain_bridge import CrossDomainBridge
from models.context_encoder import VisualContextEncoder
from models.ego_motion_compensator import IntrinsicTorsoFrameNormalizer
from models.clothing_occlusion_gater import ClothingOcclusionAwareGater


class VRUIntentionClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_fine_grained_classes: int = 5, # 0: None, 1: CU, 2: CFU, 3: WTT, 4: LOS
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.bridge = CrossDomainBridge(embed_dim=embed_dim)
        self.context_encoder = VisualContextEncoder(embed_dim=embed_dim)
        self.intrinsic_norm = IntrinsicTorsoFrameNormalizer()
        self.clothing_gater = ClothingOcclusionAwareGater(embed_dim=embed_dim)

        # Dynamic Reliability Gating Network: computes alpha in [0, 1]
        # alpha -> 1: relies on 3D Biomechanics anchor; alpha -> 0: relies on visual context
        self.reliability_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        # Fusion projection
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # 1. Primary Binary Intention Head
        self.head_intention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(embed_dim // 2, 2)
        )

        # 2. Fine-Grained Unstructured Behavior Head
        self.head_fine_grained = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(embed_dim // 2, num_fine_grained_classes)
        )

        # 3. Time-to-Cross (TTC) Regression Head (in seconds)
        self.head_ttc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.ReLU() # TTC is strictly non-negative
        )

    def forward(
        self,
        domain: str = "idd_ped", # "waymo" or "idd_ped"
        poses_3d: torch.Tensor = None,
        poses_2d: torch.Tensor = None,
        bboxes: torch.Tensor = None,
        infra_cues: torch.Tensor = None,
        force_lane_dropout: bool = False,
        use_intrinsic_torso: bool = True,
        use_clothing_gater: bool = True,
        disable_biomechanics: bool = False,
        disable_visual_context: bool = False,
    ) -> dict:
        """
        Multi-stream forward pass with explicit ablation controls.
        """
        lift_outputs = None
        gate_alpha_upper = None

        # 1. Extract Biomechanical Stream
        if domain == "waymo":
            if use_intrinsic_torso and poses_3d is not None:
                norm_out = self.intrinsic_norm(poses_3d)
                poses_in = norm_out["intrinsic_poses_3d"]
            else:
                poses_in = poses_3d
            z_bio = self.bridge.process_source_waymo(poses_in)
        else:
            # IDD-PeD: Lift 2D -> 3D
            lift_outputs = self.bridge.lifter(poses_2d)
            lifted_3d = lift_outputs["pred_poses_3d_global"]

            if use_intrinsic_torso:
                norm_out = self.intrinsic_norm(lifted_3d)
                poses_in = norm_out["intrinsic_poses_3d"]
            else:
                # Ablation: raw camera-frame (vulnerable to pitch / camera shake)
                pelvis_pos = lifted_3d[:, :, 8:9, :]
                poses_in = lifted_3d - pelvis_pos

            z_backbone = self.bridge.biomech_backbone(poses_in)

            if use_clothing_gater:
                gate_out = self.clothing_gater(poses_in)
                z_bio = 0.5 * z_backbone + 0.5 * gate_out["z_kinematic_fused"]
                gate_alpha_upper = gate_out["upper_weight"]
            else:
                # Ablation: static unweighted average without dynamic gating
                z_upper = self.clothing_gater.upper_encoder(poses_in[:, :, :8, :].reshape(poses_in.size(0), poses_in.size(1), -1))
                z_lower = self.clothing_gater.lower_encoder(poses_in[:, :, 8:, :].reshape(poses_in.size(0), poses_in.size(1), -1))
                z_unweighted = 0.5 * (z_upper + z_lower).mean(dim=1)
                z_bio = 0.5 * z_backbone + 0.5 * z_unweighted
                gate_alpha_upper = torch.full((poses_in.size(0), poses_in.size(1), 1), 0.5, device=poses_in.device)

        # 2. Extract Visual Context Stream
        if bboxes is not None and not disable_visual_context:
            z_ctx = self.context_encoder(bboxes, infra_cues, force_lane_dropout=force_lane_dropout)
        else:
            z_ctx = torch.zeros_like(z_bio)

        # 3. Dynamic Reliability Gating
        if disable_biomechanics:
            alpha = torch.zeros((z_bio.size(0), 1), device=z_bio.device)
            z_fused = z_ctx
        elif disable_visual_context or bboxes is None:
            alpha = torch.ones((z_bio.size(0), 1), device=z_bio.device)
            z_fused = z_bio
        else:
            gate_input = torch.cat([z_bio, z_ctx], dim=-1)
            alpha = self.reliability_gate(gate_input)
            z_fused = alpha * z_bio + (1.0 - alpha) * z_ctx

        z_fused = self.fusion_norm(z_fused)

        # 4. Predictions
        logits_intention = self.head_intention(z_fused)
        logits_fine = self.head_fine_grained(z_fused)
        pred_ttc = self.head_ttc(z_fused).squeeze(-1)

        return {
            "logits_intention": logits_intention,
            "probs_intention": F.softmax(logits_intention, dim=-1),
            "logits_fine_grained": logits_fine,
            "pred_ttc": pred_ttc,
            "reliability_alpha": alpha,
            "gate_alpha_upper": gate_alpha_upper,
            "z_bio": z_bio,
            "z_ctx": z_ctx,
            "z_fused": z_fused,
            "lift_outputs": lift_outputs,
        }
