"""
Cross-Domain Reality Gap Bridge
===============================
Unifies 3D metric biomechanical lifting, domain adversarial alignment,
and multi-modal cross-attention between Waymo LiDAR and IDD-PeD video.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.biolift_former import BioLiftFormer
from models.biomech_backbone import BiomechSTGNNBackbone
from losses.domain_alignment_loss import DomainDiscriminator, MultiKernelMMDLoss


class CrossAttentionFusion(nn.Module):
    """Cross-attention where 3D metric biomechanics queries visual context."""
    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, z_bio: torch.Tensor, z_ctx: torch.Tensor) -> torch.Tensor:
        # z_bio: (B, D) -> (B, 1, D)
        # z_ctx: (B, D) -> (B, 1, D)
        q = z_bio.unsqueeze(1)
        kv = z_ctx.unsqueeze(1)
        norm_q = self.norm1(q)
        attn_out, _ = self.cross_attn(norm_q, kv, kv)
        x = q + attn_out
        x = x + self.mlp(self.norm2(x))
        return x.squeeze(1) # (B, D)


class CrossDomainBridge(nn.Module):
    """
    Coordinates the entire reality gap transfer pipeline.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        use_domain_discriminator: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_disc = use_domain_discriminator

        self.lifter = BioLiftFormer(embed_dim=embed_dim)
        self.biomech_backbone = BiomechSTGNNBackbone(embed_dim=embed_dim)
        self.fusion = CrossAttentionFusion(embed_dim=embed_dim)

        if use_domain_discriminator:
            self.discriminator = DomainDiscriminator(in_features=embed_dim, hidden_dim=64)
            self.mmd_loss = MultiKernelMMDLoss()

    def process_source_waymo(self, poses_3d: torch.Tensor, velocities_3d: torch.Tensor = None) -> torch.Tensor:
        """Processes 3D metric LiDAR keypoints directly through the biomechanical backbone."""
        return self.biomech_backbone(poses_3d, velocities_3d)

    def process_target_idd(self, poses_2d: torch.Tensor) -> tuple:
        """
        Lifts 2D keypoints to 3D metric coordinates, then processes through the biomechanical backbone.
        Returns:
            z_bio: latent biomechanical representation
            lift_outputs: dictionary with 3D predicted poses and root depth
        """
        lift_outputs = self.lifter(poses_2d)
        lifted_3d = lift_outputs["pred_poses_3d_global"]
        z_bio = self.biomech_backbone(lifted_3d)
        return z_bio, lift_outputs
