"""
BioLift-Former: 2D-to-3D Metric Biomechanical Lifting Transformer
================================================================
Lifts 2D monocular pose sequences (from IDD-PeD / MMPose) into metric 3D coordinates
(x, y, z) in meters, anchored by human anthropometric bone constraints.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.joint_taxonomy import NUM_CANONICAL_JOINTS, CANONICAL_BONES, ANTHROPOMETRIC_BONE_LENGTHS


class SpatialJointEmbedding(nn.Module):
    """Embeds 2D joint coordinates + joint ID embeddings into latent space."""
    def __init__(self, in_channels: int = 2, embed_dim: int = 128, num_joints: int = 15):
        super().__init__()
        self.coord_proj = nn.Linear(in_channels, embed_dim)
        self.joint_id_embed = nn.Embedding(num_joints, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, J, 2)
        B, T, J, _ = x.shape
        coord_feat = self.coord_proj(x) # (B, T, J, embed_dim)
        joint_ids = torch.arange(J, device=x.device).unsqueeze(0).unsqueeze(0) # (1, 1, J)
        joint_embed = self.joint_id_embed(joint_ids) # (1, 1, J, embed_dim)
        return self.norm(coord_feat + joint_embed)


class TemporalTransformerBlock(nn.Module):
    """Multi-Head Self-Attention across temporal frames for each joint."""
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B * J, T, embed_dim)
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class BioLiftFormer(nn.Module):
    """
    2D to 3D Biomechanical Lifting Network.
    Predicts 3D coordinates relative to pelvis root, plus root depth estimation.
    """
    def __init__(
        self,
        num_joints: int = 15,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim

        self.joint_embed = SpatialJointEmbedding(in_channels=2, embed_dim=embed_dim, num_joints=num_joints)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, 128, 1, embed_dim)) # max 128 frames
        nn.init.trunc_normal_(self.pos_embed_t, std=0.02)

        self.temporal_layers = nn.ModuleList([
            TemporalTransformerBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Inter-joint kinematic graph message passing
        self.joint_interaction = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 3D Coordinate Output Head: (x, y, z) in meters relative to pelvis
        self.head_rel_3d = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 3)
        )

        # Root (Pelvis) Metric Depth Estimator
        self.head_root_depth = nn.Sequential(
            nn.Linear(embed_dim * num_joints, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3) # (X_root, Y_root, Z_root)
        )

    def forward(self, kpts_2d: torch.Tensor) -> dict:
        """
        Args:
            kpts_2d: shape (B, T, 15, 2) normalized or pixel coordinates
        Returns:
            dict containing:
                'pred_poses_3d': (B, T, 15, 3) relative metric poses
                'pred_root_3d': (B, T, 3) root translation
                'pred_poses_3d_global': (B, T, 15, 3) global metric coordinates
        """
        B, T, J, _ = kpts_2d.shape

        # Normalize 2D input around pelvis (joint 8)
        pelvis_2d = kpts_2d[:, :, 8:9, :]
        kpts_2d_centered = kpts_2d - pelvis_2d

        # 1. Joint Embedding
        h = self.joint_embed(kpts_2d_centered) # (B, T, J, D)

        # 2. Add temporal position embedding
        if T <= self.pos_embed_t.shape[1]:
            h = h + self.pos_embed_t[:, :T, :, :]
        else:
            # Interpolate if longer
            pos_interp = F.interpolate(
                self.pos_embed_t.permute(0, 3, 1, 2),
                size=(T, 1),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
            h = h + pos_interp

        # 3. Temporal Self-Attention per joint
        # Reshape to (B * J, T, D)
        h_t = h.permute(0, 2, 1, 3).reshape(B * J, T, self.embed_dim)
        for layer in self.temporal_layers:
            h_t = layer(h_t)
        h = h_t.reshape(B, J, T, self.embed_dim).permute(0, 2, 1, 3) # (B, T, J, D)

        # 4. Joint interaction
        h = h + self.joint_interaction(h)

        # 5. Predict relative 3D coordinates (meters)
        rel_3d = self.head_rel_3d(h) # (B, T, J, 3)
        # Ensure root pelvis (joint 8) is at origin [0, 0, 0]
        rel_3d = rel_3d - rel_3d[:, :, 8:9, :]

        # 6. Predict root position
        h_pool = h.reshape(B, T, J * self.embed_dim)
        root_3d = self.head_root_depth(h_pool) # (B, T, 3)

        global_3d = rel_3d + root_3d.unsqueeze(2)

        return {
            "pred_poses_3d": rel_3d,
            "pred_root_3d": root_3d,
            "pred_poses_3d_global": global_3d,
        }
