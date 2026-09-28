"""
Bio-STGNN: Biomechanical Spatio-Temporal Graph Transformer Backbone
===================================================================
Processes 3D metric kinematic streams:
- Metric Joint Positions (15 joints)
- Metric Joint Velocities (m/s)
- Anthropometric Bone Vectors (14 bones)
- Center of Mass (CoM) Kinematics and Kinetic Energy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.joint_taxonomy import get_adjacency_matrix, NUM_CANONICAL_JOINTS


class SpatialGraphConv(nn.Module):
    """Kinematic Graph Convolution layer along skeletal bones."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.GELU()

        adj = get_adjacency_matrix()
        self.register_buffer("adj", torch.from_numpy(adj))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, J, in_features)
        # linear: (B, T, J, out_features)
        h = self.linear(x)
        # Graph convolution: A * H along J dimension
        # adj: (J, J)
        out = torch.einsum("vw,btwd->btvd", self.adj, h)
        return self.act(self.norm(out))


class BiomechSTGNNBackbone(nn.Module):
    """
    Spatio-Temporal Biomechanics Backbone.
    Extracts physically grounded invariant representations of pedestrian locomotion.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_temporal_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Input feature dimensions per joint: position (3) + velocity (3) = 6
        self.joint_stem = nn.Sequential(
            nn.Linear(6, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # Spatial Graph Convolutions
        self.sgc1 = SpatialGraphConv(embed_dim, embed_dim)
        self.sgc2 = SpatialGraphConv(embed_dim, embed_dim)

        # Whole-body Center of Mass & Kinetic Energy encoder
        self.com_encoder = nn.Sequential(
            nn.Linear(4, embed_dim // 2), # com_vel (3) + kinetic_energy (1)
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_temporal_layers)

        # Learnable temporal CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, 128, embed_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        self.proj_out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(
        self,
        positions_3d: torch.Tensor,
        velocities_3d: torch.Tensor = None,
        com_velocity: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            positions_3d: (B, T, 15, 3) in meters
            velocities_3d: Optional (B, T, 15, 3) in m/s (auto-computed if None)
            com_velocity: Optional (B, T, 3)
        Returns:
            z_bio: (B, embed_dim) Biomechanical feature vector
        """
        B, T, J, _ = positions_3d.shape

        if velocities_3d is None:
            # Approximate velocity from positions
            dt = 0.1 # 10 fps
            velocities_3d = torch.zeros_like(positions_3d)
            if T > 1:
                velocities_3d[:, 1:] = (positions_3d[:, 1:] - positions_3d[:, :-1]) / dt
                velocities_3d[:, 0] = velocities_3d[:, 1]

        # 1. Concatenate Joint pos + vel
        joint_input = torch.cat([positions_3d, velocities_3d], dim=-1) # (B, T, J, 6)
        h = self.joint_stem(joint_input) # (B, T, J, D)

        # 2. Spatial Graph Convolutions
        h = self.sgc1(h)
        h = self.sgc2(h)

        # Pool across joints: (B, T, D)
        h_spatial = h.mean(dim=2)

        # 3. Whole-body CoM Dynamics
        if com_velocity is None:
            # Pelvis velocity proxy
            com_vel = velocities_3d[:, :, 8, :] # (B, T, 3)
        else:
            com_vel = com_velocity

        ke = 0.5 * torch.sum(velocities_3d ** 2, dim=(-1, -2)).unsqueeze(-1) # (B, T, 1)
        com_feat = torch.cat([com_vel, ke], dim=-1) # (B, T, 4)
        h_com = self.com_encoder(com_feat) # (B, T, D)

        # Fuse spatial joints + CoM
        h_seq = h_spatial + h_com # (B, T, D)

        # 4. Temporal Transformer with CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        h_seq = torch.cat([cls_tokens, h_seq], dim=1) # (B, T+1, D)

        # Add position embedding
        seq_len = h_seq.shape[1]
        h_seq = h_seq + self.temporal_pos_embed[:, :seq_len, :]

        h_trans = self.temporal_transformer(h_seq)
        cls_out = h_trans[:, 0, :] # Extract CLS token: (B, D)

        z_bio = self.proj_out(cls_out)
        return z_bio
