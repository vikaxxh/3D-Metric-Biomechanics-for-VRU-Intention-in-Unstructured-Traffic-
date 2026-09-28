"""
Visual Scene Context Encoder with Controlled Infrastructure Dropout
===================================================================
Encodes visual context (bounding boxes, scene geometry, road semantics).
Supports controlled Lane & Traffic Signal Dropout to benchmark resilience
when road infrastructure disappears in unstructured traffic environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualContextEncoder(nn.Module):
    """
    Encodes pedestrian bounding box dynamics and surrounding road infrastructure cues.
    Provides controllable infrastructure dropout to test the reality gap.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        bbox_dim: int = 4, # [x1, y1, x2, y2]
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_prob = dropout_prob

        # Bounding box trajectory encoder (T, 4)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(bbox_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # Infrastructure semantic cue encoder: [lane_marking_flag, traffic_light_state, road_boundary_dist]
        self.infra_mlp = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        self.temporal_gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=2,
            batch_first=True
        )

        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        bboxes: torch.Tensor,
        infra_cues: torch.Tensor = None,
        force_lane_dropout: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            bboxes: (B, T, 4) bounding box coordinates
            infra_cues: Optional (B, T, 3) [lane_present, signal_state, curb_present]
            force_lane_dropout: if True, completely zeros out infrastructure cues (unstructured emulation)
        Returns:
            z_ctx: (B, embed_dim) Visual context representation
        """
        B, T, _ = bboxes.shape
        device = bboxes.device

        # 1. Bounding box feature extraction
        h_bbox = self.bbox_mlp(bboxes) # (B, T, D)

        # 2. Infrastructure cues
        if infra_cues is None or force_lane_dropout:
            # Completely missing infrastructure (Unstructured India traffic)
            h_infra = torch.zeros_like(h_bbox)
        else:
            # Random dropout during training if specified
            if self.training and self.dropout_prob > 0.0:
                mask = (torch.rand(B, 1, 1, device=device) > self.dropout_prob).float()
                infra_cues = infra_cues * mask
            h_infra = self.infra_mlp(infra_cues)

        h_combined = h_bbox + h_infra # (B, T, D)

        # Temporal aggregation
        _, h_n = self.temporal_gru(h_combined)
        z_ctx = self.out_norm(h_n[-1]) # (B, D)

        return z_ctx
