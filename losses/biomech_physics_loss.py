"""
Physics-Informed Biomechanical Loss Functions
============================================
Enforces physical and anatomical constraints on 3D lifted keypoints:
1. Bone length anthropometric prior loss
2. Temporal bone length consistency (rigid body invariance)
3. Kinematic smoothness / bounded acceleration loss
4. Ground plane non-penetration penalty
5. Differentiable 2D reprojection loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.joint_taxonomy import CANONICAL_BONES, ANTHROPOMETRIC_BONE_LENGTHS


class BiomechPhysicsLoss(nn.Module):
    def __init__(
        self,
        weight_3d: float = 1.0,
        weight_bone: float = 0.5,
        weight_bone_var: float = 0.3,
        weight_smooth: float = 0.1,
        weight_ground: float = 0.2,
        weight_reproj: float = 0.5,
    ):
        super().__init__()
        self.w_3d = weight_3d
        self.w_bone = weight_bone
        self.w_bone_var = weight_bone_var
        self.w_smooth = weight_smooth
        self.w_ground = weight_ground
        self.w_reproj = weight_reproj

        # Precompute target bone lengths tensor
        self.target_lengths = []
        for p, c in CANONICAL_BONES:
            self.target_lengths.append(ANTHROPOMETRIC_BONE_LENGTHS[(p, c)])

    def forward(
        self,
        pred_poses_3d: torch.Tensor,
        gt_poses_3d: torch.Tensor = None,
        gt_poses_2d: torch.Tensor = None,
        camera_k: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            pred_poses_3d: (B, T, 15, 3) predicted 3D keypoints in meters
            gt_poses_3d: Optional (B, T, 15, 3) ground truth 3D keypoints (from Waymo)
            gt_poses_2d: Optional (B, T, 15, 2) input 2D keypoints
            camera_k: Optional (B, 3, 3) camera intrinsic matrix
        """
        device = pred_poses_3d.device
        dtype = pred_poses_3d.dtype
        target_lens = torch.tensor(self.target_lengths, device=device, dtype=dtype).view(1, 1, -1) # (1, 1, B)

        losses = {}
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)

        # 1. Supervised 3D Loss (if GT available)
        if gt_poses_3d is not None:
            l_3d = F.smooth_l1_loss(pred_poses_3d, gt_poses_3d)
            losses["loss_3d"] = l_3d
            total_loss = total_loss + self.w_3d * l_3d

        # 2. Compute current bone lengths: shape (B, T, num_bones)
        bone_lengths = []
        for p, c in CANONICAL_BONES:
            diff = pred_poses_3d[:, :, c, :] - pred_poses_3d[:, :, p, :]
            dist = torch.norm(diff, p=2, dim=-1) # (B, T)
            bone_lengths.append(dist)
        curr_bone_lens = torch.stack(bone_lengths, dim=-1) # (B, T, num_bones)

        # Anthropometric bone prior loss
        l_bone = F.mse_loss(curr_bone_lens, target_lens.expand_as(curr_bone_lens))
        losses["loss_bone_prior"] = l_bone
        total_loss = total_loss + self.w_bone * l_bone

        # Temporal rigid bone length variance (a limb must not stretch over time)
        bone_temporal_var = torch.var(curr_bone_lens, dim=1).mean()
        losses["loss_bone_var"] = bone_temporal_var
        total_loss = total_loss + self.w_bone_var * bone_temporal_var

        # 3. Kinematic smoothness (second difference acceleration penalty)
        if pred_poses_3d.shape[1] > 2:
            accel = pred_poses_3d[:, 2:, :, :] - 2 * pred_poses_3d[:, 1:-1, :, :] + pred_poses_3d[:, :-2, :, :]
            l_smooth = torch.mean(torch.sum(accel ** 2, dim=-1))
            losses["loss_smoothness"] = l_smooth
            total_loss = total_loss + self.w_smooth * l_smooth

        # 4. Ground plane non-penetration (Ankles joint 13 and 14: Y must be >= 0)
        l_ankle_y = pred_poses_3d[:, :, 13, 1]
        r_ankle_y = pred_poses_3d[:, :, 14, 1]
        ground_pen = F.relu(-l_ankle_y).pow(2).mean() + F.relu(-r_ankle_y).pow(2).mean()
        losses["loss_ground"] = ground_pen
        total_loss = total_loss + self.w_ground * ground_pen

        # 5. Differentiable 2D Reprojection Loss
        if gt_poses_2d is not None and camera_k is not None:
            # Perspective projection: u = fx * X / Z + cx, v = -fy * Y / Z + cy
            fx = camera_k[:, 0, 0].view(-1, 1, 1)
            fy = camera_k[:, 1, 1].view(-1, 1, 1)
            cx = camera_k[:, 0, 2].view(-1, 1, 1)
            cy = camera_k[:, 1, 2].view(-1, 1, 1)

            Z = torch.clamp(pred_poses_3d[:, :, :, 2], min=0.5)
            u_proj = fx * (pred_poses_3d[:, :, :, 0] / Z) + cx
            v_proj = -fy * (pred_poses_3d[:, :, :, 1] / Z) + cy
            pred_2d = torch.stack([u_proj, v_proj], dim=-1)

            l_reproj = F.l1_loss(pred_2d, gt_poses_2d)
            losses["loss_reproj"] = l_reproj
            total_loss = total_loss + self.w_reproj * l_reproj

        losses["loss_total"] = total_loss
        return losses
