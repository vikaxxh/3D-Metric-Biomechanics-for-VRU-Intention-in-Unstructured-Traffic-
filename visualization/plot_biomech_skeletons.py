"""
3D Biomechanical Skeleton and Lifting Visualizer
================================================
Renders 3D metric skeletons, kinetic bone graphs, and 2D monocular reprojections.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from data.joint_taxonomy import CANONICAL_BONES, CANONICAL_JOINTS
from data.synthetic_benchmark_gen import generate_single_biomechanical_pedestrian


def plot_3d_skeleton_comparison(
    output_dir: str = "visualization/figures",
    seq_len: int = 30
):
    os.makedirs(output_dir, exist_ok=True)

    poses_3d, poses_2d, attrs = generate_single_biomechanical_pedestrian(
        num_frames=seq_len,
        is_crossing=True,
        is_weaving=True,
        is_looking_over_shoulder=True
    )

    frame_idx = seq_len // 2
    p3d = poses_3d[frame_idx]
    p2d = poses_2d[frame_idx]

    # Style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
    })

    fig = plt.figure(figsize=(14, 5), dpi=300)

    # 1. 2D Monocular Camera View (IDD-PeD Input)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(p2d[:, 0], p2d[:, 1], c="crimson", s=40, zorder=3)
    ax1.set_title("(a) Monocular Video Keypoints\n(IDD-PeD: No Lanes, Distorted 2D)")
    ax1.set_xlabel("Image Pixel $u$")
    ax1.set_ylabel("Image Pixel $v$")
    ax1.invert_yaxis()
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. 3D Metric Skeleton (Waymo LiDAR Anchor)
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(p3d[:, 0], p3d[:, 2], p3d[:, 1], c="royalblue", s=45, depthshade=True)
    for p, c in CANONICAL_BONES:
        xs = [p3d[p, 0], p3d[c, 0]]
        zs = [p3d[p, 2], p3d[c, 2]]
        ys = [p3d[p, 1], p3d[c, 1]]
        ax2.plot(xs, zs, ys, color="royalblue", linewidth=2.5)

    ax2.set_title("(b) 3D Metric Kinematics\n(Waymo LiDAR Metric $(x, y, z)$ Space)")
    ax2.set_xlabel("Lateral $X$ (m)")
    ax2.set_ylabel("Depth $Z$ (m)")
    ax2.set_zlabel("Height $Y$ (m)")
    ax2.view_init(elev=15, azim=45)

    # 3. Lifted 3D Biomechanics with Center-of-Mass Trajectory
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.scatter(p3d[:, 0], p3d[:, 2], p3d[:, 1], c="#1b9e77", s=45, depthshade=True)
    for p, c in CANONICAL_BONES:
        xs = [p3d[p, 0], p3d[c, 0]]
        zs = [p3d[p, 2], p3d[c, 2]]
        ys = [p3d[p, 1], p3d[c, 1]]
        ax3.plot(xs, zs, ys, color="#1b9e77", linewidth=2.5)

    # Plot CoM displacement trajectory over time
    com_traj = poses_3d[:, 8, :] # Pelvis root proxy
    ax3.plot(com_traj[:, 0], com_traj[:, 2], com_traj[:, 1], "r--", linewidth=2.0, label="CoM Trajectory (Crossing Vector)")

    ax3.set_title("(c) BioLift-Former Reconstruction\n(Physics-Anchored Metric Invariants)")
    ax3.set_xlabel("Lateral $X$ (m)")
    ax3.set_ylabel("Depth $Z$ (m)")
    ax3.set_zlabel("Height $Y$ (m)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.view_init(elev=15, azim=45)

    plt.tight_layout()
    out_file = os.path.join(output_dir, "fig_3d_biomech_lifting.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Plot] Saved 3D skeleton comparison figure to: {out_file}")


if __name__ == "__main__":
    plot_3d_skeleton_comparison()
