"""
Stage 1: Waymo 3D Metric Biomechanics Pre-Training
=================================================
Pre-trains the Biomechanical STGNN Backbone on high-precision 3D LiDAR
skeletal sequences from the Waymo Open Dataset.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.biomech_backbone import BiomechSTGNNBackbone
from losses.biomech_physics_loss import BiomechPhysicsLoss
from data.dataset_loader import get_dataloaders
from evaluation.metrics import compute_intention_metrics


def train_waymo_stage1(config_path: str = "configs/waymo_pretrain.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else cfg.get("device", "cpu"))
    os.makedirs("checkpoints", exist_ok=True)

    print(f"===============================================================")
    print(f" [STAGE 1] Pre-training 3D Biomechanics on Waymo Open Dataset")
    print(f" Device: {device} | Epochs: {cfg['training']['epochs']}")
    print(f"===============================================================")

    # Dataloaders
    loaders = get_dataloaders(cfg, batch_size=cfg["training"]["batch_size"])
    train_loader = loaders.get("waymo_train")
    val_loader = loaders.get("waymo_val")

    if train_loader is None or len(train_loader) == 0:
        print("[Error] No Waymo training data found. Please run data generator first.")
        return

    # Model: Backbone + temporary intention classification head
    backbone = BiomechSTGNNBackbone(
        embed_dim=cfg["model"]["embed_dim"],
        num_temporal_layers=cfg["model"]["num_temporal_layers"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    head = nn.Sequential(
        nn.LayerNorm(cfg["model"]["embed_dim"]),
        nn.Linear(cfg["model"]["embed_dim"], 2)
    ).to(device)

    physics_loss_fn = BiomechPhysicsLoss(
        weight_bone=cfg["biomechanics"]["weight_bone"],
        weight_bone_var=cfg["biomechanics"]["weight_bone_var"],
        weight_smooth=cfg["biomechanics"]["weight_smooth"],
        weight_ground=cfg["biomechanics"]["weight_ground"]
    )
    cls_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    best_val_f1 = 0.0
    save_path = cfg["training"]["save_checkpoint"]

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        backbone.train()
        head.train()
        total_loss = 0.0

        for batch in train_loader:
            poses_3d = batch["poses_3d"].to(device) # (B, T, 15, 3)
            velocities = batch["velocities"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            z_bio = backbone(poses_3d, velocities)
            logits = head(z_bio)

            loss_cls = cls_criterion(logits, labels)
            phys_losses = physics_loss_fn(pred_poses_3d=poses_3d)
            loss_phys = phys_losses["loss_bone_prior"] + phys_losses["loss_bone_var"] + phys_losses["loss_smoothness"]

            loss = loss_cls + 0.2 * loss_phys
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        # Validation
        backbone.eval()
        head.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                poses_3d = batch["poses_3d"].to(device)
                velocities = batch["velocities"].to(device)
                labels = batch["label"].to(device)

                z_bio = backbone(poses_3d, velocities)
                logits = head(z_bio)
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

                val_preds.extend(probs)
                val_targets.extend(labels.cpu().numpy())

        metrics = compute_intention_metrics(val_targets, val_preds)
        val_f1 = metrics["f1_macro"]

        print(f"Epoch {epoch:02d}/{cfg['training']['epochs']:02d} | "
              f"Train Loss: {total_loss / len(train_loader):.4f} | "
              f"Val Acc: {metrics['accuracy']:.1f}% | "
              f"Val F1: {metrics['f1_macro']:.1f}% | "
              f"AUC: {metrics['auc_roc']:.3f}")

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "backbone_state": backbone.state_dict(),
                "head_state": head.state_dict(),
                "metrics": metrics,
                "epoch": epoch,
            }, save_path)

    print(f"\n[Stage 1 Complete] Best Pre-trained Waymo Biomechanics checkpoint saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    train_waymo_stage1()
