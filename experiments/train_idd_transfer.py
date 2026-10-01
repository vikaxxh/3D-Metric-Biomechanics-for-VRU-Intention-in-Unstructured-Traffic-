"""
Stage 2: Cross-Domain Reality Gap Transfer & IDD-PeD Adaptation
==============================================================
Transfers the 3D metric biomechanical anchor from Waymo to IDD-PeD:
1. Initializes STGNN backbone with Waymo 3D metric pre-training
2. Trains BioLift-Former to lift monocular 2D keypoints to 3D metric space
3. Applies Physics-Informed losses (rigid bone invariance, kinematics)
4. Domain Alignment via MMD and Domain Adversarial Discriminator
5. Multi-task heads: Binary Intention, Fine-grained Behaviors (CU, CFU, WTT, LOS), TTC
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.vr_intention_classifier import VRUIntentionClassifier
from losses.biomech_physics_loss import BiomechPhysicsLoss
from losses.domain_alignment_loss import MultiKernelMMDLoss, DomainDiscriminator
from data.dataset_loader import get_dataloaders
from evaluation.metrics import compute_intention_metrics, compute_mpjpe, compute_bone_consistency_violation


def train_idd_stage2(config_path: str = "configs/idd_ped_transfer.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else cfg.get("device", "cpu"))
    os.makedirs("checkpoints", exist_ok=True)

    print(f"===============================================================")
    print(f" [STAGE 2] Cross-Domain Reality Gap Transfer to IDD-PeD")
    print(f" Device: {device} | Epochs: {cfg['training']['epochs']}")
    print(f"===============================================================")

    loaders = get_dataloaders(cfg, batch_size=cfg["training"]["batch_size"])
    waymo_train = loaders.get("waymo_train")
    idd_train = loaders.get("idd_train")
    idd_val = loaders.get("idd_val")

    if idd_train is None or len(idd_train) == 0:
        print("[Error] No IDD-PeD training data found.")
        return

    # Model: Full multi-task intention classifier
    model = VRUIntentionClassifier(
        embed_dim=cfg["model"]["embed_dim"],
        num_fine_grained_classes=cfg["model"]["num_fine_grained_classes"],
        dropout_prob=cfg["model"]["dropout"]
    ).to(device)

    # Load Stage 1 Pre-trained Waymo Biomechanics if available
    pretrained_path = cfg["training"].get("pretrained_checkpoint", "checkpoints/waymo_pretrain_best.pt")
    if os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        if "backbone_state" in ckpt:
            model.bridge.biomech_backbone.load_state_dict(ckpt["backbone_state"], strict=False)
            print(f"[Stage 2] Successfully loaded Waymo 3D Biomechanics from: {pretrained_path}")

    # Losses
    physics_loss_fn = BiomechPhysicsLoss(
        weight_bone=0.8,
        weight_bone_var=0.4,
        weight_smooth=0.2,
        weight_ground=0.0 # Deactivated: invalid in torso frame & uncalibrated pitch shocks
    )
    mmd_loss_fn = MultiKernelMMDLoss()
    domain_disc = DomainDiscriminator(in_features=cfg["model"]["embed_dim"]).to(device)

    # Class weighting to balance ~16.5% crossing vs 83.5% non-crossing in IDD-PeD
    cls_weights = torch.tensor([1.0, 4.92], device=device)
    criterion_cls = nn.CrossEntropyLoss(weight=cls_weights)
    criterion_fine = nn.CrossEntropyLoss()
    criterion_ttc = nn.L1Loss()
    criterion_domain = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        list(model.parameters()) + list(domain_disc.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    best_val_f1 = 0.0
    save_path = cfg["training"]["save_checkpoint"]

    waymo_iter = iter(waymo_train) if waymo_train is not None else None

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        domain_disc.train()
        total_loss = 0.0

        for idd_batch in idd_train:
            poses_2d = idd_batch["poses_2d"].to(device) # (B, T, 15, 2)
            bboxes_2d = idd_batch["bboxes_2d"].to(device)
            infra_cues = idd_batch["infra_cues"].to(device)
            labels = idd_batch["label"].to(device)
            fine_labels = idd_batch["fine_label"].to(device)
            ttc_gt = idd_batch["ttc"].to(device)

            optimizer.zero_grad()

            # Forward pass on IDD-PeD
            out = model(
                domain="idd_ped",
                poses_2d=poses_2d,
                bboxes=bboxes_2d,
                infra_cues=infra_cues,
                force_lane_dropout=False
            )

            # 1. Classification & TTC Losses
            l_intent = criterion_cls(out["logits_intention"], labels)
            l_fine = criterion_fine(out["logits_fine_grained"], fine_labels)
            l_ttc = criterion_ttc(out["pred_ttc"], ttc_gt.view(-1))

            # 2. Physics-Informed 3D Biomechanics Lifting Loss
            lifted_3d = out["lift_outputs"]["pred_poses_3d_global"]
            phys_loss = physics_loss_fn(pred_poses_3d=lifted_3d)
            l_phys = phys_loss["loss_bone_prior"] + phys_loss["loss_bone_var"] + phys_loss["loss_smoothness"] + phys_loss["loss_ground"]

            # 3. Domain Adaptation Loss with Waymo (Source Domain)
            l_domain_total = torch.tensor(0.0, device=device)
            if waymo_iter is not None:
                try:
                    w_batch = next(waymo_iter)
                except StopIteration:
                    waymo_iter = iter(waymo_train)
                    w_batch = next(waymo_iter)

                w_poses_3d = w_batch["poses_3d"].to(device)
                w_vel = w_batch["velocities"].to(device)
                z_waymo = model.bridge.process_source_waymo(w_poses_3d, w_vel)
                z_idd = out["z_bio"]

                # MMD Alignment
                l_mmd = mmd_loss_fn(z_waymo, z_idd)

                # Adversarial Domain Classification (Waymo=0, IDD=1)
                disc_w = domain_disc(z_waymo, alpha=0.5)
                disc_i = domain_disc(z_idd, alpha=0.5)
                l_disc = criterion_domain(disc_w, torch.zeros_like(disc_w)) + \
                         criterion_domain(disc_i, torch.ones_like(disc_i))

                l_domain_total = 0.15 * l_mmd + 0.10 * l_disc

            loss = l_intent + 0.3 * l_fine + 0.2 * l_ttc + 0.3 * l_phys + l_domain_total
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        # Validation on IDD-PeD
        model.eval()
        val_preds, val_targets = [], []
        val_lifted_all = []

        with torch.no_grad():
            for batch in idd_val:
                poses_2d = batch["poses_2d"].to(device)
                bboxes_2d = batch["bboxes_2d"].to(device)
                infra_cues = batch["infra_cues"].to(device)
                labels = batch["label"].to(device)

                out = model(
                    domain="idd_ped",
                    poses_2d=poses_2d,
                    bboxes=bboxes_2d,
                    infra_cues=infra_cues,
                    force_lane_dropout=False
                )

                probs = out["probs_intention"][:, 1].cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(labels.cpu().numpy())

                lifted_np = out["lift_outputs"]["pred_poses_3d_global"].cpu().numpy()
                val_lifted_all.append(lifted_np)

        metrics = compute_intention_metrics(val_targets, val_preds)
        val_f1 = metrics["f1_macro"]

        all_lifted_np = np.concatenate(val_lifted_all, axis=0) if val_lifted_all else np.zeros((1, 15, 3))
        bone_viol_pct = compute_bone_consistency_violation(all_lifted_np)

        print(f"Epoch {epoch:02d}/{cfg['training']['epochs']:02d} | "
              f"Loss: {total_loss / len(idd_train):.4f} | "
              f"IDD Val Acc: {metrics['accuracy']:.1f}% | "
              f"Val F1: {metrics['f1_macro']:.1f}% | "
              f"AUC: {metrics['auc_roc']:.3f} | "
              f"Bone Viol: {bone_viol_pct:.1f}%")

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "model_state": model.state_dict(),
                "domain_disc_state": domain_disc.state_dict(),
                "metrics": metrics,
                "epoch": epoch,
            }, save_path)

    print(f"\n[Stage 2 Complete] Best IDD-PeD Transfer Model saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    train_idd_stage2()
