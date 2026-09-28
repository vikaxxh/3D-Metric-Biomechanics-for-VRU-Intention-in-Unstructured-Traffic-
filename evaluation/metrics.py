"""
Quantitative Evaluation Metrics for VRU Intention and 3D Biomechanics
=====================================================================
Metrics for IEEE IROS / ICRA / VTC benchmarking:
- Intention Classification: Accuracy, Balanced Acc, F1-macro, Precision, Recall, AUC-ROC, PR-AUC
- 3D Biomechanical Lifting: MPJPE (mm), PA-MPJPE (mm), Bone Consistency Violation Rate (%)
- Early Horizon Evaluation: Time-to-Cross MAE (s), Prediction Accuracy vs. Lookahead Time (s)
- Reality Gap Robustness: Degradation Delta and Lane Dropout Sensitivity Slope
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from data.joint_taxonomy import CANONICAL_BONES, ANTHROPOMETRIC_BONE_LENGTHS


def compute_intention_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes classification metrics for binary crossing intention.
    Args:
        y_true: binary array of shape (N,)
        y_pred_probs: probability array of shape (N,) or (N, 2)
        threshold: decision threshold
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_probs = np.asarray(y_pred_probs)
    if y_pred_probs.ndim == 2 and y_pred_probs.shape[1] == 2:
        probs = y_pred_probs[:, 1]
    else:
        probs = y_pred_probs.ravel()

    preds = (probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, preds))
    bacc = float(balanced_accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    f1_pos = float(f1_score(y_true, preds, pos_label=1, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, probs))
    except Exception:
        pr_auc = float(np.mean(y_true))

    return {
        "accuracy": acc * 100.0,
        "balanced_acc": bacc * 100.0,
        "f1_macro": f1 * 100.0,
        "f1_crossing": f1_pos * 100.0,
        "precision": prec * 100.0,
        "recall": rec * 100.0,
        "auc_roc": auc,
        "pr_auc": pr_auc,
    }


def compute_mpjpe(
    pred_poses_3d: np.ndarray,
    gt_poses_3d: np.ndarray,
    align_root: bool = True
) -> float:
    """
    Mean Per-Joint Position Error (MPJPE) in millimeters.
    Args:
        pred_poses_3d: (N, J, 3) or (N, T, J, 3) in meters
        gt_poses_3d: same shape as pred in meters
        align_root: if True, aligns root pelvis (joint 8) to origin
    Returns:
        mpjpe_mm: error in millimeters
    """
    pred = pred_poses_3d.copy()
    gt = gt_poses_3d.copy()

    if align_root:
        # Joint 8 is Pelvis
        if pred.ndim == 4:
            pred = pred - pred[:, :, 8:9, :]
            gt = gt - gt[:, :, 8:9, :]
        elif pred.ndim == 3:
            pred = pred - pred[:, 8:9, :]
            gt = gt - gt[:, 8:9, :]

    diff = pred - gt
    dist_m = np.linalg.norm(diff, axis=-1) # in meters
    mpjpe_mm = float(np.mean(dist_m) * 1000.0) # convert to mm
    return mpjpe_mm


def compute_bone_consistency_violation(
    pred_poses_3d: np.ndarray,
    tolerance_pct: float = 0.15
) -> float:
    """
    Calculates percentage of predicted bone lengths that deviate from
    standard anthropometric bounds by more than tolerance_pct (e.g. 15%).
    Returns:
        violation_rate_pct: float percentage [0, 100]
    """
    total_bones = 0
    violated_bones = 0

    if pred_poses_3d.ndim == 4:
        # Flatten (N, T, J, 3) -> (N*T, J, 3)
        poses = pred_poses_3d.reshape(-1, pred_poses_3d.shape[2], 3)
    else:
        poses = pred_poses_3d

    for (p, c), exp_len in ANTHROPOMETRIC_BONE_LENGTHS.items():
        diff = poses[:, c, :] - poses[:, p, :]
        lens = np.linalg.norm(diff, axis=-1)
        err = np.abs(lens - exp_len) / exp_len
        violated_bones += np.sum(err > tolerance_pct)
        total_bones += len(lens)

    if total_bones == 0:
        return 0.0
    return float((violated_bones / total_bones) * 100.0)


def compute_ttc_mae(y_true_ttc: np.ndarray, y_pred_ttc: np.ndarray) -> float:
    """Mean Absolute Error for Time-to-Cross regression in seconds."""
    mae = float(np.mean(np.abs(y_true_ttc - y_pred_ttc)))
    return mae
