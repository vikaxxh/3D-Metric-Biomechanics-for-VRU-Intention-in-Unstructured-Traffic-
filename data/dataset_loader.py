"""
PyTorch Datasets and DataLoaders for Waymo 3D and IDD-PeD
=========================================================
Handles batch collation, tensor conversion, temporal standardization,
and robust handling of variable-length real-world tracking sequences.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Any, Optional
from data.waymo_adapter import Waymo3DKeypointAdapter
from data.idd_ped_adapter import IDDPeDAdapter


def pad_or_truncate_sequence(arr: np.ndarray, target_len: int = 16) -> np.ndarray:
    """Standardizes temporal dimension T to target_len via slicing or edge padding."""
    T = arr.shape[0]
    if T == target_len:
        return arr
    elif T > target_len:
        # Take the most recent target_len frames leading up to decision
        return arr[-target_len:]
    else:
        # Edge pad first frame to reach target_len
        pad_count = target_len - T
        pad_prefix = np.repeat(arr[:1], pad_count, axis=0)
        return np.concatenate([pad_prefix, arr], axis=0)


class Waymo3DPyTorchDataset(Dataset):
    def __init__(self, data_dir: str, fps: float = 10.0, seq_len: int = 16):
        self.adapter = Waymo3DKeypointAdapter(fps=fps)
        self.adapter.load_from_directory(data_dir)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.adapter)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.adapter[idx]
        raw_poses_3d = sample["poses_3d"] # (T, 15, 3)
        features = sample["features"]

        poses_3d_np = pad_or_truncate_sequence(raw_poses_3d, self.seq_len)
        vel_np = pad_or_truncate_sequence(features["velocities"], self.seq_len)
        com_np = pad_or_truncate_sequence(features["com_velocity"], self.seq_len)

        poses_3d = torch.from_numpy(poses_3d_np).float()
        velocities = torch.from_numpy(vel_np).float()
        com_vel = torch.from_numpy(com_np).float()
        label = torch.tensor(sample["intention_label"], dtype=torch.long)

        return {
            "poses_3d": poses_3d,
            "velocities": velocities,
            "com_vel": com_vel,
            "label": label,
            "seq_id": sample["sequence_id"],
        }


class IDDPeDPyTorchDataset(Dataset):
    def __init__(self, data_dir: str, fps: float = 10.0, seq_len: int = 16):
        self.adapter = IDDPeDAdapter(fps=fps)
        self.adapter.load_from_directory(data_dir)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.adapter)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.adapter[idx]
        raw_poses_2d = sample["poses_2d"] # (T, 15, 2)
        raw_bboxes_2d = sample["bboxes_2d"] # (T, 4)

        poses_2d_np = pad_or_truncate_sequence(raw_poses_2d, self.seq_len)
        bboxes_2d_np = pad_or_truncate_sequence(raw_bboxes_2d, self.seq_len)

        poses_2d = torch.from_numpy(poses_2d_np).float()
        bboxes_2d = torch.from_numpy(bboxes_2d_np).float()
        label = torch.tensor(sample["intention_label"], dtype=torch.long)

        # Fine-grained attribute index
        attr_idx = 0
        if sample.get("crossing_attribute") == "CU":
            attr_idx = 1
        elif sample.get("crossing_attribute") == "CFU":
            attr_idx = 2
        if sample.get("is_weaving"):
            attr_idx = 3
        elif sample.get("is_looking_over_shoulder"):
            attr_idx = 4
        fine_label = torch.tensor(attr_idx, dtype=torch.long)

        lane_val = 1.0 if sample.get("lane_markings_present") else 0.0
        signal_val = 1.0 if sample.get("traffic_light_present") else 0.0

        infra_cues = torch.zeros((self.seq_len, 3), dtype=torch.float32)
        infra_cues[:, 0] = lane_val
        infra_cues[:, 1] = signal_val

        cross_pt = sample.get("crossing_point", -1)
        if sample["intention_label"] == 1 and cross_pt > 0:
            ttc_val = max(0.0, float(cross_pt) / 10.0)
            ttc = torch.tensor([ttc_val], dtype=torch.float32)
        else:
            ttc = torch.tensor([0.0], dtype=torch.float32)

        return {
            "poses_2d": poses_2d,
            "bboxes_2d": bboxes_2d,
            "infra_cues": infra_cues,
            "label": label,
            "fine_label": fine_label,
            "ttc": ttc,
            "seq_id": sample["sequence_id"],
        }


def get_dataloaders(
    config: dict,
    batch_size: int = 16,
    num_workers: int = 0,
    seq_len: int = 16,
) -> Dict[str, DataLoader]:
    """Builds train and validation dataloaders for both Waymo and IDD-PeD."""
    data_cfg = config.get("data", {})

    loaders = {}
    if os.path.exists(data_cfg.get("waymo_train_dir", "")):
        ds_waymo_train = Waymo3DPyTorchDataset(data_cfg["waymo_train_dir"], seq_len=seq_len)
        loaders["waymo_train"] = DataLoader(ds_waymo_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if os.path.exists(data_cfg.get("waymo_val_dir", "")):
        ds_waymo_val = Waymo3DPyTorchDataset(data_cfg["waymo_val_dir"], seq_len=seq_len)
        loaders["waymo_val"] = DataLoader(ds_waymo_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if os.path.exists(data_cfg.get("idd_ped_train_dir", "")):
        ds_idd_train = IDDPeDPyTorchDataset(data_cfg["idd_ped_train_dir"], seq_len=seq_len)
        loaders["idd_train"] = DataLoader(ds_idd_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if os.path.exists(data_cfg.get("idd_ped_val_dir", "")):
        ds_idd_val = IDDPeDPyTorchDataset(data_cfg["idd_ped_val_dir"], seq_len=seq_len)
        loaders["idd_val"] = DataLoader(ds_idd_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if os.path.exists(data_cfg.get("idd_ped_test_dir", "")):
        ds_idd_test = IDDPeDPyTorchDataset(data_cfg["idd_ped_test_dir"], seq_len=seq_len)
        loaders["idd_test"] = DataLoader(ds_idd_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return loaders
