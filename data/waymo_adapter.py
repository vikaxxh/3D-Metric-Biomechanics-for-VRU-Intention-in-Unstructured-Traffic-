"""
Waymo Open Dataset 3D Keypoints Adapter
=======================================
Parses and normalizes Waymo 3D Pedestrian Keypoint sequences into canonical
15-joint metric coordinate frames.
"""

import os
import json
import glob
import numpy as np
from typing import Dict, List, Any, Optional, Iterator
from data.joint_taxonomy import waymo_to_canonical, CANONICAL_BONES
from data.biomechanics_features import extract_full_biomechanical_feature_vector


class Waymo3DKeypointAdapter:
    """
    Adapter for Waymo Open Dataset 3D Keypoint sequences.
    Reads exported JSON/PKL/NPY annotations containing:
    - sequence_id: str
    - track_id: int / str
    - timestamps: List[float] (10Hz)
    - keypoints_3d: np.ndarray shape (T, 14, 3) in meters
    - bboxes_3d: np.ndarray shape (T, 7) [x, y, z, dx, dy, dz, heading]
    - intention_label: int (0: Not Crossing, 1: Crossing)
    """

    def __init__(self, data_root: Optional[str] = None, fps: float = 10.0):
        self.data_root = data_root
        self.fps = fps
        self.samples: List[Dict[str, Any]] = []

    def load_from_directory(self, dir_path: str) -> int:
        """Loads all processed .json or .npz sequence files from dir_path."""
        if not os.path.exists(dir_path):
            return 0

        files = sorted(glob.glob(os.path.join(dir_path, "*.json")) + 
                       glob.glob(os.path.join(dir_path, "*.npz")))
        for fpath in files:
            sample = self.parse_file(fpath)
            if sample is not None:
                self.samples.append(sample)
        return len(self.samples)

    def parse_file(self, fpath: str) -> Optional[Dict[str, Any]]:
        """Parses a single file into a canonical sample dictionary."""
        try:
            if fpath.endswith(".json"):
                with open(fpath, "r") as f:
                    raw = json.load(f)
                kpts_14 = np.array(raw["keypoints_3d"], dtype=np.float32)
                seq_id = raw.get("sequence_id", os.path.basename(fpath))
                track_id = raw.get("track_id", 0)
                label = raw.get("intention_label", int(raw.get("crossing", 0)))
                bboxes = np.array(raw.get("bboxes_3d", []), dtype=np.float32)
            elif fpath.endswith(".npz"):
                data = np.load(fpath, allow_pickle=True)
                kpts_14 = data["keypoints_3d"].astype(np.float32)
                seq_id = str(data.get("sequence_id", os.path.basename(fpath)))
                track_id = int(data.get("track_id", 0))
                label = int(data.get("intention_label", 0))
                bboxes = data.get("bboxes_3d", np.zeros((len(kpts_14), 7), dtype=np.float32))
            else:
                return None

            # Convert 14 Waymo joints to 15 Canonical joints
            canonical_3d = waymo_to_canonical(kpts_14)
            features = extract_full_biomechanical_feature_vector(canonical_3d, fps=self.fps)

            return {
                "dataset": "waymo",
                "sequence_id": seq_id,
                "track_id": track_id,
                "poses_3d": canonical_3d,
                "features": features,
                "bboxes_3d": bboxes,
                "intention_label": label,
                "num_frames": canonical_3d.shape[0],
            }
        except Exception as e:
            print(f"[WaymoAdapter] Error parsing {fpath}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
