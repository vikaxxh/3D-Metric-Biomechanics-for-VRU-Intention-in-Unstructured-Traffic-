"""
IDD-PeD (Indian Driving Dataset - Pedestrian) Adapter
====================================================
Parses IDD-PeD dataset files:
- 2D Keypoints from MMPose (17 COCO joints)
- Spatial bounding boxes and occlusion flags
- 6 Fine-Grained Behavioral Attributes:
    1. Crossing Behavior (CU, CFU, CD, CFD, CI, N/A)
    2. Traffic Interaction (WTT, HG, Other, N/A)
    3. Pedestrian Activity (Walking, Sprinting, N/A)
    4. Attention Indicators (LOS, FTT, NL, DB)
    5. Social Dynamics (GS, CFA, AWC, N/A)
    6. Stationary Behavior (Sitting, Standing, IWA, Other, N/A)
- Unstructured Scene Attributes: missing lane markings, absence of traffic lights
"""

import os
import json
import glob
import numpy as np
from typing import Dict, List, Any, Optional
from data.joint_taxonomy import coco_to_canonical

# Canonical Mapping for Crossing Intention
# Positive (Crossing = 1): CU, CFU, CD, CFD
# Negative (Not Crossing = 0): N/A, CI (Crossing Irrelevant to ego-vehicle)
CROSSING_LABELS_MAP = {
    "CU": 1,   # Crossing Undesignated
    "CFU": 1,  # Crossing Fast Undesignated
    "CD": 1,   # Crossing Designated
    "CFD": 1,  # Crossing Fast Designated
    "CI": 0,   # Crossing Irrelevant
    "N/A": 0,  # Not Crossing
}


class IDDPeDAdapter:
    """
    Adapter for IDD-PeD sequences.
    """

    def __init__(self, data_root: Optional[str] = None, fps: float = 10.0):
        self.data_root = data_root
        self.fps = fps
        self.samples: List[Dict[str, Any]] = []

    def load_from_directory(self, dir_path: str) -> int:
        """Loads all processed .json or .npz IDD-PeD files from dir_path."""
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
        """Parses a single IDD-PeD annotation file."""
        try:
            if fpath.endswith(".json"):
                with open(fpath, "r") as f:
                    raw = json.load(f)
                kpts_17 = np.array(raw["keypoints_2d"], dtype=np.float32)
                bboxes = np.array(raw.get("bboxes_2d", []), dtype=np.float32)
                seq_id = raw.get("sequence_id", os.path.basename(fpath))
                track_id = raw.get("track_id", 0)
                behaviors = raw.get("behavioral_attributes", {})
                lane_present = raw.get("lane_markings_present", False)
                traffic_light = raw.get("traffic_light_present", False)
            elif fpath.endswith(".npz"):
                data = np.load(fpath, allow_pickle=True)
                kpts_17 = data["keypoints_2d"].astype(np.float32)
                bboxes = data.get("bboxes_2d", np.zeros((len(kpts_17), 4), dtype=np.float32))
                seq_id = str(data.get("sequence_id", os.path.basename(fpath)))
                track_id = int(data.get("track_id", 0))
                behaviors = dict(data.get("behavioral_attributes", {}).item()) if "behavioral_attributes" in data else {}
                lane_present = bool(data.get("lane_markings_present", False))
                traffic_light = bool(data.get("traffic_light_present", False))
            else:
                return None

            # Convert 17 COCO joints to 15 Canonical joints (2D pixels / normalized)
            canonical_2d = coco_to_canonical(kpts_17)

            # Determine binary intention label (prefer human annotation in npz)
            if "intention_label" in data:
                intention_label = int(data["intention_label"])
            else:
                cross_attr = behaviors.get("crossing_behavior", "N/A")
                intention_label = CROSSING_LABELS_MAP.get(cross_attr, 0)

            cross_pt = int(data.get("crossing_point", -1))

            # Fine-grained behavior flags
            is_weaving = behaviors.get("traffic_interaction") == "WTT"
            is_looking_over_shoulder = behaviors.get("attention_indicators") == "LOS"
            is_sprinting = behaviors.get("pedestrian_activity") == "Sprinting"

            return {
                "dataset": "idd_ped",
                "sequence_id": seq_id,
                "track_id": track_id,
                "poses_2d": canonical_2d,
                "bboxes_2d": bboxes,
                "behaviors": behaviors,
                "crossing_attribute": cross_attr,
                "intention_label": intention_label,
                "crossing_point": cross_pt,
                "is_weaving": is_weaving,
                "is_looking_over_shoulder": is_looking_over_shoulder,
                "is_sprinting": is_sprinting,
                "lane_markings_present": lane_present,
                "traffic_light_present": traffic_light,
                "num_frames": canonical_2d.shape[0],
            }
        except Exception as e:
            print(f"[IDDPeDAdapter] Error parsing {fpath}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
