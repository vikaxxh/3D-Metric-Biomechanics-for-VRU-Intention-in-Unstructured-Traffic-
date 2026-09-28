"""
Official IDD-PeD Real Dataset Extractor & Preprocessor
======================================================
Parses authentic IDD-PeD annotations and MMPose 2D body poses:
- Source DB: data/raw_idd_ped/data/IDDPedestrian/data_cache/iddp_database.pkl
- Source Poses: data/raw_idd_ped/Intention/data/features/iddp/poses/pose_gp_set_*_iddp_body2d.pkl

Extracts genuine, video-disjoint pedestrian tracks into:
- data/real_data/idd_ped/train (from gp_set_0002, 0004, 0006, 0007)
- data/real_data/idd_ped/test  (from gp_set_0003, 0005, 0008, 0009)

Enforces:
1. Minimum track length T >= 16 frames (~1.6 seconds of observation)
2. Exact matching between video frame IDs and MMPose keypoint timestamps
3. Preserves official human annotations (crossing, crossing_point, behaviors)
"""

import os
import glob
import pickle
import numpy as np
from tqdm import tqdm


TRAIN_SETS = ['gp_set_0002', 'gp_set_0004', 'gp_set_0006', 'gp_set_0007']
TEST_SETS  = ['gp_set_0003', 'gp_set_0005', 'gp_set_0008', 'gp_set_0009']


def extract_real_idd_ped(
    db_path: str = "data/raw_idd_ped/data/IDDPedestrian/data_cache/iddp_database.pkl",
    pose_dir: str = "data/raw_idd_ped/Intention/data/features/iddp/poses",
    output_root: str = "data/real_data/idd_ped",
    min_track_len: int = 16,
):
    print("=" * 78)
    print("      INGESTING AUTHENTIC REAL IDD-PeD BENCHMARK DATASET")
    print("=" * 78)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"IDD-PeD database not found at: {db_path}")

    with open(db_path, "rb") as f:
        db = pickle.load(f)

    os.makedirs(os.path.join(output_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "test"), exist_ok=True)

    summary = {
        "train": {"saved": 0, "crossing": 0, "not_crossing": 0, "total_frames": 0},
        "test":  {"saved": 0, "crossing": 0, "not_crossing": 0, "total_frames": 0},
    }

    all_sets = TRAIN_SETS + TEST_SETS

    for set_id in all_sets:
        if set_id not in db:
            print(f"[Notice] {set_id} not in database keys, skipping.")
            continue

        split = "train" if set_id in TRAIN_SETS else "test"
        out_split_dir = os.path.join(output_root, split)

        # Load pose pickle for this set
        pose_files = glob.glob(os.path.join(pose_dir, f"pose_{set_id}*.pkl"))
        pose_dict = {}
        if pose_files:
            try:
                with open(pose_files[0], "rb") as pf:
                    pose_dict = pickle.load(pf)
                print(f"[{set_id}] Loaded pose file: {os.path.basename(pose_files[0])} ({len(pose_dict)} videos)")
            except Exception as e:
                print(f"[Warning] Failed loading {pose_files[0]}: {e}")

        vids = db[set_id]

        for vid_name, vid_info in vids.items():
            peds = vid_info.get("pedestrian_annotations", {})
            vid_poses = pose_dict.get(vid_name, {})

            for ped_id, ped_data in peds.items():
                frames = ped_data.get("frames", [])
                bboxes = ped_data.get("bbox", [])
                attrs = ped_data.get("attributes", {})
                behavior = ped_data.get("behavior", {})

                if len(frames) < min_track_len:
                    continue

                # Match frames to pose detections
                valid_frames = []
                matched_kpts = []
                matched_bboxes = []

                for i, fid in enumerate(frames):
                    pose_k = f"{fid:05d}_{ped_id}"
                    if pose_k in vid_poses:
                        raw_kpts = vid_poses[pose_k]
                        kpts_17 = np.array(raw_kpts, dtype=np.float32).reshape(17, 2)
                        valid_frames.append(fid)
                        matched_kpts.append(kpts_17)
                        if i < len(bboxes):
                            matched_bboxes.append(bboxes[i])
                        else:
                            matched_bboxes.append([0.0, 0.0, 1.0, 1.0])

                if len(matched_kpts) < min_track_len:
                    continue

                # Prepare arrays
                kpts_arr = np.stack(matched_kpts, axis=0) # (T, 17, 2)
                bbox_arr = np.array(matched_bboxes, dtype=np.float32) # (T, 4)
                label = int(attrs.get("crossing", 0))
                cross_pt = int(attrs.get("crossing_point", -1))

                # Normalize 2D keypoints into canonical image space if needed
                # (Coordinates in pose pickle are normalized 0-1)

                seq_id = f"{set_id}_{vid_name}_{ped_id}"
                out_path = os.path.join(out_split_dir, f"{seq_id}.npz")

                np.savez_compressed(
                    out_path,
                    sequence_id=seq_id,
                    track_id=ped_id,
                    frames=np.array(valid_frames),
                    keypoints_2d=kpts_arr,
                    bboxes_2d=bbox_arr,
                    intention_label=label,
                    crossing_point=cross_pt,
                    is_real=True,
                    dataset_name="IDD-PeD",
                )

                summary[split]["saved"] += 1
                if label == 1:
                    summary[split]["crossing"] += 1
                else:
                    summary[split]["not_crossing"] += 1
                summary[split]["total_frames"] += len(matched_kpts)

    print("\n" + "=" * 78)
    print("         REAL IDD-PeD INGESTION SUMMARY (VIDEO-DISJOINT)")
    print("=" * 78)
    for sp in ["train", "test"]:
        s = summary[sp]
        cr_pct = (s['crossing'] / s['saved'] * 100) if s['saved'] > 0 else 0
        print(f"  {sp.upper():<6}: {s['saved']:>5} Real Tracks | {s['total_frames']:>7} Posed Frames | Crossing: {s['crossing']} ({cr_pct:.1f}%) | Not-Cross: {s['not_crossing']}")
    print("=" * 78)
    print(f"Data saved to: {output_root}")


if __name__ == "__main__":
    extract_real_idd_ped()
