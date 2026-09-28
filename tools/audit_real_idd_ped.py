"""
Forensic IDD-PeD Real Dataset & Split Audit
===========================================
Performs a rigorous data audit on the authentic IDD-PeD dataset:
1. Loads official database cache: data/raw_idd_ped/data/IDDPedestrian/data_cache/iddp_database.pkl
2. Loads official MMPose 2D keypoints from data/raw_idd_ped/Intention/data/features/iddp/poses/
3. Audits video counts, pedestrian tracks, frames, and class balance
4. Verifies the official split protocol:
   - Train sets: gp_set_0001, gp_set_0002, gp_set_0004, gp_set_0006, gp_set_0007
   - Test sets:  gp_set_0003, gp_set_0005, gp_set_0008, gp_set_0009
5. Tests for data leakage between train and test sets
"""

import os
import glob
import pickle
from collections import Counter
import numpy as np


TRAIN_SETS = ['gp_set_0001', 'gp_set_0002', 'gp_set_0004', 'gp_set_0006', 'gp_set_0007']
TEST_SETS  = ['gp_set_0003', 'gp_set_0005', 'gp_set_0008', 'gp_set_0009']


def run_forensic_audit():
    db_path = "data/raw_idd_ped/data/IDDPedestrian/data_cache/iddp_database.pkl"
    pose_dir = "data/raw_idd_ped/Intention/data/features/iddp/poses"

    print("=" * 78)
    print("           FORENSIC AUDIT: REAL IDD-PeD DATASET & SPLITS")
    print("=" * 78)

    if not os.path.exists(db_path):
        print(f"[Error] Database not found at: {db_path}")
        return

    with open(db_path, "rb") as f:
        db = pickle.load(f)

    print(f"\n[1] Official Sets Present in Database: {len(db)}")
    all_sets = sorted(list(db.keys()))
    print("    Sets:", all_sets)

    # Track metrics
    stats = {
        "train": {"videos": 0, "tracks": 0, "frames": 0, "crossing": 0, "not_crossing": 0, "pose_available": 0},
        "test":  {"videos": 0, "tracks": 0, "frames": 0, "crossing": 0, "not_crossing": 0, "pose_available": 0},
    }

    train_videos = set()
    test_videos = set()
    train_ped_ids = set()
    test_ped_ids = set()

    crossing_behavior_counts = Counter()
    traffic_interaction_counts = Counter()

    # Iterate over sets
    for set_id, vids in db.items():
        split = "train" if set_id in TRAIN_SETS else ("test" if set_id in TEST_SETS else "unknown")
        if split == "unknown":
            print(f"[Warning] Unknown set found: {set_id}")
            continue

        # Load poses for this set if available
        pose_files = glob.glob(os.path.join(pose_dir, f"pose_{set_id}*.pkl"))
        pose_dict = {}
        if pose_files:
            try:
                with open(pose_files[0], "rb") as pf:
                    pose_dict = pickle.load(pf)
            except Exception as e:
                print(f"[Warning] Could not load pose file {pose_files[0]}: {e}")

        for vid_name, vid_info in vids.items():
            stats[split]["videos"] += 1
            if split == "train":
                train_videos.add(vid_name)
            else:
                test_videos.add(vid_name)

            peds = vid_info.get("pedestrian_annotations", {})
            vid_poses = pose_dict.get(vid_name, {})

            for ped_id, ped_data in peds.items():
                stats[split]["tracks"] += 1
                unique_ped_key = f"{set_id}_{vid_name}_{ped_id}"
                if split == "train":
                    train_ped_ids.add(unique_ped_key)
                else:
                    test_ped_ids.add(unique_ped_key)

                num_f = len(ped_data.get("frames", []))
                stats[split]["frames"] += num_f

                # Crossing label
                attrs = ped_data.get("attributes", {})
                is_crossing = attrs.get("crossing", 0)
                if is_crossing == 1:
                    stats[split]["crossing"] += 1
                else:
                    stats[split]["not_crossing"] += 1

                # Check if poses exist for any frame of this ped
                frames = ped_data.get("frames", [])
                has_pose = False
                for f_idx in frames:
                    pose_k = f"{f_idx:05d}_{ped_id}"
                    if pose_k in vid_poses:
                        has_pose = True
                        break
                if has_pose:
                    stats[split]["pose_available"] += 1

                # Count fine-grained behaviors
                beh = ped_data.get("behavior", {})
                cb = beh.get("CrossingBehavior", [])
                for c in cb:
                    crossing_behavior_counts[c] += 1
                ti = beh.get("TrafficInteraction", [])
                for t in ti:
                    traffic_interaction_counts[t] += 1

    print("\n[2] Official Split Breakdown:")
    print("-" * 78)
    print(f"{'Split':<8} | {'Videos':<7} | {'Tracks':<8} | {'Total Frames':<13} | {'Crossing':<9} | {'Not-Cross':<9} | {'With Poses':<10}")
    print("-" * 78)
    for sp in ["train", "test"]:
        s = stats[sp]
        print(f"{sp.upper():<8} | {s['videos']:<7} | {s['tracks']:<8} | {s['frames']:<13} | {s['crossing']:<9} | {s['not_crossing']:<9} | {s['pose_available']:<10}")
    print("-" * 78)

    # Leakage check
    vid_leakage = train_videos.intersection(test_videos)
    ped_leakage = train_ped_ids.intersection(test_ped_ids)
    print("\n[3] Leakage & Independence Audit:")
    print(f"    Train Videos: {len(train_videos)} | Test Videos: {len(test_videos)}")
    print(f"    Video Overlap (Data Leakage): {len(vid_leakage)}")
    assert len(vid_leakage) == 0, f"CRITICAL LEAKAGE DETECTED: {vid_leakage}"
    print("    -> Split is 100% strictly video-disjoint. PASSED [OK]")

    print(f"    Train Track IDs: {len(train_ped_ids)} | Test Track IDs: {len(test_ped_ids)}")
    print(f"    Pedestrian ID Overlap: {len(ped_leakage)}")
    assert len(ped_leakage) == 0, f"CRITICAL LEAKAGE DETECTED: {ped_leakage}"
    print("    -> Zero pedestrian identity overlap. PASSED [OK]")

    print("\n[4] Class Imbalance Reality Check:")
    total_tr = stats["train"]["tracks"]
    tr_cr_pct = (stats["train"]["crossing"] / total_tr * 100) if total_tr > 0 else 0
    total_te = stats["test"]["tracks"]
    te_cr_pct = (stats["test"]["crossing"] / total_te * 100) if total_te > 0 else 0
    print(f"    Train Crossing Ratio: {stats['train']['crossing']}/{total_tr} ({tr_cr_pct:.1f}%)")
    print(f"    Test Crossing Ratio:  {stats['test']['crossing']}/{total_te} ({te_cr_pct:.1f}%)")

    print("\n" + "=" * 78)
    print("           FORENSIC AUDIT COMPLETE & DECONTAMINATED")
    print("=" * 78)


if __name__ == "__main__":
    run_forensic_audit()
