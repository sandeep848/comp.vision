"""
Unit tests for Dataset splitting logic to guarantee zero video_id / group_id leakage.
"""

import pytest
import pandas as pd
import numpy as np
from deepfake_robustness.datasets.dataset import assign_group_splits

def test_zero_group_leakage_across_splits():
    """Verify assign_group_splits guarantees zero group_id leakage across train, val, and test splits."""
    # Generate synthetic video manifest with 50 source video groups
    records = []
    for g in range(50):
        group_id = f"video_{g:03d}"
        for f in range(20):  # 20 frames per video
            records.append({
                "image_path": f"/tmp/frames/{group_id}_frame_{f:02d}.jpg",
                "video_id": group_id,
                "label": g % 2
            })
            
    df = pd.DataFrame(records)
    
    # Run assign_group_splits across multiple random seeds
    for seed in [42, 100, 2026, 999]:
        split_df = assign_group_splits(df, train_ratio=0.70, validation_ratio=0.15, seed=seed)
        
        train_groups = set(split_df[split_df["split"] == "train"]["group_id"])
        val_groups = set(split_df[split_df["split"] == "val"]["group_id"])
        test_groups = set(split_df[split_df["split"] == "test"]["group_id"])
        
        # Assert mutually disjoint sets (Zero leakage)
        assert len(train_groups.intersection(val_groups)) == 0, f"Leakage between train and val on seed {seed}"
        assert len(train_groups.intersection(test_groups)) == 0, f"Leakage between train and test on seed {seed}"
        assert len(val_groups.intersection(test_groups)) == 0, f"Leakage between val and test on seed {seed}"

def test_split_ratios():
    """Verify assign_group_splits correctly approximates 70% / 15% / 15% group allocation."""
    records = [{"video_id": f"vid_{i}", "label": i % 2} for i in range(100)]
    df = pd.DataFrame(records)
    
    split_df = assign_group_splits(df, train_ratio=0.70, validation_ratio=0.15, seed=42)
    group_counts = split_df.groupby("split")["group_id"].nunique()
    
    assert group_counts["train"] == 70
    assert abs(group_counts["val"] - 15) <= 1
    assert abs(group_counts["test"] - 15) <= 1


def test_small_bucket_split_error():
    """Verify assign_group_splits raises ValueError for < 3 groups instead of creating split leakage."""
    df_1 = pd.DataFrame([{"video_id": "v0", "label": 0}])
    with pytest.raises(ValueError, match="too few unique source video groups"):
        assign_group_splits(df_1)

    df_2 = pd.DataFrame([{"video_id": "v0", "label": 0}, {"video_id": "v1", "label": 1}])
    with pytest.raises(ValueError, match="too few unique source video groups"):
        assign_group_splits(df_2)


def test_four_groups_still_too_few_for_disjoint_split():
    """Even with more groups (2 per class = 4 total), if no single class bucket has >= 3
    groups, a genuine non-overlapping 3-way split is still impossible and must raise -
    the implementation must never fall back to reusing a group across splits."""
    df = pd.DataFrame([
        {"video_id": "r0", "label": 0}, {"video_id": "r1", "label": 0},
        {"video_id": "f0", "label": 1}, {"video_id": "f1", "label": 1},
    ])
    with pytest.raises(ValueError, match="too few unique source video groups"):
        assign_group_splits(df)


def test_mixed_bucket_sizes_still_disjoint_when_one_bucket_has_enough_groups():
    """If one class bucket has enough groups to internally split 3 ways, val/test can be
    populated even when the other class bucket is too small to contribute - as long as the
    combined result stays leakage-free (no group ever appears in more than one split)."""
    records = [{"video_id": f"real_{g}", "label": 0} for g in range(10)]
    records.append({"video_id": "fake_0", "label": 1})
    df = pd.DataFrame(records)

    split_df = assign_group_splits(df, train_ratio=0.70, validation_ratio=0.15, seed=42)

    train_groups = set(split_df.loc[split_df["split"] == "train", "group_id"])
    val_groups = set(split_df.loc[split_df["split"] == "val", "group_id"])
    test_groups = set(split_df.loc[split_df["split"] == "test", "group_id"])

    assert val_groups, "expected a non-empty val split"
    assert test_groups, "expected a non-empty test split"
    assert not (train_groups & val_groups)
    assert not (train_groups & test_groups)
    assert not (val_groups & test_groups)


def test_group_sets_match_final_row_split_labels():
    """Regression test for the collapse-to-train bug: the group_id membership used to decide
    split assignment must exactly match the final per-row 'split' column for every row, and
    no group may be shared between two splits."""
    records = []
    for g in range(30):
        for f in range(5):
            records.append({
                "image_path": f"/tmp/frames/video_{g:03d}_frame_{f:02d}.jpg",
                "video_id": f"video_{g:03d}",
                "label": g % 2,
            })
    df = pd.DataFrame(records)
    split_df = assign_group_splits(df, train_ratio=0.70, validation_ratio=0.15, seed=42)

    train_groups = set(split_df.loc[split_df["split"] == "train", "group_id"])
    val_groups = set(split_df.loc[split_df["split"] == "val", "group_id"])
    test_groups = set(split_df.loc[split_df["split"] == "test", "group_id"])

    assert train_groups and val_groups and test_groups
    assert not (train_groups & val_groups)
    assert not (train_groups & test_groups)
    assert not (val_groups & test_groups)

    # Every group must appear under exactly one split label across all of its rows -
    # group-set membership and per-row split labels can never disagree.
    group_split_counts = split_df.groupby("group_id")["split"].nunique()
    assert (group_split_counts == 1).all()

