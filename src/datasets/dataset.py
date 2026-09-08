import random
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.configs import config
from src.degradations.transforms import get_transforms


def _dataloader_worker_init_fn(worker_id):
    """Seed a DataLoader worker deterministically from the base torch seed.

    Defined at module scope (not nested inside get_dataloaders) so it is picklable — a local
    closure cannot be pickled by 'spawn'/'forkserver' multiprocessing start methods (the
    default on Windows, macOS, and, as of Python 3.14, non-macOS POSIX platforms too).
    """
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class DeepfakeImageDataset(Dataset):
    def __init__(self, dataframe, transform, image_modifier=None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.image_modifier = image_modifier

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = row["image_path"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            raise RuntimeError(f"Could not load image: {image_path}") from error

        if self.image_modifier is not None:
            try:
                image = self.image_modifier(image, image_path=str(image_path))
            except TypeError:
                image = self.image_modifier(image)

        image = self.transform(image)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        manipulation = str(row["manipulation"]) if "manipulation" in row else (str(row["category"]) if "category" in row else "unknown")

        return {
            "image": image,
            "label": label,
            "path": image_path,
            "video_id": row["video_id"],
            "manipulation": manipulation,
        }




def build_connected_groups(video_ids):
    """Build connected components from video IDs using scipy."""
    # Collect all explicit vids AND implicit identities
    nodes = set()
    edges = []
    
    for vid in video_ids:
        s = str(vid).split('.')[0]
        nodes.add(s)
        parts = s.split('_')
        if len(parts) >= 2:
            id1, id2 = parts[0], parts[1]
            if (id1.isdigit() and id2.isdigit()) or (id1.startswith('id') and id1[2:].isdigit() and id2.startswith('id') and id2[2:].isdigit()):
                nodes.add(id1); nodes.add(id2)
                edges.append((s, id1))
                edges.append((s, id2))
                edges.append((id1, id2))
            elif id1.startswith('id') and id1[2:].isdigit():
                nodes.add(id1)
                edges.append((s, id1))
        elif len(parts) == 1 and parts[0].startswith('id') and parts[0][2:].isdigit():
            nodes.add(parts[0])
            edges.append((s, parts[0]))
            
    nodes = list(nodes)
    vid_to_idx = {v: i for i, v in enumerate(nodes)}
    
    rows, cols = [], []
    for u, v in edges:
        rows.append(vid_to_idx[u])
        cols.append(vid_to_idx[v])
        
    if rows:
        import numpy as np
        graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes), len(nodes)))
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    else:
        labels = range(len(nodes))
        
    # Return mapping only for the original video_ids
    return {str(vid).split('.')[0]: f"group_{labels[vid_to_idx[str(vid).split('.')[0]]]}" for vid in video_ids}




def assign_group_splits(dataframe, train_ratio=None, validation_ratio=None, seed=42):
    """Assign train/val/test splits with stratification by group-level majority label.

    Uses pre-computed group_id if present; otherwise computes robust graph connected-components.

    Strict leakage policy: a group_id is never placed in more than one split. If there are too
    few independent groups (per class-dominance bucket) to carve out non-empty, non-overlapping
    train/val/test partitions, this raises ValueError rather than silently reusing groups across
    splits. Tiny/debugging datasets that cannot satisfy this should pre-assign an explicit
    'split' column instead of relying on this function.
    """
    if train_ratio is None:
        train_ratio = getattr(config, "TRAIN_RATIO", 0.70)
    if validation_ratio is None:
        validation_ratio = getattr(config, "VAL_RATIO", 0.15)
    dataframe = dataframe.copy()

    if "group_id" not in dataframe.columns and "video_id" in dataframe.columns:
        group_map = build_connected_groups(dataframe["video_id"].unique())
        dataframe["group_id"] = dataframe["video_id"].map(
            lambda v: group_map.get(str(v).split('.')[0], str(v))
        )

    # Stratify groups by majority label
    group_fake_ratio = (
        dataframe.groupby("group_id")["label"]
        .mean()
        .rename("fake_ratio")
        .reset_index()
    )
    fake_dominant = sorted(
        group_fake_ratio.loc[group_fake_ratio["fake_ratio"] >= 0.5, "group_id"].astype(str).tolist()
    )
    real_dominant = sorted(
        group_fake_ratio.loc[group_fake_ratio["fake_ratio"] < 0.5, "group_id"].astype(str).tolist()
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(fake_dominant)
    rng.shuffle(real_dominant)

    def _split_bucket(bucket, train_r, val_r):
        """Split a list of same-class-dominant groups into disjoint (train, val, test) lists.

        Never overlaps groups across the returned lists. If the bucket is too small to
        independently carve out a val/test slice (fewer than 3 groups), the whole bucket is
        kept in train and empty lists are returned for val/test — the other class bucket (if
        any) may still supply non-empty val/test groups. If neither bucket can, the caller's
        final non-empty-split check below raises a clear error instead of allowing leakage.
        """
        n = len(bucket)
        if n == 0:
            return [], [], []

        if n < 3:
            return list(bucket), [], []

        train_end = max(1, int(n * train_r))
        val_end = train_end + max(1, int(n * val_r))
        if val_end >= n:
            val_end = n - 1
            train_end = max(1, val_end - 1)
        return bucket[:train_end], bucket[train_end:val_end], bucket[val_end:]

    fake_train, fake_val, fake_test = _split_bucket(fake_dominant, train_ratio, validation_ratio)
    real_train, real_val, real_test = _split_bucket(real_dominant, train_ratio, validation_ratio)

    train_groups = set(fake_train + real_train)
    validation_groups = set(fake_val + real_val)
    test_groups = set(fake_test + real_test)

    # This should be structurally unreachable given the disjoint bucket allocation above, but
    # is kept as a defensive invariant check — leakage must never be silently tolerated.
    overlap_tv = train_groups & validation_groups
    overlap_tt = train_groups & test_groups
    overlap_vt = validation_groups & test_groups
    if overlap_tv or overlap_tt or overlap_vt:
        raise AssertionError(
            f"Internal error: group split produced overlapping group_ids: "
            f"train∩val={overlap_tv}, train∩test={overlap_tt}, val∩test={overlap_vt}"
        )

    if not validation_groups or not test_groups:
        total_groups = len(fake_dominant) + len(real_dominant)
        raise ValueError(
            f"Cannot produce a leakage-free train/val/test split: too few unique source video "
            f"groups ({total_groups} available) to form non-empty, non-overlapping "
            f"validation and test partitions. Provide more source videos, or pre-assign an "
            f"explicit 'split' column for tiny/debugging datasets."
        )

    def map_split(group_id):
        group_id = str(group_id)
        if group_id in train_groups:
            return "train"
        if group_id in validation_groups:
            return "val"
        if group_id in test_groups:
            return "test"
        raise ValueError(f"Unknown group_id '{group_id}' not assigned to any split.")

    dataframe["split"] = dataframe["group_id"].map(map_split)
    return dataframe


def validate_celebdf_manifest(df_or_path):
    """Validate that a DataFrame or CSV file path represents a genuine Celeb-DF manifest across all rows."""
    if isinstance(df_or_path, (str, Path)):
        path = Path(df_or_path)
        if not path.exists() or path.stat().st_size == 0:
            return False, f"Celeb-DF manifest file does not exist or is empty: {path}"
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return False, f"Failed to read Celeb-DF manifest CSV ({e}): {path}"
    else:
        df = df_or_path

    if df is None or len(df) == 0:
        return False, "Celeb-DF manifest is empty."

    required_cols = {"image_path", "video_id", "label"}
    if not required_cols.issubset(df.columns):
        return False, f"Celeb-DF manifest missing required columns: {required_cols - set(df.columns)}"

    sample_paths = df["image_path"].astype(str).tolist()
    for p in sample_paths:
        if "ffpp_c23" in p or "processed_faces/ffpp" in p or "ffpp_c40" in p:
            return False, f"Manifest contains FaceForensics++ paths instead of Celeb-DF: {p}"

    if "category" in df.columns:
        cats = set(df["category"].dropna().unique())
        valid_celebdf_cats = {"Celeb-real", "Celeb-synthesis", "YouTube-real"}
        if cats and not cats.issubset(valid_celebdf_cats):
            return False, f"Manifest categories {cats} contain invalid categories outside {valid_celebdf_cats}"

    return True, "Valid Celeb-DF manifest."


def validate_manifest(df_or_path):
    """Validate full training/evaluation dataset manifest for structural integrity, split isolation, and labels."""
    if isinstance(df_or_path, (str, Path)):
        path = Path(df_or_path)
        if not path.exists() or path.stat().st_size == 0:
            return False, f"Manifest file does not exist or is empty: {path}"
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return False, f"Failed to read manifest CSV ({e}): {path}"
    else:
        df = df_or_path

    if df is None or len(df) == 0:
        return False, "Manifest is empty."

    required_cols = {"image_path", "video_id", "label", "group_id"}
    if not required_cols.issubset(df.columns):
        return False, f"Manifest missing required columns: {required_cols - set(df.columns)}"

    # Check for NaNs
    if df["label"].isna().any():
        return False, "Manifest contains NaN labels."
    if df["video_id"].isna().any():
        return False, "Manifest contains NaN video_ids."
    if df["group_id"].isna().any():
        return False, "Manifest contains NaN group_ids."
    if df["image_path"].isna().any():
        return False, "Manifest contains NaN image_paths."

    # Check for duplicates
    if df["image_path"].duplicated().any():
        return False, "Manifest contains duplicate image_paths."

    # Verify file existence
    import os
    missing = df["image_path"].apply(lambda p: not os.path.exists(p))
    if missing.any():
        return False, f"Manifest contains nonexistent image files. E.g. {df[missing]['image_path'].iloc[0]}"

    labels = set(df["label"].unique())
    if not labels.issubset({0, 1, 0.0, 1.0}):
        return False, f"Manifest labels contain invalid values: {labels - {0, 1, 0.0, 1.0}}"
        
    # Check consistent labels per video
    video_labels = df.groupby("video_id")["label"].nunique()
    if (video_labels > 1).any():
        return False, "Manifest contains inconsistent labels for a single video_id."

    if "split" in df.columns:
        valid_splits = {"train", "val", "test"}
        splits = set(df["split"].unique())
        if not splits.issubset(valid_splits):
            return False, f"Manifest contains unexpected splits: {splits - valid_splits}"
            
        for s in valid_splits:
            if s not in splits or len(df[df["split"] == s]) == 0:
                return False, f"Manifest split '{s}' is empty."

        train_grps = set(df[df["split"] == "train"]["group_id"].astype(str))
        val_grps = set(df[df["split"] == "val"]["group_id"].astype(str))
        test_grps = set(df[df["split"] == "test"]["group_id"].astype(str))

        leakage_tv = train_grps.intersection(val_grps)
        leakage_tt = train_grps.intersection(test_grps)
        leakage_vt = val_grps.intersection(test_grps)

        # Group leakage is never acceptable, regardless of dataset size — a manifest with
        # overlapping group_ids across splits is invalid.
        if leakage_tv:
            return False, f"Group leakage detected between train and val splits: {leakage_tv}"
        if leakage_tt:
            return False, f"Group leakage detected between train and test splits: {leakage_tt}"
        if leakage_vt:
            return False, f"Group leakage detected between val and test splits: {leakage_vt}"

    return True, "Valid manifest."


def _load_celebdf_test_videos(celebdf_root):
    """Parse List_of_testing_videos.txt into a set of matchable identifiers.

    The official file lists entries like "1 Celeb-synthesis/id0_id1_0000.mp4". We store the
    relative path, the bare filename, and the stem so callers can match against whichever
    representation their source layout naturally produces (pre-extracted image-crop folder
    name vs. a raw video file path) without duplicating this parsing logic.
    """
    test_list_path = Path(celebdf_root) / "List_of_testing_videos.txt"
    test_videos = set()
    if test_list_path.exists():
        with open(test_list_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    rel_path = parts[1].strip()
                    test_videos.add(rel_path)
                    test_videos.add(Path(rel_path).name)
                    test_videos.add(Path(rel_path).stem)
    return test_videos


def generate_celebdf_manifest(celebdf_root=None, output_path=None, max_frames_per_video=10):
    """Canonical Celeb-DF v2 manifest generator (single source of truth).

    Supports two source layouts, matching either of which is expected to be sufficient:
      1. Pre-extracted face-crop images organized as ``<root>/<category>/<video_name>/*.jpg``.
      2. Raw video files anywhere under ``<root>`` (used as a fallback when no pre-extracted
         images are found; frames are extracted via OpenCV).

    Schema (consistent regardless of source layout): image_path, label, video_id, category,
    split, dataset, group_id.

    Semantics: ALL videos are retained (never silently dropped) — videos listed in the
    official ``List_of_testing_videos.txt`` are labeled ``split="test"``, everything else is
    labeled ``split="train"``. This matches how ``evaluate.run_celebdf_eval`` consumes the
    manifest (it filters to ``split == "test"`` when the column is present, and otherwise
    falls back to using the entire manifest as the test set).
    """
    if celebdf_root is None:
        celebdf_root = getattr(config, "CELEBDF_ROOT", Path("datasets/Celeb-DF-v2"))
    celebdf_root = Path(celebdf_root)

    if output_path is None:
        output_path = getattr(config, "CELEBDF_MANIFEST_PATH", Path("deepfake_robustness/celebdf_manifest.csv"))
    output_path = Path(output_path) if output_path is not None else None

    categories = {"Celeb-real": 0, "YouTube-real": 0, "Celeb-synthesis": 1}
    test_videos = _load_celebdf_test_videos(celebdf_root)

    records = []
    image_extensions = {".jpg", ".jpeg", ".png"}

    # 1. Pre-extracted face-crop images.
    for category_dir in celebdf_root.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        category_name = category_dir.name
        label = categories.get(category_name, 0 if category_name in ("Celeb-real", "YouTube-real") else 1)

        for item in category_dir.rglob("*"):
            if item.is_file() and item.suffix.lower() in image_extensions:
                vid_name = item.parent.name
                rel_video_path = f"{category_name}/{vid_name}.mp4"
                split = "test" if (rel_video_path in test_videos or vid_name in test_videos) else "train"
                records.append({
                    "image_path": str(item.resolve()),
                    "label": float(label),
                    "video_id": vid_name,
                    "category": category_name,
                    "split": split,
                    "dataset": "Celeb-DF-v2",
                })

    # 2. Fallback: raw video files, extracted via OpenCV, only used when no pre-extracted
    #    image crops were found anywhere under the root.
    if not records:
        video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        video_files = sorted(
            p for p in celebdf_root.rglob("*")
            if p.is_file() and p.suffix.lower() in video_extensions
        )
        if video_files:
            import cv2

            processed_dir = celebdf_root / "processed_faces"
            max_frames = max_frames_per_video or 10

            for vid_path in video_files:
                vid_name = vid_path.stem
                parent_dir = vid_path.parent.name

                if parent_dir in categories:
                    category_name, label = parent_dir, categories[parent_dir]
                elif "synthesis" in parent_dir.lower() or "synthesis" in vid_name.lower() or "fake" in vid_name.lower():
                    category_name, label = "Celeb-synthesis", 1
                elif "youtube" in parent_dir.lower():
                    category_name, label = "YouTube-real", 0
                else:
                    category_name, label = "Celeb-real", 0

                capture = cv2.VideoCapture(str(vid_path))
                if not capture.isOpened():
                    continue
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    capture.release()
                    continue
                step = max(1, total_frames // max_frames)

                vid_out_dir = processed_dir / category_name / vid_name
                vid_out_dir.mkdir(parents=True, exist_ok=True)
                split = "test" if (vid_name in test_videos or f"{category_name}/{vid_name}.mp4" in test_videos) else "train"

                frame_idx, saved_count = 0, 0
                while capture.isOpened() and saved_count < max_frames:
                    success, frame = capture.read()
                    if not success:
                        break
                    if frame_idx % step == 0:
                        out_file = vid_out_dir / f"frame_{saved_count:04d}.jpg"
                        cv2.imwrite(str(out_file), frame)
                        records.append({
                            "image_path": str(out_file),
                            "label": float(label),
                            "video_id": vid_name,
                            "category": category_name,
                            "split": split,
                            "dataset": "Celeb-DF-v2",
                        })
                        saved_count += 1
                    frame_idx += 1
                capture.release()

    df = pd.DataFrame(records)
    if not df.empty:
        group_map = build_connected_groups(df["video_id"].unique())
        df["group_id"] = df["video_id"].map(lambda v: group_map.get(str(v), str(v)))
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
    return df


def get_dataloaders(manifest_df):
    if "split" not in manifest_df.columns:
        manifest_df = assign_group_splits(manifest_df, seed=config.SEED)

    train_df = manifest_df[manifest_df["split"] == "train"].copy()
    val_df = manifest_df[manifest_df["split"] == "val"].copy()
    test_df = manifest_df[manifest_df["split"] == "test"].copy()

    sampler = None
    if len(train_df) > 0 and train_df["label"].nunique() > 1:
        if config.BALANCING_STRATEGY == "oversampling":
            class_counts = train_df["label"].value_counts()
            max_size = class_counts.max()
            lst = [group.sample(max_size, replace=True, random_state=config.SEED) for _, group in train_df.groupby("label")]
            train_df = pd.concat(lst, ignore_index=True).sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
        elif config.BALANCING_STRATEGY == "undersampling":
            class_counts = train_df["label"].value_counts()
            min_size = class_counts.min()
            lst = [group.sample(min_size, replace=False, random_state=config.SEED) for _, group in train_df.groupby("label")]
            train_df = pd.concat(lst, ignore_index=True).sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
        elif config.BALANCING_STRATEGY == "sampler":
            class_counts = train_df["label"].astype(int).value_counts().to_dict()
            sample_weights = train_df["label"].astype(int).map(lambda label: 1.0 / class_counts[label]).to_numpy()
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True
            )

    train_transform, eval_transform = get_transforms()

    train_dataset = DeepfakeImageDataset(train_df, train_transform)
    val_dataset = DeepfakeImageDataset(val_df, eval_transform)
    test_dataset = DeepfakeImageDataset(test_df, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
        drop_last=True,
        worker_init_fn=_dataloader_worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
    )

    return train_loader, val_loader, test_loader

