"""
End-to-end smoke test: invokes the real train.py CLI as a subprocess against a fully
isolated, synthetic manifest - exercising the actual training entry point (argument
parsing, manifest loading, dataloaders, model build, one real optimizer step, checkpoint
save) without touching any real dataset or the default config.MANIFEST_PATH/OUTPUT_ROOT.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.configs import config  # noqa: E402
from src.models.model import build_model  # noqa: E402
from src.training.train import set_seed  # noqa: E402


def test_end_to_end_smoke(tmp_path):
    # 1. Create a tiny synthetic manifest with consistent per-video labels and disjoint
    # train/val/test groups (assigned explicitly, so assign_group_splits's stricter
    # group-count requirements - see tests/test_dataset.py - are not a factor here).
    data_dir = tmp_path / "smoke_data"
    data_dir.mkdir()

    # 8 videos (2 frames each): 4 in train, 2 in val, 2 in test. Each video's frames all
    # share the same label, and each split's videos include both classes.
    video_specs = [
        ("vid_0", 0, "group_train"), ("vid_1", 1, "group_train"),
        ("vid_2", 0, "group_train"), ("vid_3", 1, "group_train"),
        ("vid_4", 0, "group_val"), ("vid_5", 1, "group_val"),
        ("vid_6", 0, "group_test"), ("vid_7", 1, "group_test"),
    ]
    split_for_group = {"group_train": "train", "group_val": "val", "group_test": "test"}

    rows = []
    img_idx = 0
    for video_id, label, group_id in video_specs:
        for _ in range(2):
            img_path = data_dir / f"img_{img_idx}.jpg"
            Image.new("RGB", (32, 32), color="red").save(img_path)
            rows.append({
                "image_path": str(img_path),
                "video_id": video_id,
                "label": label,
                "group_id": group_id,
                "split": split_for_group[group_id],
            })
            img_idx += 1

    manifest_path = tmp_path / "smoke_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    # 2. Run train.py via its real, supported CLI: --manifest overrides the manifest path
    # without mutating config.MANIFEST_PATH; --no-pretrained and --num_workers 0 keep the
    # run fast, deterministic, and independent of network access; --batch_size 2 ensures the
    # tiny 4-row train split survives the dataloader's drop_last=True (batch_size=32 would
    # drop the only batch and the test would pass trivially without training anything).
    output_root = tmp_path / "outputs"
    env = os.environ.copy()
    env["OUTPUT_ROOT"] = str(output_root)
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/training/train.py"),
            "--manifest", str(manifest_path),
            "--limit_batches", "1",
            "--epochs", "1",
            "--batch_size", "2",
            "--num_workers", "0",
            "--no-pretrained",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=300,
    )

    assert result.returncode == 0, f"Smoke test failed. Output:\n{result.stdout}\nError:\n{result.stderr}"
    assert "Training completed" in result.stdout

    # 3. Verify the run genuinely reached the training path (not skipped/short-circuited):
    # an experiment directory with a history/checkpoint should exist under the isolated
    # OUTPUT_ROOT, and the epoch summary line should be present in stdout.
    assert output_root.exists(), "Expected training outputs under the isolated OUTPUT_ROOT"
    history_files = list(output_root.glob("*/history.csv"))
    assert history_files, f"Expected an experiment history.csv under {output_root}"
    assert "Epoch 01/1" in result.stdout

    experiment_dir = history_files[0].parent

    # 4. A zero-batch or skipped run would leave train_loss exactly at its 0.0 sentinel (see
    # train_one_epoch's `running_loss / total_samples if total_samples > 0 else 0.0`) - assert
    # a genuine, finite, non-sentinel loss was computed for the single processed batch.
    history_df = pd.read_csv(history_files[0])
    epoch1_train_loss = float(history_df.loc[history_df["epoch"] == 1, "train_loss"].iloc[0])
    assert np.isfinite(epoch1_train_loss)
    assert epoch1_train_loss != 0.0, "train_loss is exactly 0.0 - no batch was actually processed"

    # 5. Strongest proof of a real optimizer step: reconstruct the exact same freshly-initialized
    # (pretrained=False) model using the same fixed seed the subprocess used (config.SEED, applied
    # via set_seed before any model is built - see train.py main()), and confirm the checkpoint's
    # saved weights differ from that fresh initialization. If no optimizer step had occurred, the
    # checkpointed classifier weights would be bit-identical to a fresh init with the same seed.
    checkpoint_path = experiment_dir / "last_model.pt"
    assert checkpoint_path.exists() and checkpoint_path.stat().st_size > 0

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trained_state = checkpoint["model_state_dict"]

    set_seed(config.SEED)
    fresh_model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion")
    fresh_state = fresh_model.state_dict()

    classifier_key = "head.fc2.weight"
    assert not torch.allclose(fresh_state[classifier_key], trained_state[classifier_key]), (
        "Checkpointed classifier weights are identical to a fresh initialization - "
        "no real optimizer step appears to have been applied."
    )
