"""
Metrics Utilities for Statistical Rigor, Calibration, and Hypothesis Testing.
Includes Non-parametric Bootstrap 95% Confidence Intervals, Paired Significance Testing,
Video-Level Aggregation, and Expected Calibration Error (ECE) calculations.
"""

from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, adaptive: bool = False) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    Args:
        y_true: Binary ground truth targets (0 or 1).
        y_prob: Predicted confidence probabilities for positive class (0.0 to 1.0).
        n_bins: Number of probability calibration bins.
        adaptive: If True, uses equal-frequency (quantile) binning instead of equal-width binning.

    Returns:
        Scalar ECE score (lower is better calibrated).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    total_samples = len(y_true)

    if total_samples == 0:
        return float("nan")

    if adaptive:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_boundaries = np.quantile(y_prob, quantiles)
        bin_boundaries = np.unique(bin_boundaries)
        if len(bin_boundaries) < 2:
            return 0.0
        n_bins = max(1, len(bin_boundaries) - 1)
    else:
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)



def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    threshold: float = 0.50,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute 95% Non-parametric Percentile Confidence Intervals via Frame-Level Bootstrap Resampling.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n_samples = len(y_true)

    rng = np.random.default_rng(seed)

    boot_acc = []
    boot_f1 = []
    boot_auc = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n_samples, n_samples)
        sub_y_true = y_true[indices]
        sub_y_prob = y_prob[indices]
        sub_y_pred = (sub_y_prob >= threshold).astype(int)

        if len(np.unique(sub_y_true)) < 2:
            continue

        boot_acc.append(accuracy_score(sub_y_true, sub_y_pred))
        boot_f1.append(f1_score(sub_y_true, sub_y_pred, zero_division=0))
        boot_auc.append(roc_auc_score(sub_y_true, sub_y_prob))

    def get_stats(arr: list) -> Dict[str, float]:
        if not arr:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        arr_np = np.array(arr)
        return {
            "mean": float(np.mean(arr_np)),
            "ci_lower": float(np.percentile(arr_np, 2.5)),
            "ci_upper": float(np.percentile(arr_np, 97.5)),
        }

    return {
        "accuracy": get_stats(boot_acc),
        "f1": get_stats(boot_f1),
        "roc_auc": get_stats(boot_auc),
    }


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_prob_std: np.ndarray,
    y_prob_rob: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
    threshold_std: float = 0.50,
    threshold_rob: float = 0.50
) -> Dict[str, float]:
    """
    Perform Paired Bootstrap Difference Test using true observed dataset difference delta_obs.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob_std = np.asarray(y_prob_std).astype(float)
    y_prob_rob = np.asarray(y_prob_rob).astype(float)
    n_samples = len(y_true)

    if n_samples == 0 or len(np.unique(y_true)) < 2:
        return {"mean_auc_diff": 0.0, "p_value_auc": 1.0, "mean_f1_diff": 0.0, "p_value_f1": 1.0}

    # Compute observed statistics directly on original un-resampled data
    obs_std_auc = float(roc_auc_score(y_true, y_prob_std))
    obs_rob_auc = float(roc_auc_score(y_true, y_prob_rob))
    obs_auc_diff = obs_rob_auc - obs_std_auc

    obs_std_f1 = float(f1_score(y_true, (y_prob_std >= threshold_std).astype(int), zero_division=0))
    obs_rob_f1 = float(f1_score(y_true, (y_prob_rob >= threshold_rob).astype(int), zero_division=0))
    obs_f1_diff = obs_rob_f1 - obs_std_f1

    rng = np.random.default_rng(seed)
    auc_diffs = []
    f1_diffs = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n_samples, n_samples)
        sub_true = y_true[idx]
        sub_std = y_prob_std[idx]
        sub_rob = y_prob_rob[idx]

        if len(np.unique(sub_true)) < 2:
            continue

        std_auc = roc_auc_score(sub_true, sub_std)
        rob_auc = roc_auc_score(sub_true, sub_rob)
        auc_diffs.append(rob_auc - std_auc)

        std_f1 = f1_score(sub_true, (sub_std >= threshold_std).astype(int), zero_division=0)
        rob_f1 = f1_score(sub_true, (sub_rob >= threshold_rob).astype(int), zero_division=0)
        f1_diffs.append(rob_f1 - std_f1)

    auc_diffs = np.array(auc_diffs)
    f1_diffs = np.array(f1_diffs)

    shifted_auc = auc_diffs - float(np.mean(auc_diffs))
    shifted_f1 = f1_diffs - float(np.mean(f1_diffs))

    B_auc = len(shifted_auc)
    B_f1 = len(shifted_f1)

    p_val_auc = float((np.sum(np.abs(shifted_auc) >= np.abs(obs_auc_diff)) + 1) / (B_auc + 1)) if B_auc > 0 else 1.0
    p_val_f1 = float((np.sum(np.abs(shifted_f1) >= np.abs(obs_f1_diff)) + 1) / (B_f1 + 1)) if B_f1 > 0 else 1.0

    return {
        "mean_auc_diff": obs_auc_diff,
        "p_value_auc": min(1.0, max(0.0, p_val_auc)),
        "mean_f1_diff": obs_f1_diff,
        "p_value_f1": min(1.0, max(0.0, p_val_f1)),
    }


def _get_video_group_cols(df: pd.DataFrame) -> List[str]:
    if "manipulation" in df.columns:
        return ["manipulation", "video_id"]
    if "category" in df.columns:
        return ["category", "video_id"]
    return ["video_id"]


def compute_video_level_metrics(predictions_df: pd.DataFrame, threshold: float = 0.50) -> Dict[str, Any]:
    """
    Pool frame-level probability predictions per unique video_id to calculate video-level metrics & ECE.
    """
    if predictions_df is None or len(predictions_df) == 0:
        return {
            "video_roc_auc": float("nan"),
            "video_accuracy": float("nan"),
            "video_f1": float("nan"),
            "video_ece": float("nan"),
            "num_videos": 0
        }

    group_cols = _get_video_group_cols(predictions_df)
    if not (predictions_df.groupby(group_cols)["label"].nunique() == 1).all():
        raise ValueError(f"Label discrepancy detected within video groups: {group_cols}")

    video_df = predictions_df.groupby(group_cols).agg({
        "label": "first",
        "prob_fake": "mean"
    }).reset_index()

    y_true = video_df["label"].to_numpy().astype(int)
    y_prob = video_df["prob_fake"].to_numpy().astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    video_auc = float("nan")
    video_acc = float("nan")
    video_f1 = float("nan")
    video_ece = float("nan")

    if len(np.unique(y_true)) == 2:
        video_auc = float(roc_auc_score(y_true, y_prob))
        video_acc = float(accuracy_score(y_true, y_pred))
        video_f1 = float(f1_score(y_true, y_pred, zero_division=0))
        video_ece = calculate_ece(y_true, y_prob)

    return {
        "video_roc_auc": video_auc,
        "video_accuracy": video_acc,
        "video_f1": video_f1,
        "video_ece": video_ece,
        "num_videos": len(video_df),
        "video_df": video_df,
    }


def bootstrap_video_level_ci(
    predictions_df: pd.DataFrame,
    threshold: float = 0.50,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute 95% Non-parametric Percentile Confidence Intervals with VIDEO-LEVEL sampling unit.
    """
    group_cols = _get_video_group_cols(predictions_df)
    video_summary = predictions_df.groupby(group_cols).agg({
        "label": "first",
        "prob_fake": "mean"
    }).reset_index()

    videos = video_summary["video_id"].to_numpy()
    labels = video_summary["label"].to_numpy().astype(int)
    probs = video_summary["prob_fake"].to_numpy().astype(float)
    n_videos = len(videos)

    if n_videos == 0 or len(np.unique(labels)) < 2:
        return {
            "accuracy": {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
            "f1": {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
            "roc_auc": {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
        }

    rng = np.random.default_rng(seed)
    boot_acc, boot_f1, boot_auc = [], [], []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n_videos, n_videos)
        sub_true = labels[idx]
        sub_prob = probs[idx]
        sub_pred = (sub_prob >= threshold).astype(int)

        if len(np.unique(sub_true)) < 2:
            continue

        boot_acc.append(accuracy_score(sub_true, sub_pred))
        boot_f1.append(f1_score(sub_true, sub_pred, zero_division=0))
        boot_auc.append(roc_auc_score(sub_true, sub_prob))

    def get_stats(arr: list) -> Dict[str, float]:
        if not arr:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        arr_np = np.array(arr)
        return {
            "mean": float(np.mean(arr_np)),
            "ci_lower": float(np.percentile(arr_np, 2.5)),
            "ci_upper": float(np.percentile(arr_np, 97.5)),
        }

    return {
        "accuracy": get_stats(boot_acc),
        "f1": get_stats(boot_f1),
        "roc_auc": get_stats(boot_auc),
    }


def paired_video_bootstrap_test(
    predictions_std_df: pd.DataFrame,
    predictions_rob_df: pd.DataFrame,
    threshold_std: float = 0.50,
    threshold_rob: float = 0.50,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Perform Paired Bootstrap Difference Test at the VIDEO sampling level.
    """
    group_cols = _get_video_group_cols(predictions_std_df)
    std_vid = predictions_std_df.groupby(group_cols).agg({"label": "first", "prob_fake": "mean"}).reset_index()
    rob_vid = predictions_rob_df.groupby(group_cols).agg({"label": "first", "prob_fake": "mean"}).reset_index()

    merged = pd.merge(std_vid, rob_vid, on=group_cols, suffixes=("_std", "_rob"))
    if len(merged) == 0:
        return {"mean_auc_diff": 0.0, "p_value_auc": 1.0, "mean_f1_diff": 0.0, "p_value_f1": 1.0}

    if not (merged["label_std"] == merged["label_rob"]).all():
        raise ValueError("Paired video bootstrap test requires matching labels across compared models.")

    y_true = merged["label_std"].to_numpy().astype(int)
    prob_std = merged["prob_fake_std"].to_numpy().astype(float)
    prob_rob = merged["prob_fake_rob"].to_numpy().astype(float)

    return paired_bootstrap_test(
        y_true=y_true,
        y_prob_std=prob_std,
        y_prob_rob=prob_rob,
        n_bootstraps=n_bootstraps,
        seed=seed,
        threshold_std=threshold_std,
        threshold_rob=threshold_rob
    )


from scipy.stats import false_discovery_control

def apply_fdr_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Benjamini-Hochberg False Discovery Rate (FDR) multiple-testing correction via SciPy."""
    p_vals = np.asarray(p_values, dtype=float)
    if len(p_vals) == 0:
        return np.array([]), np.array([], dtype=bool)
    adjusted = false_discovery_control(p_vals)
    return adjusted, adjusted <= alpha


