import argparse
import hashlib
import io
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import cv2

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from torch.utils.data import DataLoader

from deepfake_robustness.configs import config
from deepfake_robustness.datasets.dataset import DeepfakeImageDataset
from deepfake_robustness.datasets.dataset import generate_celebdf_manifest, validate_celebdf_manifest
from deepfake_robustness.degradations.transforms import get_transforms
from deepfake_robustness.models.model import build_model, resolve_checkpoint_model_kwargs
from deepfake_robustness.training.train import evaluate_model, DeepfakeLoss
from deepfake_robustness.evaluation.metrics_utils import (
    bootstrap_video_level_ci,
    paired_video_bootstrap_test,
    calculate_ece,
    compute_video_level_metrics,
    apply_fdr_correction,
)

def apply_advanced_tier_distortion(pil_img, tier_cfg, sample_id="", global_seed=42):
    """Apply deterministic synthetic platform distortion to PIL Image."""
    seed_str = f"{global_seed}_{sample_id}_{tier_cfg.get('jpeg_quality')}_{tier_cfg.get('resize_scale')}_{tier_cfg.get('blur_sigma')}_{tier_cfg.get('noise_std')}"
    seed_val = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_val)

    img = pil_img.copy()
    w, h = img.size

    # 1. Resizing / Downscaling
    scale = tier_cfg.get("resize_scale", 1.0)
    if scale < 1.0:
        nw = max(16, int(w * scale))
        nh = max(16, int(h * scale))
        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        img = img.resize((w, h), Image.Resampling.BILINEAR)

    # 2. Gaussian Blur
    blur_sigma = tier_cfg.get("blur_sigma")
    if blur_sigma is not None and blur_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur_sigma))

    # 3. Motion Blur
    motion_size = tier_cfg.get("motion_blur_size")
    if motion_size is not None and motion_size > 1:
        img_np = np.array(img)
        kernel = np.zeros((motion_size, motion_size))
        kernel[int((motion_size - 1) / 2), :] = np.ones(motion_size)
        kernel /= motion_size
        img_np = cv2.filter2D(img_np, -1, kernel)
        img = Image.fromarray(img_np)

    # 4. Color Jitter (Deterministic RNG)
    jitter = tier_cfg.get("color_jitter")
    if jitter is not None and jitter > 0:
        b_factor = 1.0 + float(rng.uniform(-jitter, jitter))
        c_factor = 1.0 + float(rng.uniform(-jitter, jitter))
        img = ImageEnhance.Brightness(img).enhance(b_factor)
        img = ImageEnhance.Contrast(img).enhance(c_factor)

    # 5. Gaussian Noise (Deterministic RNG)
    noise_std = tier_cfg.get("noise_std")
    if noise_std is not None and noise_std > 0:
        img_np = np.array(img, dtype=np.float32)
        noise = rng.normal(0, noise_std, img_np.shape)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)

    # 6. JPEG Compression
    q = tier_cfg.get("jpeg_quality")
    if q is not None:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(q))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img


class TierDistortionModifier:
    def __init__(self, tier_cfg, global_seed=42):
        self.tier_cfg = tier_cfg
        self.global_seed = global_seed

    def __call__(self, img, image_path=""):
        sample_id = image_path if image_path else getattr(img, "filename", hashlib.md5(img.tobytes()).hexdigest())
        return apply_advanced_tier_distortion(img, self.tier_cfg, sample_id=sample_id, global_seed=self.global_seed)





def evaluate_single_model_on_tier(model, test_df, eval_transform, tier_cfg, criterion, device, limit_batches=None):
    modifier = TierDistortionModifier(tier_cfg, global_seed=config.SEED)
    ds = DeepfakeImageDataset(test_df, transform=eval_transform, image_modifier=modifier)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    metrics, preds_df = evaluate_model(model, loader, criterion, device, limit_batches=limit_batches)
    return metrics, preds_df

def load_model_from_checkpoint(ckpt_path, device):
    import warnings
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("configuration", {})
    if not isinstance(cfg, dict):
        cfg = {}

    model_name, model_variant, branch_mode = resolve_checkpoint_model_kwargs(ckpt)

    if "model_name" not in ckpt and "model_name" not in cfg:
        warnings.warn(f"Checkpoint at {ckpt_path} missing self-describing metadata. Falling back to config.py defaults.")

    model = build_model(model_name=model_name, pretrained=False, model_variant=model_variant, branch_mode=branch_mode).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    threshold = float(cfg.get("optimal_threshold", 0.50))
    return model, threshold, ckpt


def run_comparative_benchmark(n_bootstraps=1000, limit_batches=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Running Comparative Deepfake Detection Benchmark Matrix on {device}...")

    manifest_df = pd.read_csv(config.MANIFEST_PATH)
    if "split" not in manifest_df.columns:
        from deepfake_robustness.datasets.dataset import assign_group_splits
        manifest_df = assign_group_splits(manifest_df, seed=config.SEED)
    test_df = manifest_df[manifest_df["split"] == "test"].copy()
    _, eval_transform = get_transforms()
    criterion = DeepfakeLoss(loss_type="bce", smoothing=0.0)

    variant_suffix = f"_{config.MODEL_VARIANT}" if getattr(config, "MODEL_VARIANT", "fusion") != "fusion" else ""

    # 1. Load Standard / Clean Model
    std_name = f"{config.MODEL_NAME}{variant_suffix}_clean"
    std_ckpt_path = config.OUTPUT_ROOT / std_name / "best_model.pt"
    if not std_ckpt_path.exists():
        std_ckpt_path = config.OUTPUT_ROOT / f"{config.MODEL_NAME}{variant_suffix}_standard" / "best_model.pt"
    if not std_ckpt_path.exists():
        std_ckpt_path = config.OUTPUT_ROOT / f"{config.MODEL_NAME}_clean" / "best_model.pt"
    if not std_ckpt_path.exists():
        raise FileNotFoundError(f"Standard model checkpoint not found at: {std_ckpt_path}")
    
    std_model, std_thresh, std_ckpt = load_model_from_checkpoint(std_ckpt_path, device)
    print(f"Loaded Standard Model from {std_ckpt_path} (threshold={std_thresh:.4f})...")

    # 2. Load Robustness Model
    rob_name = f"{config.MODEL_NAME}{variant_suffix}_degradation"
    rob_ckpt_path = config.OUTPUT_ROOT / rob_name / "best_model.pt"
    if not rob_ckpt_path.exists():
        rob_ckpt_path = config.OUTPUT_ROOT / f"{config.MODEL_NAME}_degradation" / "best_model.pt"
    if not rob_ckpt_path.exists():
        raise FileNotFoundError(f"Robustness model checkpoint not found at: {rob_ckpt_path}")

    rob_model, rob_thresh, rob_ckpt = load_model_from_checkpoint(rob_ckpt_path, device)
    print(f"Loading Robustness Model from {rob_ckpt_path} (threshold={rob_thresh:.4f})...")

    results = []
    raw_p_values = []
    raw_preds_dir = config.OUTPUT_ROOT / "predictions"
    raw_preds_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*120)
    print(f"{'Degradation Tier':<25} | {'Std AUC (95% CI)':<22} | {'Std ECE':<8} | {'Rob AUC (95% CI)':<22} | {'Rob ECE':<8} | {'Δ AUC':<7} | {'p-value'}")
    print("="*120)

    for tier_name, tier_cfg in config.DEGRADATION_TIERS.items():
        std_m, std_preds = evaluate_single_model_on_tier(std_model, test_df, eval_transform, tier_cfg, criterion, device, limit_batches=limit_batches)
        rob_m, rob_preds = evaluate_single_model_on_tier(rob_model, test_df, eval_transform, tier_cfg, criterion, device, limit_batches=limit_batches)

        assert (std_preds["image_path"].to_numpy() == rob_preds["image_path"].to_numpy()).all(), "Paired evaluation image path mismatch!"
        assert (std_preds["label"].to_numpy() == rob_preds["label"].to_numpy()).all(), "Paired evaluation label mismatch!"

        # Persist predictions
        std_preds_out = std_preds.copy()
        std_preds_out["tier"] = tier_name
        std_preds_out["checkpoint"] = "standard"
        std_preds_out["abstain"] = False # Replace with logic if abstain is added
        
        rob_preds_out = rob_preds.copy()
        rob_preds_out["tier"] = tier_name
        rob_preds_out["checkpoint"] = "robustness"
        rob_preds_out["abstain"] = False

        std_preds_out.to_csv(raw_preds_dir / f"preds_std_{tier_name.replace(' ', '_')}.csv", index=False)
        rob_preds_out.to_csv(raw_preds_dir / f"preds_rob_{tier_name.replace(' ', '_')}.csv", index=False)

        std_vid = compute_video_level_metrics(std_preds, threshold=std_thresh)
        rob_vid = compute_video_level_metrics(rob_preds, threshold=rob_thresh)

        y_true = std_preds["label"].to_numpy().astype(int)
        std_probs = std_preds["prob_fake"].to_numpy().astype(float)
        rob_probs = rob_preds["prob_fake"].to_numpy().astype(float)

        # Video-level bootstrap non-parametric 95% Confidence Intervals
        std_ci = bootstrap_video_level_ci(std_preds, threshold=std_thresh, n_bootstraps=n_bootstraps, seed=config.SEED)
        rob_ci = bootstrap_video_level_ci(rob_preds, threshold=rob_thresh, n_bootstraps=n_bootstraps, seed=config.SEED)

        std_low, std_high = std_ci["roc_auc"]["ci_lower"], std_ci["roc_auc"]["ci_upper"]
        rob_low, rob_high = rob_ci["roc_auc"]["ci_lower"], rob_ci["roc_auc"]["ci_upper"]

        # Video-level paired bootstrap hypothesis testing
        rob_p = paired_video_bootstrap_test(
            std_preds, rob_preds, threshold_std=std_thresh, threshold_rob=rob_thresh, n_bootstraps=n_bootstraps, seed=config.SEED
        )
        p_val = float(rob_p.get("p_value_auc", 1.0))
        raw_p_values.append(p_val)

        std_frame_ece = calculate_ece(y_true, std_probs)
        rob_frame_ece = calculate_ece(y_true, rob_probs)
        std_video_ece = std_vid.get("video_ece", std_frame_ece)
        rob_video_ece = rob_vid.get("video_ece", rob_frame_ece)

        delta_auc = (rob_vid["video_roc_auc"] - std_vid["video_roc_auc"]) * 100.0 if not np.isnan(std_vid["video_roc_auc"]) else (rob_m["roc_auc"] - std_m["roc_auc"]) * 100.0

        results.append({
            "tier": tier_name,
            "std_auc": std_vid["video_roc_auc"] if not np.isnan(std_vid["video_roc_auc"]) else std_m["roc_auc"],
            "std_frame_auc": std_m["roc_auc"],
            "std_video_auc": std_vid["video_roc_auc"],
            "std_ci_low": std_low,
            "std_ci_high": std_high,
            "std_ece": std_video_ece,
            "std_frame_ece": std_frame_ece,
            "rob_auc": rob_vid["video_roc_auc"] if not np.isnan(rob_vid["video_roc_auc"]) else rob_m["roc_auc"],
            "rob_frame_auc": rob_m["roc_auc"],
            "rob_video_auc": rob_vid["video_roc_auc"],
            "rob_ci_low": rob_low,
            "rob_ci_high": rob_high,
            "rob_ece": rob_video_ece,
            "rob_frame_ece": rob_frame_ece,
            "delta_auc": delta_auc,
            "p_value": p_val,
            "std_thresh": std_thresh,
            "rob_thresh": rob_thresh,
        })

    # Apply Benjamini-Hochberg FDR correction across degradation tiers
    adj_p_values, sig_mask = apply_fdr_correction(raw_p_values, alpha=0.05)
    for idx, res in enumerate(results):
        res["fdr_p_value"] = float(adj_p_values[idx])
        res["is_significant"] = bool(sig_mask[idx])

        std_str = f"{res['std_auc']*100:.2f}% [{res['std_ci_low']*100:.1f}%, {res['std_ci_high']*100:.1f}%]"
        rob_str = f"{res['rob_auc']*100:.2f}% [{res['rob_ci_low']*100:.1f}%, {res['rob_ci_high']*100:.1f}%]"
        p_str = f"{res['fdr_p_value']:.3f}" if res['fdr_p_value'] >= 0.001 else "< 0.001 ***"

        print(f"{res['tier']:<25} | {std_str:<22} | {res['std_ece']:.4f}   | {rob_str:<22} | {res['rob_ece']:.4f}   | {res['delta_auc']:+6.2f}% | {p_str}")

    print("="*120)

    # Save CSV and Markdown reports with provenance
    prov = get_experiment_provenance()
    for res in results:
        res.update(prov)

    report_df = pd.DataFrame(results)
    csv_path = config.OUTPUT_ROOT / "comparative_robustness_stat_results.csv"
    report_df.to_csv(csv_path, index=False)

    report_path = config.OUTPUT_ROOT / "comparative_robustness_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Comparative Deepfake Detection Robustness & Statistical Rigor Report\n\n")
        f.write("> **University of Technology Nuremberg — Statistically Rigorous Benchmark Report**\n")
        f.write(f"> **Model Architecture**: {config.MODEL_NAME.upper()} Dual-Branch Fusion\n")
        f.write(f"> **Provenance**: Git `{prov['git_commit'][:8]}` | PyTorch `{prov['torch_version']}` | Evaluated at `{prov['timestamp']}`\n")
        f.write("> **Statistical Protocol**: 95% Non-parametric Percentile Bootstrap CIs, Video & Frame ECE, Paired Video Bootstrap Hypothesis Testing, Benjamini-Hochberg FDR Correction.\n\n")
        f.write("---\n\n## 1. Degradation Benchmark Matrix\n\n")
        f.write("| Degradation Tier | Standard ROC-AUC (95% CI) | Standard ECE | Robustness ROC-AUC (95% CI) | Robustness ECE | $\\Delta$ ROC-AUC Gain | FDR-Adjusted $p$-value |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            p_s = f"`{r['fdr_p_value']:.3f}`" if r['fdr_p_value'] >= 0.001 else "`< 0.001 ***`"
            if not r['is_significant']:
                p_s += " (n.s.)"
            f.write(f"| **{r['tier']}** | {r['std_auc']*100:.2f}% [{r['std_ci_low']*100:.1f}%, {r['std_ci_high']*100:.1f}%] | {r['std_ece']:.4f} | **{r['rob_auc']*100:.2f}% [{r['rob_ci_low']*100:.1f}%, {r['rob_ci_high']*100:.1f}%]** | **{r['rob_ece']:.4f}** | **{r['delta_auc']:+.2f}%** | {p_s} |\n")

    print(f"\n[+] Exported comparative report to: {report_path}")


def get_experiment_provenance():
    import subprocess, sys
    commit_hash = "unknown"
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        commit_hash = res.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # Optional provenance metadata only (e.g. no git installed, or not a git checkout).
        pass
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": commit_hash,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
    }




def run_celebdf_eval(limit_batches=None):
    """Run Celeb-DF v2 cross-dataset generalization evaluation supporting frame & video metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Starting Celeb-DF v2 Cross-Dataset Generalization Evaluation on {device}...")

    manifest_df = None
    if config.CELEBDF_MANIFEST_PATH.exists() and config.CELEBDF_MANIFEST_PATH.stat().st_size > 0:
        is_valid, reason = validate_celebdf_manifest(config.CELEBDF_MANIFEST_PATH)
        if is_valid:
            manifest_df = pd.read_csv(config.CELEBDF_MANIFEST_PATH)
        else:
            print(f"[!] Warning: Tracked Celeb-DF manifest validation failed ({reason}).")
            manifest_df = None

    if manifest_df is None or len(manifest_df) == 0:
        if config.CELEBDF_ROOT.exists():
            print(f"--> Regenerating Celeb-DF manifest from: {config.CELEBDF_ROOT}")
            manifest_df = generate_celebdf_manifest(config.CELEBDF_ROOT, config.CELEBDF_MANIFEST_PATH)
            is_valid, reason = validate_celebdf_manifest(manifest_df)
            if not is_valid:
                raise ValueError(f"Regenerated Celeb-DF manifest is invalid ({reason}).")
        else:
            raise ValueError(f"Valid Celeb-DF manifest not found at '{config.CELEBDF_MANIFEST_PATH}' and dataset root '{config.CELEBDF_ROOT}' does not exist.")

    # Filter to official test split if 'split' column is present
    if "split" in manifest_df.columns:
        test_df = manifest_df[manifest_df["split"] == "test"].copy()
    else:
        test_df = manifest_df.copy()

    if len(test_df) == 0:
        raise ValueError("Celeb-DF test dataset is empty after split filtering.")

    if test_df["label"].nunique() < 2:
        raise ValueError(f"Celeb-DF test set requires samples from both classes, but found labels: {test_df['label'].unique()}")

    n_vids = test_df["video_id"].nunique() if "video_id" in test_df.columns else len(test_df)
    print(f"Total manifest samples: {len(manifest_df):,}")
    print(f"Official test samples:  {len(test_df):,}")
    print(f"Number of test videos:  {n_vids:,}")
    print(f"Test class counts:      {test_df['label'].value_counts().to_dict()}")

    _, eval_transform = get_transforms()
    dataset = DeepfakeImageDataset(test_df, eval_transform)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    criterion = DeepfakeLoss(loss_type="bce", smoothing=config.LABEL_SMOOTHING)

    variant_suffix = f"_{config.MODEL_VARIANT}" if getattr(config, "MODEL_VARIANT", "fusion") != "fusion" else ""

    # Standard Model
    std_ckpt_path = config.OUTPUT_ROOT / f"{config.MODEL_NAME}{variant_suffix}_clean" / "best_model.pt"
    if not std_ckpt_path.exists():
        std_ckpt_path = config.OUTPUT_ROOT / f"{config.MODEL_NAME}{variant_suffix}_standard" / "best_model.pt"
    if not std_ckpt_path.exists():
        std_ckpt_path = config.OUTPUT_ROOT / f"{config.MODEL_NAME}_clean" / "best_model.pt"

    if not std_ckpt_path.exists():
        raise FileNotFoundError(f"Standard model checkpoint not found at: {std_ckpt_path}. Please train standard model first.")
    
    std_model, std_thresh, std_ckpt_data = load_model_from_checkpoint(std_ckpt_path, device)
    std_metrics, std_preds = evaluate_model(std_model, dataloader, criterion, device, limit_batches=limit_batches)
    std_vid = compute_video_level_metrics(std_preds, threshold=std_thresh)

    # Robustness Model
    rob_ckpt_path = config.OUTPUT_ROOT / f"{config.MODEL_NAME}{variant_suffix}_degradation" / "best_model.pt"
    if not rob_ckpt_path.exists():
        rob_ckpt_path = config.OUTPUT_ROOT / f"{config.MODEL_NAME}_degradation" / "best_model.pt"

    if not rob_ckpt_path.exists():
        raise FileNotFoundError(f"Robustness model checkpoint not found at: {rob_ckpt_path}. Please train degradation model first.")

    rob_model, rob_thresh, rob_ckpt_data = load_model_from_checkpoint(rob_ckpt_path, device)
    rob_metrics, rob_preds = evaluate_model(rob_model, dataloader, criterion, device, limit_batches=limit_batches)
    rob_vid = compute_video_level_metrics(rob_preds, threshold=rob_thresh)

    print("\n" + "="*90)
    print("        CELEB-DF V2 CROSS-DATASET GENERALIZATION BENCHMARK        ")
    print("="*90)
    print(f"Standard Model   | Frame AUC: {std_metrics['roc_auc']*100:.2f}% | Video AUC: {std_vid['video_roc_auc']*100:.2f}% | Video Acc: {std_vid['video_accuracy']*100:.2f}% | Video F1: {std_vid['video_f1']*100:.2f}%")
    print(f"Robustness Model | Frame AUC: {rob_metrics['roc_auc']*100:.2f}% | Video AUC: {rob_vid['video_roc_auc']*100:.2f}% | Video Acc: {rob_vid['video_accuracy']*100:.2f}% | Video F1: {rob_vid['video_f1']*100:.2f}%")
    print("="*90)


def main():
    parser = argparse.ArgumentParser(description="Unified Deepfake Detection Evaluation Harness")
    parser.add_argument("--mode", type=str, default="comparative", choices=["comparative", "celebdf"], help="Evaluation mode")
    parser.add_argument("--celebdf", action="store_true", help="Shortcut flag for Celeb-DF evaluation")
    parser.add_argument("--bootstraps", type=int, default=1000, help="Number of bootstrap resamples for CIs")
    parser.add_argument("--limit_batches", type=int, default=None, help="Limit number of batches for quick validation")
    args = parser.parse_args()

    if args.celebdf or args.mode == "celebdf":
        run_celebdf_eval(limit_batches=args.limit_batches)
    else:
        run_comparative_benchmark(n_bootstraps=args.bootstraps, limit_batches=args.limit_batches)


if __name__ == "__main__":
    main()