import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from deepfake_robustness.configs import config
from deepfake_robustness.degradations.transforms import RandomJPEGCompression, RandomDownscaleRestore, get_transforms
from deepfake_robustness.evaluation.evaluate import load_model_from_checkpoint

def calculate_ssim(img1, img2):
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    return ssim(arr1, arr2, channel_axis=-1, data_range=255)

def apply_pipeline(img, ops):
    res = img.copy()
    for op in ops:
        res = op(res)
    return res

def tune_pipeline_b(orig_img, pipeline_a_img, base_scale):
    ssim_a = calculate_ssim(orig_img, pipeline_a_img)
    best_q = 50
    best_diff = 1.0
    best_img = None
    
    downscale_op = RandomDownscaleRestore(scales=[base_scale], probability=1.0)
    img_downscaled = downscale_op(orig_img)
    
    low, high = 10, 100
    for _ in range(10):
        if low > high:
            break
        mid = (low + high) // 2
        jpeg_op = RandomJPEGCompression(quality_range=(mid, mid), probability=1.0)
        img_b = jpeg_op(img_downscaled)
        
        ssim_b = calculate_ssim(orig_img, img_b)
        diff = ssim_b - ssim_a
        
        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_q = mid
            best_img = img_b
            
        if abs(diff) <= 0.01:
            break
            
        if diff < 0:
            low = mid + 1
        else:
            high = mid - 1
            
    return best_img, best_q, best_diff

def run_phase1():
    print("[Phase 1] Starting Go/No-Go Measurement")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    manifest_df = pd.read_csv(config.MANIFEST_PATH)
    if "split" in manifest_df.columns:
        manifest_df = manifest_df[manifest_df["split"] == "val"]
    frames = manifest_df['image_path'].tolist()
    random.seed(config.SEED)
    random.shuffle(frames)
    frames = frames[:5000]
    
    print(f"Sampled {len(frames)} frames for Go/No-Go test.")
    
    ckpt_path = config.OUTPUT_ROOT / "efficientnet_b0_degradation" / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = config.OUTPUT_ROOT / "efficientnet_b0_clean" / "best_model.pt"
    model, _, _ = load_model_from_checkpoint(ckpt_path, device)
    model.eval()
    
    _, eval_transform = get_transforms()
    
    features_X = []
    labels_y = []
    
    jpeg_a = RandomJPEGCompression(quality_range=(70, 70), probability=1.0)
    downscale_a = RandomDownscaleRestore(scales=[0.75], probability=1.0)
    pipeline_a_ops = [jpeg_a, downscale_a]
    
    for idx, path in enumerate(tqdm(frames)):
        orig_img = Image.open(path).convert("RGB")
        img_a = apply_pipeline(orig_img, pipeline_a_ops)
        img_b, _, _ = tune_pipeline_b(orig_img, img_a, base_scale=0.75)
        
        t_a = eval_transform(img_a).unsqueeze(0).to(device)
        t_b = eval_transform(img_b).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, feat_a = model(t_a)
            _, feat_b = model(t_b)
            
        features_X.append(feat_a.cpu().numpy().flatten())
        labels_y.append(0)
        
        features_X.append(feat_b.cpu().numpy().flatten())
        labels_y.append(1)
        
    features_X = np.array(features_X)
    labels_y = np.array(labels_y)
    
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, features_X, labels_y, cv=5, scoring='roc_auc')
    auc_mean, auc_std = scores.mean(), scores.std()
    
    print(f"Binary Probe ROC-AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    if auc_mean - 2*auc_std > 0.5:
        print("✅ Go/No-Go Gate Passed")
    else:
        print("❌ Go/No-Go Gate Failed")

if __name__ == "__main__":
    run_phase1()
