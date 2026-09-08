"""
Grad-CAM (Gradient-weighted Class Activation Mapping) interpretability tooling
for the dual-branch (RGB + SRM/frequency) deepfake detector defined in model.py.

This module is an EVALUATION/INTERPRETABILITY layer only. It does not change the
model architecture, training objective, training strategy, data splits, or
evaluation metrics used by train.py / evaluate.py. It only reads already-trained
checkpoints and visualizes them.

--------------------------------------------------------------------------------
Method (Selvaraju et al., 2017)
--------------------------------------------------------------------------------
For a chosen convolutional layer producing activations A (shape [C, h, w]) and a
scalar target score y (a pre-sigmoid logit here, see "Binary classification"
below), Grad-CAM computes:

    alpha_k = (1 / (h * w)) * sum_{i,j} (dy / dA_k[i, j])      # global-avg-pooled gradient
    GradCAM = ReLU( sum_k alpha_k * A_k )                       # weighted combination + ReLU

alpha_k measures how strongly channel k's activation map influences the target
score; the ReLU keeps only evidence that positively supports the target class
(negative contributions are suppressed, matching the original paper). The result
is a single-channel, low-resolution localization map that is then upsampled to
the input image resolution and min-max normalized to [0, 1] for visualization.

--------------------------------------------------------------------------------
Binary classification handling
--------------------------------------------------------------------------------
DeepfakeModel (see model.py) is a SINGLE-LOGIT binary classifier: `EnhancedHead`
ends in `nn.Linear(512, 1)` and the model returns that raw logit directly (no
sigmoid applied internally). Training/evaluation apply
`torch.sigmoid(logit)` to get P(fake) (see train.evaluate_model). Consequently:

  - target_class="fake": y = logit               (higher logit -> more "fake")
  - target_class="real": y = -logit               (since P(real) = 1 - sigmoid(logit),
                                                     the "real" score is the negation
                                                     of the fake logit, not a separate
                                                     output unit)

Grad-CAM backpropagates from the raw logit (pre-sigmoid), not from the
thresholded class decision or the sigmoid probability, as recommended by the
original Grad-CAM formulation.

--------------------------------------------------------------------------------
Target layer selection
--------------------------------------------------------------------------------
See `get_target_layer()`. In short: the RGB branch target is the final RGB
convolutional feature map that actually gets globally pooled before the
classifier head - which is the raw backbone output for `rgb_only` /
`fusion_no_attn`, or the cross-attention (SFCA) output for the `fusion` variant
(since attention runs *after* the backbone but *before* pooling in that variant).
The frequency/SRM branch target is the final MS-SRM CNN block (`freq_convs`),
which is genuinely convolutional and differentiable, so Grad-CAM applies to it
identically - it is not a different, ad-hoc "frequency visualization" method.

--------------------------------------------------------------------------------
IMPORTANT METHODOLOGICAL CAUTIONS - read before interpreting any Grad-CAM output
--------------------------------------------------------------------------------
1. Grad-CAM shows which spatial regions influenced the model's output score. It
   does NOT prove the model has located the true physical manipulation region.
2. A Grad-CAM heatmap is not a segmentation mask and should never be scored
   against ground-truth manipulation masks as if it were one.
3. Grad-CAM is a QUALITATIVE, exploratory diagnostic. It must not replace or be
   conflated with the project's quantitative robustness evaluation (accuracy,
   F1, ROC-AUC, ECE, bootstrap CIs - see evaluate.py / metrics_utils.py).
4. Heatmaps are sensitive to preprocessing, the chosen target layer, gradient
   noise, and model confidence. Two visually different heatmaps do not
   necessarily imply a meaningful difference in the underlying decision logic.
5. Comparing clean-vs-degraded or standard-vs-robust heatmaps (this module's
   comparison modes, and `compare_gradcam_maps()`) is exploratory unless
   accompanied by a clearly justified quantitative similarity protocol
   (sample size, aggregation, statistical test) - a single-image cosine
   similarity / correlation / SSIM number is a descriptive aid, not a
   statistical claim.

This module never changes training behavior: it only loads existing checkpoints
(via evaluate.load_model_from_checkpoint) and runs forward/backward passes for
visualization purposes.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from src.configs import config
from src.evaluation.evaluate import apply_advanced_tier_distortion, load_model_from_checkpoint
from src.degradations.transforms import get_transforms


# ---------------------------------------------------------------------------
# 1. Target layer selection
# ---------------------------------------------------------------------------

def get_target_layer(model, branch="rgb"):
    """Return the nn.Module whose output activations Grad-CAM should hook into.

    Inspects the actual instantiated submodules of `model` (a DeepfakeModel) rather
    than assuming layer names, so this works correctly for whichever branch_mode /
    model_variant combination the checkpoint was actually trained with.

    branch="rgb": the final RGB convolutional feature map that is globally pooled
        immediately before the classifier head.
          - model_variant == "fusion" (cross-attention active): this is the
            Spatial-Frequency Cross-Attention output (`model.sfca`), because in
            that forward path `attended_rgb_map = sfca(rgb_map, freq_map)` is what
            gets pooled, not the raw backbone map.
          - otherwise (rgb_only / fusion_no_attn, or any variant where sfca was not
            built): this is the raw RGB backbone output (`model.rgb_backbone`),
            which for an EfficientNet-B0 RGB branch is precisely "the final
            meaningful convolutional feature map before global pooling".

    branch="freq": the final block of the MS-SRM frequency CNN (`model.freq_convs`),
        which is likewise pooled directly and fed to the classifier head. This is
        a genuine Grad-CAM target (same activation-gradient method), not a
        different visualization technique.
    """
    branch = branch.lower()
    if branch == "rgb":
        if model.rgb_backbone is None:
            raise ValueError(
                f"Model (branch_mode='{model.branch_mode}', model_variant='{model.model_variant}') "
                "has no RGB backbone; branch='rgb' Grad-CAM is not applicable."
            )
        if model.model_variant == "fusion" and model.sfca is not None:
            return model.sfca
        return model.rgb_backbone
    elif branch == "freq":
        if not hasattr(model, "freq_convs"):
            raise ValueError(
                f"Model (branch_mode='{model.branch_mode}', model_variant='{model.model_variant}') "
                "has no frequency/SRM branch; branch='freq' Grad-CAM is not applicable."
            )
        return model.freq_convs
    else:
        raise ValueError(f"Unknown branch '{branch}'. Expected 'rgb' or 'freq'.")


# ---------------------------------------------------------------------------
# 2. Core Grad-CAM engine
# ---------------------------------------------------------------------------

class GradCAM:
    """Reusable Grad-CAM engine for a single (model, target_layer) pair.

    Usage:
        with GradCAM(model, target_layer) as engine:
            cam_batch, fake_prob_batch = engine.generate(input_tensor, target_class="fake")

    Hooks are registered on __enter__ and removed on __exit__, so the model is
    left exactly as it was found (no permanent forward/backward hooks remain
    attached). `generate()` also restores the model's train/eval mode and clears
    any populated `.grad` tensors afterward.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_handle = None
        self._bwd_handle = None

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __enter__(self):
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        return False

    def remove_hooks(self):
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None

    def generate(self, input_tensor, target_class="fake", output_size=None):
        """Run one forward+backward pass and return (cam_batch, fake_prob_batch).

        cam_batch: float32 numpy array [B, H, W] in [0, 1] (H, W = output_size, or
            the input's own spatial size if output_size is None).
        fake_prob_batch: float32 numpy array [B] with sigmoid(logit) per sample,
            regardless of which target_class the CAM was computed for.
        """
        if self._fwd_handle is None or self._bwd_handle is None:
            raise RuntimeError(
                "GradCAM hooks are not registered. Use `with GradCAM(model, layer) as engine:`."
            )
        if target_class not in ("fake", "real"):
            raise ValueError(f"Unsupported target_class '{target_class}'. Expected 'fake' or 'real'.")

        was_training = self.model.training
        device = next(self.model.parameters()).device

        try:
            self.model.eval()
            self.model.zero_grad(set_to_none=True)
            self.activations = None
            self.gradients = None

            with torch.enable_grad():
                x = input_tensor.to(device).clone().detach().requires_grad_(True)
                logits, _ = self.model(x)
                logits = logits.squeeze(1)  # single-logit binary classifier: [B]

                score = logits if target_class == "fake" else -logits
                score.sum().backward()

                fake_prob = torch.sigmoid(logits.detach())

            if self.activations is None or self.gradients is None:
                raise RuntimeError(
                    "Grad-CAM hooks did not fire during the forward/backward pass - "
                    "the target_layer is not on the active forward path for this "
                    f"model (branch_mode='{self.model.branch_mode}', "
                    f"model_variant='{self.model.model_variant}')."
                )

            activations = self.activations.detach()  # A_k: [B, C, h, w]
            gradients = self.gradients.detach()       # dy/dA_k: [B, C, h, w]

            # alpha_k = mean over spatial positions of dy/dA_k
            alpha = gradients.mean(dim=(2, 3), keepdim=True)
            # GradCAM = ReLU(sum_k alpha_k * A_k)
            cam = torch.relu((alpha * activations).sum(dim=1, keepdim=True))  # [B, 1, h, w]

            if output_size is None:
                output_size = tuple(input_tensor.shape[-2:])
            cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
            cam = cam.squeeze(1)  # [B, H, W]

            cam = _safe_normalize(cam)

            return cam.cpu().numpy().astype(np.float32), fake_prob.cpu().numpy().astype(np.float32)
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)


def _safe_normalize(cam):
    """Min-max normalize each sample of a [B, H, W] CAM tensor to [0, 1].

    Safe against degenerate/constant activations: if max == min for a sample, the
    denominator is clamped away from zero so the result is a finite all-zero map
    instead of NaN/Inf.
    """
    b = cam.shape[0]
    flat = cam.reshape(b, -1)
    cmin = flat.min(dim=1, keepdim=True)[0]
    cmax = flat.max(dim=1, keepdim=True)[0]
    denom = (cmax - cmin).clamp(min=1e-8)
    norm = (flat - cmin) / denom
    return norm.reshape(cam.shape)


def generate_gradcam(model, image, target_layer, target_class="fake"):
    """Reusable, UI-friendly Grad-CAM entry point.

    This is the function a future interactive demo (upload image, adjust
    compression/resizing, compare predictions) should call directly.

    Args:
        model: a DeepfakeModel instance (any branch_mode/model_variant).
        image: a preprocessed input tensor, shape [3, H, W] or [1, 3, H, W],
            normalized EXACTLY like standard inference preprocessing (see
            transforms.get_transforms()'s evaluation_transform / this module's
            `load_and_preprocess_image`). This function does not re-derive
            preprocessing so callers stay in full control of the input pipeline.
        target_layer: an nn.Module, typically from `get_target_layer(model, branch)`.
        target_class: "fake" or "real" (see module docstring for the single-logit
            binary-classification handling).

    Returns:
        (cam, fake_probability): cam is a numpy [H, W] float32 array in [0, 1];
        fake_probability is a python float, sigmoid(logit) for the input.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    with GradCAM(model, target_layer) as engine:
        cam_batch, fake_prob_batch = engine.generate(image, target_class=target_class)
    return cam_batch[0], float(fake_prob_batch[0])


# ---------------------------------------------------------------------------
# 3. Preprocessing (reuses the project's existing pipelines - no second,
#    inconsistent preprocessing/degradation implementation)
# ---------------------------------------------------------------------------

def load_and_preprocess_image(image_path, degradation_tier=None):
    """Load a face image and preprocess it exactly like standard inference.

    Uses transforms.get_transforms()'s evaluation_transform (Resize -> ToTensor ->
    Normalize) - the same transform train.py/evaluate.py use at eval time. If
    `degradation_tier` names a key in config.DEGRADATION_TIERS, the PIL image is
    first degraded via evaluate.apply_advanced_tier_distortion (the project's
    single deterministic degradation implementation used for the tier benchmark
    matrix), matching how evaluate.py's TierDistortionModifier applies it.

    Returns (pil_image, input_tensor) where pil_image is the (possibly degraded)
    RGB PIL.Image actually fed into preprocessing, and input_tensor is a
    normalized [1, 3, H, W] tensor.
    """
    pil_image = Image.open(image_path).convert("RGB")

    if degradation_tier is not None:
        if degradation_tier not in config.DEGRADATION_TIERS:
            raise ValueError(
                f"Unknown degradation tier '{degradation_tier}'. "
                f"Available: {sorted(config.DEGRADATION_TIERS.keys())}"
            )
        tier_cfg = config.DEGRADATION_TIERS[degradation_tier]
        pil_image = apply_advanced_tier_distortion(
            pil_image, tier_cfg, sample_id=str(image_path), global_seed=config.SEED
        )

    _, eval_transform = get_transforms()
    input_tensor = eval_transform(pil_image).unsqueeze(0)
    return pil_image, input_tensor


def predict(model, input_tensor, threshold=0.5):
    """Run a plain (no-grad) forward pass and return (predicted_class, fake_probability)."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits, _ = model(input_tensor.to(device))
        fake_prob = torch.sigmoid(logits.squeeze(1)).item()
    model.train(was_training)
    predicted_class = "fake" if fake_prob >= threshold else "real"
    return predicted_class, fake_prob


# ---------------------------------------------------------------------------
# 4. Visualization helpers
# ---------------------------------------------------------------------------

def _cam_to_heatmap_image(cam_2d):
    """Map a normalized [0, 1] CAM array to an RGB heatmap PIL image (jet colormap)."""
    heatmap_u8 = np.uint8(255 * np.clip(cam_2d, 0.0, 1.0))
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap_rgb)


def overlay_heatmap(pil_image, cam_2d, alpha=0.45):
    """Resize a CAM to the image's native resolution and blend it as a heatmap overlay.

    Returns (overlay_image, heatmap_image), both RGB PIL Images at the original
    image's resolution.
    """
    base = pil_image.convert("RGB")
    w, h = base.size
    cam_resized = cv2.resize(cam_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_img = _cam_to_heatmap_image(cam_resized)
    overlay_img = Image.blend(base, heatmap_img, alpha=alpha)
    return overlay_img, heatmap_img


def _save_panel_figure(panels, output_path, suptitle=None):
    """Save a single row of (title, PIL image) panels to output_path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _json_safe(obj):
    """Recursively replace non-finite floats (NaN/Inf) with None so output is strictly valid JSON."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _save_metadata(metadata, output_path):
    """Write a JSON metadata sidecar next to output_path (same stem, .json suffix).

    NaN/Inf similarity values (e.g. cosine similarity of two all-zero degenerate
    CAMs) are written as JSON `null` rather than the non-standard `NaN` token, so
    the file is portable to strict JSON parsers/report tooling.
    """
    output_path = Path(output_path)
    meta_path = output_path.with_suffix(".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(metadata), f, indent=2, default=str, allow_nan=False)
    return meta_path


# ---------------------------------------------------------------------------
# 5. Optional quantitative clean-vs-degraded / standard-vs-robust comparison
#    (EXPLORATORY ONLY - see module docstring caution #5)
# ---------------------------------------------------------------------------

def compare_gradcam_maps(cam_a, cam_b):
    """Exploratory similarity metrics between two same-shape normalized CAM arrays.

    Returns a dict with cosine_similarity, pearson_correlation, and a windowed
    SSIM approximation (computed with scipy.ndimage.uniform_filter rather than
    scikit-image, which is not a project dependency). These are descriptive aids
    for a single image pair, NOT a statistical claim - see module docstring
    caution #5. They are not used anywhere in the main quantitative robustness
    evaluation (evaluate.py / metrics_utils.py).
    """
    a = np.asarray(cam_a, dtype=np.float64)
    b = np.asarray(cam_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"CAM shape mismatch: {a.shape} vs {b.shape}")

    flat_a, flat_b = a.ravel(), b.ravel()
    denom = np.linalg.norm(flat_a) * np.linalg.norm(flat_b)
    cosine = float(np.dot(flat_a, flat_b) / denom) if denom > 1e-12 else float("nan")

    if np.std(flat_a) < 1e-12 or np.std(flat_b) < 1e-12:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(flat_a, flat_b)[0, 1])

    ssim = _windowed_ssim(a, b)

    return {"cosine_similarity": cosine, "pearson_correlation": pearson, "ssim": ssim}


def _windowed_ssim(a, b, window=7):
    """Lightweight windowed SSIM for two [0, 1]-range 2D arrays of identical shape."""
    from scipy.ndimage import uniform_filter

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    mu_a = uniform_filter(a, window)
    mu_b = uniform_filter(b, window)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a2 = uniform_filter(a * a, window) - mu_a2
    sigma_b2 = uniform_filter(b * b, window) - mu_b2
    sigma_ab = uniform_filter(a * b, window) - mu_ab

    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    ssim_map = num / den
    return float(np.clip(ssim_map, -1.0, 1.0).mean())


# ---------------------------------------------------------------------------
# 6. Orchestration: the three CLI run modes
# ---------------------------------------------------------------------------

def _compute_visualization(model, threshold, image_path, branch="rgb", target_class="predicted", degradation_tier=None):
    pil_image, input_tensor = load_and_preprocess_image(image_path, degradation_tier=degradation_tier)
    predicted_class, fake_prob = predict(model, input_tensor, threshold=threshold)
    resolved_target_class = predicted_class if target_class == "predicted" else target_class

    target_layer = get_target_layer(model, branch=branch)
    cam, _ = generate_gradcam(model, input_tensor, target_layer, target_class=resolved_target_class)
    overlay_img, heatmap_img = overlay_heatmap(pil_image, cam)

    return {
        "pil_image": pil_image,
        "cam": cam,
        "heatmap_img": heatmap_img,
        "overlay_img": overlay_img,
        "predicted_class": predicted_class,
        "fake_probability": fake_prob,
        "target_class": resolved_target_class,
        "target_layer": target_layer,
    }


def run_single_visualization(checkpoint_path, image_path, output_path, branch="rgb",
                              target_class="predicted", degradation_tier=None, device=None):
    """Original face | Grad-CAM heatmap | Heatmap overlay, for a single checkpoint/image."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, threshold, ckpt = load_model_from_checkpoint(Path(checkpoint_path), device)

    res = _compute_visualization(
        model, threshold, image_path, branch=branch, target_class=target_class, degradation_tier=degradation_tier
    )

    suptitle = (
        f"Predicted: {res['predicted_class']} | Fake prob: {res['fake_probability']:.3f} | "
        f"Threshold: {threshold:.3f} | Branch: {branch} | Variant: {model.model_variant}"
    )
    _save_panel_figure(
        [
            ("Original face", res["pil_image"]),
            ("Grad-CAM heatmap", res["heatmap_img"]),
            ("Heatmap overlay", res["overlay_img"]),
        ],
        output_path,
        suptitle=suptitle,
    )

    cfg = ckpt.get("configuration", {}) if isinstance(ckpt.get("configuration", {}), dict) else {}
    metadata = {
        "image_path": str(image_path),
        "checkpoint": str(checkpoint_path),
        "model_variant": model.model_variant,
        "branch_mode": model.branch_mode,
        "training_strategy": cfg.get("training_strategy", ckpt.get("training_strategy", "unknown")),
        "degradation": degradation_tier or "clean",
        "predicted_class": res["predicted_class"],
        "fake_probability": res["fake_probability"],
        "threshold": threshold,
        "target_layer": f"{branch}:{res['target_layer'].__class__.__name__}",
        "target_class_for_cam": res["target_class"],
        "branch": branch,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = _save_metadata(metadata, output_path)

    print(f"Predicted class:  {res['predicted_class']}")
    print(f"Fake probability: {res['fake_probability']:.4f}")
    print(f"Threshold used:   {threshold:.4f}")
    print(f"Target layer:     {branch} -> {res['target_layer'].__class__.__name__}")
    print(f"Model variant:    {model.model_variant}")
    print(f"Saved panel to:   {output_path}")
    print(f"Saved metadata to: {meta_path}")
    return metadata


def run_degradation_comparison(checkpoint_path, image_path, output_path, degradation_tier,
                                branch="rgb", target_class="predicted", device=None):
    """Clean | Clean Grad-CAM | <tier> | <tier> Grad-CAM, for a single checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, threshold, ckpt = load_model_from_checkpoint(Path(checkpoint_path), device)

    clean_res = _compute_visualization(model, threshold, image_path, branch=branch, target_class=target_class, degradation_tier=None)
    degraded_res = _compute_visualization(model, threshold, image_path, branch=branch, target_class=target_class, degradation_tier=degradation_tier)

    _save_panel_figure(
        [
            (f"Clean\n{clean_res['predicted_class']} ({clean_res['fake_probability']:.3f})", clean_res["pil_image"]),
            ("Clean Grad-CAM", clean_res["overlay_img"]),
            (f"{degradation_tier}\n{degraded_res['predicted_class']} ({degraded_res['fake_probability']:.3f})", degraded_res["pil_image"]),
            (f"{degradation_tier} Grad-CAM", degraded_res["overlay_img"]),
        ],
        output_path,
        suptitle=f"{Path(checkpoint_path).parent.name} | branch={branch} | target={target_class} (exploratory - see module docstring)",
    )

    similarity = compare_gradcam_maps(clean_res["cam"], degraded_res["cam"])

    cfg = ckpt.get("configuration", {}) if isinstance(ckpt.get("configuration", {}), dict) else {}
    metadata = {
        "image_path": str(image_path),
        "checkpoint": str(checkpoint_path),
        "model_variant": model.model_variant,
        "branch_mode": model.branch_mode,
        "training_strategy": cfg.get("training_strategy", ckpt.get("training_strategy", "unknown")),
        "branch": branch,
        "degradation_a": "clean",
        "degradation_b": degradation_tier,
        "predicted_class_a": clean_res["predicted_class"],
        "fake_probability_a": clean_res["fake_probability"],
        "predicted_class_b": degraded_res["predicted_class"],
        "fake_probability_b": degraded_res["fake_probability"],
        "threshold": threshold,
        "target_layer": f"{branch}:{clean_res['target_layer'].__class__.__name__}",
        "heatmap_similarity_clean_vs_degraded": similarity,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = _save_metadata(metadata, output_path)
    print(f"Saved comparison panel to: {output_path}")
    print(f"Saved metadata to: {meta_path}")
    return metadata


def run_model_comparison(standard_checkpoint, robust_checkpoint, image_path, output_path,
                          branch="rgb", target_class="predicted", degradation_tier=None, device=None):
    """Input | Standard model Grad-CAM | Robust model Grad-CAM, optionally on a degraded input."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    std_model, std_thresh, _ = load_model_from_checkpoint(Path(standard_checkpoint), device)
    rob_model, rob_thresh, _ = load_model_from_checkpoint(Path(robust_checkpoint), device)

    std_res = _compute_visualization(std_model, std_thresh, image_path, branch=branch, target_class=target_class, degradation_tier=degradation_tier)
    rob_res = _compute_visualization(rob_model, rob_thresh, image_path, branch=branch, target_class=target_class, degradation_tier=degradation_tier)

    _save_panel_figure(
        [
            (f"Input\n({degradation_tier or 'clean'})", std_res["pil_image"]),
            (f"Standard model\n{std_res['predicted_class']} ({std_res['fake_probability']:.3f})", std_res["overlay_img"]),
            (f"Robust model\n{rob_res['predicted_class']} ({rob_res['fake_probability']:.3f})", rob_res["overlay_img"]),
        ],
        output_path,
        suptitle=f"branch={branch} | target={target_class} | degradation={degradation_tier or 'clean'} (exploratory - see module docstring)",
    )

    similarity = compare_gradcam_maps(std_res["cam"], rob_res["cam"])

    metadata = {
        "image_path": str(image_path),
        "standard_checkpoint": str(standard_checkpoint),
        "robust_checkpoint": str(robust_checkpoint),
        "branch": branch,
        "degradation": degradation_tier or "clean",
        "standard_model_variant": std_model.model_variant,
        "robust_model_variant": rob_model.model_variant,
        "standard_predicted_class": std_res["predicted_class"],
        "standard_fake_probability": std_res["fake_probability"],
        "standard_threshold": std_thresh,
        "robust_predicted_class": rob_res["predicted_class"],
        "robust_fake_probability": rob_res["fake_probability"],
        "robust_threshold": rob_thresh,
        "target_layer": f"{branch}:{std_res['target_layer'].__class__.__name__}",
        "heatmap_similarity_standard_vs_robust": similarity,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = _save_metadata(metadata, output_path)
    print(f"Saved model-comparison panel to: {output_path}")
    print(f"Saved metadata to: {meta_path}")
    return metadata


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM interpretability tool for the deepfake detector.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained model checkpoint (e.g. best_model.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to an input face image")
    parser.add_argument("--output", type=str, required=True, help="Output path for the visualization panel (PNG)")
    parser.add_argument("--branch", type=str, default="rgb", choices=["rgb", "freq"], help="Which branch to visualize")
    parser.add_argument("--target-class", type=str, default="predicted", choices=["predicted", "fake", "real"],
                         help="Score to backpropagate from: the model's own predicted class, or force 'fake'/'real'")
    parser.add_argument("--degradation", type=str, default=None, choices=list(config.DEGRADATION_TIERS.keys()),
                         help="Apply a named degradation tier (from config.DEGRADATION_TIERS) before visualizing")
    parser.add_argument("--compare-degradation", action="store_true",
                         help="Render a clean-vs-degraded comparison panel for --checkpoint (requires --degradation)")
    parser.add_argument("--compare-checkpoint", type=str, default=None,
                         help="A second checkpoint (e.g. robustness-aware model) to compare side-by-side against --checkpoint")
    args = parser.parse_args()

    if args.compare_degradation and not args.degradation:
        parser.error("--compare-degradation requires --degradation to also be specified.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.compare_checkpoint:
        run_model_comparison(
            standard_checkpoint=args.checkpoint,
            robust_checkpoint=args.compare_checkpoint,
            image_path=args.image,
            output_path=args.output,
            branch=args.branch,
            target_class=args.target_class,
            degradation_tier=args.degradation,
            device=device,
        )
    elif args.compare_degradation:
        run_degradation_comparison(
            checkpoint_path=args.checkpoint,
            image_path=args.image,
            output_path=args.output,
            degradation_tier=args.degradation,
            branch=args.branch,
            target_class=args.target_class,
            device=device,
        )
    else:
        run_single_visualization(
            checkpoint_path=args.checkpoint,
            image_path=args.image,
            output_path=args.output,
            branch=args.branch,
            target_class=args.target_class,
            degradation_tier=args.degradation,
            device=device,
        )


if __name__ == "__main__":
    main()
