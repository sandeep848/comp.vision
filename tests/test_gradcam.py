"""
Unit tests for gradcam.py Grad-CAM interpretability tooling.

Uses only synthetic (random-init, pretrained=False) models and synthetic images -
no FaceForensics++/Celeb-DF dataset access is required.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from deepfake_robustness.configs import config
from deepfake_robustness.models.model import build_model
from deepfake_robustness.degradations.transforms import get_transforms
from deepfake_robustness.evaluation.gradcam import (
    GradCAM,
    get_target_layer,
    generate_gradcam,
    load_and_preprocess_image,
    predict,
    _safe_normalize,
    run_single_visualization,
)
from deepfake_robustness.evaluation.evaluate import load_model_from_checkpoint


def _make_input(size=256, batch=1):
    return torch.randn(batch, 3, size, size)


# ---------------------------------------------------------------------------
# Target layer resolution
# ---------------------------------------------------------------------------

def test_get_target_layer_rgb_only():
    model = build_model("efficientnet_b0", pretrained=False, model_variant="rgb_only")
    layer = get_target_layer(model, branch="rgb")
    assert layer is model.rgb_backbone

    with pytest.raises(ValueError):
        get_target_layer(model, branch="freq")


def test_get_target_layer_fusion_no_attn():
    model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion_no_attn")
    rgb_layer = get_target_layer(model, branch="rgb")
    assert rgb_layer is model.rgb_backbone

    freq_layer = get_target_layer(model, branch="freq")
    assert freq_layer is model.freq_convs


def test_get_target_layer_fusion_uses_attention_output():
    model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion")
    rgb_layer = get_target_layer(model, branch="rgb")
    # In the attention-fusion variant, the SFCA output (not the raw backbone map)
    # is what actually gets pooled before the classifier head.
    assert rgb_layer is model.sfca

    freq_layer = get_target_layer(model, branch="freq")
    assert freq_layer is model.freq_convs


def test_get_target_layer_invalid_branch():
    model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion")
    with pytest.raises(ValueError):
        get_target_layer(model, branch="nonsense")


# ---------------------------------------------------------------------------
# Core Grad-CAM correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["rgb_only", "fusion", "fusion_no_attn"])
def test_gradcam_rgb_branch_shapes_and_values(variant):
    model = build_model("efficientnet_b0", pretrained=False, model_variant=variant)
    target_layer = get_target_layer(model, branch="rgb")
    x = _make_input(size=128)

    cam, fake_prob = generate_gradcam(model, x, target_layer, target_class="fake")

    assert cam.shape == (128, 128)
    assert np.isfinite(cam).all()
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert isinstance(fake_prob, float)
    assert np.isfinite(fake_prob)


@pytest.mark.parametrize("variant", ["fusion", "fusion_no_attn"])
def test_gradcam_freq_branch_shapes_and_values(variant):
    model = build_model("efficientnet_b0", pretrained=False, model_variant=variant)
    target_layer = get_target_layer(model, branch="freq")
    x = _make_input(size=128)

    cam, fake_prob = generate_gradcam(model, x, target_layer, target_class="fake")

    assert cam.shape == (128, 128)
    assert np.isfinite(cam).all()
    assert cam.min() >= 0.0 and cam.max() <= 1.0


def test_gradcam_real_target_class():
    model = build_model("efficientnet_b0", pretrained=False, model_variant="rgb_only")
    target_layer = get_target_layer(model, branch="rgb")
    x = _make_input(size=64)

    cam_fake, _ = generate_gradcam(model, x, target_layer, target_class="fake")
    cam_real, _ = generate_gradcam(model, x, target_layer, target_class="real")

    assert cam_fake.shape == cam_real.shape == (64, 64)
    assert np.isfinite(cam_fake).all() and np.isfinite(cam_real).all()


def test_gradcam_invalid_target_class_raises():
    model = build_model("efficientnet_b0", pretrained=False, model_variant="rgb_only")
    target_layer = get_target_layer(model, branch="rgb")
    x = _make_input(size=64)
    with pytest.raises(ValueError):
        generate_gradcam(model, x, target_layer, target_class="not_a_class")


def test_gradcam_batch_dimension_squeezed_input_supported():
    """generate_gradcam should accept a [3, H, W] tensor (no explicit batch dim)."""
    model = build_model("efficientnet_b0", pretrained=False, model_variant="rgb_only")
    target_layer = get_target_layer(model, branch="rgb")
    x = torch.randn(3, 64, 64)
    cam, fake_prob = generate_gradcam(model, x, target_layer, target_class="fake")
    assert cam.shape == (64, 64)


# ---------------------------------------------------------------------------
# Hook lifecycle / backward pass sanity
# ---------------------------------------------------------------------------

def test_hooks_removed_after_context_manager_exits():
    model = build_model("efficientnet_b0", pretrained=False, model_variant="rgb_only")
    target_layer = get_target_layer(model, branch="rgb")
    x = _make_input(size=64)

    with GradCAM(model, target_layer) as engine:
        assert len(target_layer._forward_hooks) == 1
        assert len(target_layer._backward_hooks) == 1
        engine.generate(x, target_class="fake")

    assert len(target_layer._forward_hooks) == 0
    assert len(target_layer._backward_hooks) == 0


def test_backward_pass_produces_gradients():
    model = build_model("efficientnet_b0", pretrained=False, model_variant="rgb_only")
    target_layer = get_target_layer(model, branch="rgb")
    x = _make_input(size=64)

    with GradCAM(model, target_layer) as engine:
        assert engine.activations is None
        assert engine.gradients is None
        engine.generate(x, target_class="fake")
        assert engine.activations is not None
        assert engine.gradients is not None
        assert engine.gradients.abs().sum().item() > 0.0


def test_generate_without_hooks_raises():
    model = build_model("efficientnet_b0", pretrained=False, model_variant="rgb_only")
    target_layer = get_target_layer(model, branch="rgb")
    engine = GradCAM(model, target_layer)
    with pytest.raises(RuntimeError):
        engine.generate(_make_input(size=64), target_class="fake")


def test_gradcam_restores_model_train_mode():
    model = build_model("efficientnet_b0", pretrained=False, model_variant="rgb_only")
    model.train()
    target_layer = get_target_layer(model, branch="rgb")
    x = _make_input(size=64, batch=2)

    with GradCAM(model, target_layer) as engine:
        engine.generate(x, target_class="fake")

    assert model.training is True


def test_gradcam_does_not_leave_stray_param_grads():
    model = build_model("efficientnet_b0", pretrained=False, model_variant="rgb_only")
    target_layer = get_target_layer(model, branch="rgb")
    x = _make_input(size=64)

    with GradCAM(model, target_layer) as engine:
        engine.generate(x, target_class="fake")

    for p in model.parameters():
        assert p.grad is None or torch.all(p.grad == 0)


# ---------------------------------------------------------------------------
# Degenerate / constant activation safety
# ---------------------------------------------------------------------------

def test_safe_normalize_constant_map_has_no_nan():
    constant_cam = torch.full((1, 8, 8), 5.0)
    normalized = _safe_normalize(constant_cam)
    assert torch.isfinite(normalized).all()
    assert torch.allclose(normalized, torch.zeros_like(normalized))


def test_safe_normalize_batch_mixed_constant_and_varying():
    cams = torch.stack([
        torch.full((8, 8), 3.0),
        torch.arange(64, dtype=torch.float32).reshape(8, 8),
    ])
    normalized = _safe_normalize(cams)
    assert torch.isfinite(normalized).all()
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    assert torch.allclose(normalized[0], torch.zeros(8, 8))


# ---------------------------------------------------------------------------
# Checkpoint loading integration
# ---------------------------------------------------------------------------

def test_checkpoint_loading_works_with_gradcam(tmp_path):
    device = torch.device("cpu")
    model = build_model("efficientnet_b0", pretrained=False, model_variant="rgb_only")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_name": "efficientnet_b0",
        "model_variant": "rgb_only",
        "branch_mode": "rgb",
        "configuration": {"optimal_threshold": 0.42, "training_strategy": "clean"},
    }
    ckpt_path = tmp_path / "best_model.pt"
    torch.save(ckpt, ckpt_path)

    loaded_model, threshold, loaded_ckpt = load_model_from_checkpoint(ckpt_path, device)
    assert threshold == pytest.approx(0.42)
    assert loaded_model.model_variant == "rgb_only"

    target_layer = get_target_layer(loaded_model, branch="rgb")
    cam, fake_prob = generate_gradcam(loaded_model, _make_input(size=64), target_layer, target_class="fake")
    assert cam.shape == (64, 64)
    assert np.isfinite(cam).all()


def test_end_to_end_cli_visualization_synthetic(tmp_path):
    """Full run_single_visualization path against a synthetic checkpoint + synthetic image."""
    device = torch.device("cpu")
    model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion_no_attn")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_name": "efficientnet_b0",
        "model_variant": "fusion_no_attn",
        "branch_mode": "fusion",
        "configuration": {"optimal_threshold": 0.5, "training_strategy": "degradation"},
    }
    ckpt_path = tmp_path / "best_model.pt"
    torch.save(ckpt, ckpt_path)

    arr = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    image_path = tmp_path / "face.jpg"
    Image.fromarray(arr).save(image_path)

    output_path = tmp_path / "gradcam_example.png"
    metadata = run_single_visualization(
        checkpoint_path=ckpt_path,
        image_path=image_path,
        output_path=output_path,
        branch="rgb",
        target_class="predicted",
        device=device,
    )

    assert output_path.exists()
    meta_path = output_path.with_suffix(".json")
    assert meta_path.exists()

    with open(meta_path) as f:
        saved_meta = json.load(f)
    assert saved_meta["model_variant"] == "fusion_no_attn"
    assert saved_meta["predicted_class"] in ("real", "fake")
    assert metadata["threshold"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Preprocessing parity with standard inference
# ---------------------------------------------------------------------------

def test_preprocessing_matches_standard_inference_transform(tmp_path):
    arr = np.random.randint(0, 256, (300, 220, 3), dtype=np.uint8)
    image_path = tmp_path / "face.png"
    Image.fromarray(arr).save(image_path)

    pil_image, gradcam_tensor = load_and_preprocess_image(image_path, degradation_tier=None)

    _, eval_transform = get_transforms()
    reference_tensor = eval_transform(Image.open(image_path).convert("RGB")).unsqueeze(0)

    assert torch.allclose(gradcam_tensor, reference_tensor, atol=1e-6)


def test_preprocessing_with_degradation_tier_uses_shared_pipeline(tmp_path):
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    image_path = tmp_path / "face.png"
    Image.fromarray(arr).save(image_path)

    pil_clean, tensor_clean = load_and_preprocess_image(image_path, degradation_tier=None)
    pil_degraded, tensor_degraded = load_and_preprocess_image(image_path, degradation_tier="strong_compression")

    # Degradation must actually change the preprocessed tensor (JPEG artifacts introduced).
    assert not torch.allclose(tensor_clean, tensor_degraded)


def test_unknown_degradation_tier_raises(tmp_path):
    arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    image_path = tmp_path / "face.png"
    Image.fromarray(arr).save(image_path)
    with pytest.raises(ValueError):
        load_and_preprocess_image(image_path, degradation_tier="not_a_real_tier")


# ---------------------------------------------------------------------------
# predict() sanity
# ---------------------------------------------------------------------------

def test_predict_returns_consistent_threshold_semantics():
    model = build_model("efficientnet_b0", pretrained=False, model_variant="rgb_only")
    x = _make_input(size=64)
    predicted_class, fake_prob = predict(model, x, threshold=0.5)
    assert predicted_class in ("real", "fake")
    assert 0.0 <= fake_prob <= 1.0
    assert (fake_prob >= 0.5) == (predicted_class == "fake")
