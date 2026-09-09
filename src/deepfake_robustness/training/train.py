import os
import time
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from deepfake_robustness.configs import config
from deepfake_robustness.datasets.dataset import get_dataloaders, assign_group_splits, validate_manifest
from deepfake_robustness.models.model import build_model, resolve_checkpoint_model_kwargs
from deepfake_robustness.degradations.transforms import get_degradation_transform_for_epoch
from deepfake_robustness.evaluation.metrics_utils import compute_video_level_metrics

class DeepfakeLoss(nn.Module):
    def __init__(self, loss_type="focal", alpha=0.50, gamma=1.5, smoothing=0.05, class_weights=None):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.class_weights = class_weights # dict {0: w0, 1: w1}

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        
        # Apply label smoothing
        if self.smoothing > 0:
            targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        else:
            targets_smooth = targets
            
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets_smooth, reduction='none')
        
        # Apply class weights if available
        if self.class_weights is not None:
            w0 = self.class_weights[0]
            w1 = self.class_weights[1]
            weight_t = targets * w1 + (1.0 - targets) * w0
            bce_loss = bce_loss * weight_t

        if self.loss_type == "focal":
            probs = torch.sigmoid(inputs)
            p_t = probs * targets_smooth + (1.0 - probs) * (1.0 - targets_smooth)
            p_t = torch.clamp(p_t, 1e-6, 1.0 - 1e-6)
            focal_weight = (1.0 - p_t) ** self.gamma
            alpha_t = self.alpha * targets_smooth + (1.0 - self.alpha) * (1.0 - targets_smooth)
            loss = alpha_t * focal_weight * bce_loss
        else:
            loss = bce_loss
            
        return loss.mean()

class CoReDFLoss(nn.Module):
    """
    Composite loss function for CoRe-DF architecture.
    Loss = L_classification + 0.5 * L_order + 0.5 * L_validity + 0.1 * L_trajectory
    """
    def __init__(self, cls_weight=1.0, order_weight=0.5, validity_weight=0.5, trajectory_weight=0.1):
        super().__init__()
        self.cls_weight = cls_weight
        self.order_weight = order_weight
        self.validity_weight = validity_weight
        self.trajectory_weight = trajectory_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        
    def forward(self, outputs_dict, targets, order_targets=None, validity_targets=None):
        deepfake_logit = outputs_dict["deepfake_logit"].squeeze(-1)
        
        # 1. Classification Loss
        l_cls = self.bce(deepfake_logit, targets.float())
        
        loss = self.cls_weight * l_cls
        
        # 2. Order Prediction Loss (Optional, if targets provided)
        # order_targets: (B, num_experts, num_experts) binary matrix
        if order_targets is not None and "order_logits" in outputs_dict:
            l_order = self.bce(outputs_dict["order_logits"], order_targets.float())
            loss += self.order_weight * l_order
            
        # 3. Expert Validity Loss (Optional, if targets provided)
        # validity_targets: (B, num_experts) binary vector of which experts should be active
        if validity_targets is not None and "expert_validity_scores" in outputs_dict:
            l_validity = self.mse(outputs_dict["expert_validity_scores"], validity_targets.float())
            loss += self.validity_weight * l_validity
            
        # Trajectory consistency loss would require paired intermediate activations (omitted for now)
            
        return loss
class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def range_test(self, train_loader, start_lr=1e-7, end_lr=10.0, num_iter=100):
        # Save model state
        import copy
        save_dict = {
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict())
        }
        
        self.model.train()
        lrs = []
        losses = []
        best_loss = float('inf')
        
        mult = (end_lr / start_lr) ** (1.0 / num_iter)
        lr = start_lr
        
        # Set learning rate in optimizer
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
            
        iterator = iter(train_loader)
        
        for i in range(num_iter):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
                
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits, _ = self.model(images)
                    logits = logits.squeeze(1)
                    loss = self.criterion(logits, labels)
            else:
                logits, _ = self.model(images)
                logits = logits.squeeze(1)
                loss = self.criterion(logits, labels)
            
            if torch.isnan(loss) or loss > best_loss * 4:
                break
                
            if loss.item() < best_loss:
                best_loss = loss.item()
                
            losses.append(loss.item())
            lrs.append(lr)
            
            loss.backward()
            self.optimizer.step()
            
            lr *= mult
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
                
        # Restore model and optimizer
        self.model.load_state_dict(save_dict["model"])
        self.optimizer.load_state_dict(save_dict["optimizer"])
        
        return lrs, losses



class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._num_updates = 0
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        for name, buffer in self.model.named_buffers():
            self.shadow[name] = buffer.data.clone()

    def register_new_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.data.clone()
        for name, buffer in self.model.named_buffers():
            if name not in self.shadow:
                self.shadow[name] = buffer.data.clone()

    def update(self):
        self._num_updates += 1
        effective_decay = min(self.decay, (1.0 + self._num_updates) / (10.0 + self._num_updates))
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - effective_decay) * param.data + effective_decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
        for name, buffer in self.model.named_buffers():
            if name in self.shadow:
                # Buffers like running_mean can be floating point or integer (e.g. num_batches_tracked)
                if buffer.data.is_floating_point():
                    self.shadow[name] = (1.0 - effective_decay) * buffer.data + effective_decay * self.shadow[name]
                else:
                    self.shadow[name] = buffer.data.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        for name, buffer in self.model.named_buffers():
            self.backup[name] = buffer.data.clone()
            buffer.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        for name, buffer in self.model.named_buffers():
            buffer.data.copy_(self.backup[name])
        self.backup = {}


def plot_diagnostic_curves(labels, probabilities, output_directory):
    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    
    # 1. Confusion Matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Validation Confusion Matrix")
    plt.savefig(output_directory / "val_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 2. ROC Curve
    from sklearn.metrics import roc_curve, auc
    if len(np.unique(labels)) == 2:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Validation ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(output_directory / "val_roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        
    # 3. Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    if len(np.unique(labels)) == 2:
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        ap = average_precision_score(labels, probabilities)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color="green", lw=2, label=f"PR curve (AP = {ap:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Validation Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.savefig(output_directory / "val_precision_recall_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        
    # 4. Calibration Curve
    from sklearn.calibration import calibration_curve
    if len(np.unique(labels)) == 2:
        prob_true, prob_pred = calibration_curve(labels, probabilities, n_bins=10)
        plt.figure(figsize=(6, 5))
        plt.plot(prob_pred, prob_true, marker="s", lw=1, label="Model")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Validation Calibration Plot")
        plt.legend(loc="upper left")
        plt.grid(alpha=0.3)
        plt.savefig(output_directory / "val_calibration_curve.png", dpi=150, bbox_inches="tight")
        plt.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_binary_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities)

    if len(labels) == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
        }

    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": float("nan"),
    }

    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = roc_auc_score(labels, probabilities)

    return metrics


def sync_scheduler_param_groups(scheduler, optimizer, base_lr):
    """Synchronize learning rate scheduler internal base_lrs and _last_lr when new param groups are added."""
    target_len = len(optimizer.param_groups)
    sub_schedulers = getattr(scheduler, "_schedulers", [scheduler])
    for sub in sub_schedulers:
        while hasattr(sub, "base_lrs") and len(sub.base_lrs) < target_len:
            sub.base_lrs.append(base_lr)
        while hasattr(sub, "_last_lr") and len(sub._last_lr) < target_len:
            sub._last_lr.append(optimizer.param_groups[len(sub._last_lr)]["lr"])
    if hasattr(scheduler, "_last_lr"):
        while len(scheduler._last_lr) < target_len:
            scheduler._last_lr.append(optimizer.param_groups[len(scheduler._last_lr)]["lr"])


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, limit_batches=None, ema=None):
    model.train()
    running_loss = 0.0
    total_samples = 0
    all_labels = []
    all_probabilities = []
    accum_steps = max(1, getattr(config, "GRADIENT_ACCUMULATION_STEPS", 1))

    effective_len = min(len(loader), limit_batches) if limit_batches is not None else len(loader)

    optimizer.zero_grad(set_to_none=True)
    accum_count = 0

    for i, batch in enumerate(tqdm(loader, total=effective_len, desc="Training", leave=False)):
        if limit_batches is not None and i >= limit_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        use_mixup = (
            getattr(config, "USE_MIXUP", False)
            and (random.random() < getattr(config, "MIXUP_PROB", 0.5))
        )
        if use_mixup and images.size(0) > 1:
            lam = np.random.beta(getattr(config, "MIXUP_ALPHA", 0.8), getattr(config, "MIXUP_ALPHA", 0.8))
            index = torch.randperm(images.size(0)).to(device)
            images = lam * images + (1 - lam) * images[index]
            targets_a, targets_b = labels, labels[index]
        else:
            use_mixup = False

        accum_count += 1
        is_last_batch = ((i + 1) == effective_len) or ((i + 1) == len(loader))
        is_accumulating = (accum_count < accum_steps) and not is_last_batch

        if device.type == "cuda" and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                if isinstance(outputs, dict):
                    logits = outputs["deepfake_logit"].squeeze(1)
                    if use_mixup:
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    else:
                        loss = criterion(outputs, labels)
                else:
                    logits, _ = outputs
                    logits = logits.squeeze(1)
                    if use_mixup:
                        loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
                    else:
                        loss = criterion(logits, labels)
                scaled_loss = loss / accum_steps

            scaler.scale(scaled_loss).backward()

            if not is_accumulating:
                scaler.unscale_(optimizer)
                if accum_count < accum_steps:
                    grad_scale = float(accum_steps) / float(accum_count)
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(grad_scale)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIPPING)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                if ema is not None:
                    ema.update_parameters(model)
        else:
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs["deepfake_logit"].squeeze(1)
                if use_mixup:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, labels)
            else:
                logits, _ = outputs
                logits = logits.squeeze(1)
                if use_mixup:
                    loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
                else:
                    loss = criterion(logits, labels)
            scaled_loss = loss / accum_steps

            scaled_loss.backward()

            if not is_accumulating:
                if accum_count < accum_steps:
                    grad_scale = float(accum_steps) / float(accum_count)
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(grad_scale)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIPPING)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                if ema is not None:
                    ema.update_parameters(model)

        probabilities = torch.sigmoid(logits.detach())
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        # Exclude mixup batches from discrete binary classification metrics
        if not use_mixup:
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    metrics = calculate_binary_metrics(all_labels, all_probabilities)
    metrics["loss"] = running_loss / total_samples if total_samples > 0 else 0.0
    return metrics


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, limit_batches=None, use_tta=False):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probabilities = []
    all_paths = []
    all_video_ids = []
    all_manipulations = []

    for i, batch in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
        if limit_batches is not None and i >= limit_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if use_tta:
            # Original
            out1 = model(images)
            logits1 = out1["deepfake_logit"].squeeze(1) if isinstance(out1, dict) else out1[0].squeeze(1)
            probs1 = torch.sigmoid(logits1)
            
            # Horizontal Flip
            images_hf = torch.flip(images, [3])
            out2 = model(images_hf)
            logits2 = out2["deepfake_logit"].squeeze(1) if isinstance(out2, dict) else out2[0].squeeze(1)
            probs2 = torch.sigmoid(logits2)
            
            # Mild Resize
            images_res = torch.nn.functional.interpolate(images, scale_factor=0.9, mode='bilinear', align_corners=False)
            images_res = torch.nn.functional.interpolate(images_res, size=images.shape[2:], mode='bilinear', align_corners=False)
            out3 = model(images_res)
            logits3 = out3["deepfake_logit"].squeeze(1) if isinstance(out3, dict) else out3[0].squeeze(1)
            probs3 = torch.sigmoid(logits3)
            
            probabilities = (probs1 + probs2 + probs3) / 3.0
            probs_clamped = torch.clamp(probabilities, 1e-6, 1.0 - 1e-6)
            logits = torch.logit(probs_clamped)
            
            if isinstance(out1, dict):
                if hasattr(criterion, 'forward') and 'outputs_dict' in criterion.forward.__code__.co_varnames:
                    loss = criterion({"deepfake_logit": logits.unsqueeze(1)}, labels)
                else:
                    loss = criterion(logits, labels)
            else:
                loss = criterion(logits, labels)
        else:
            with torch.amp.autocast(device_type="cuda") if device.type == "cuda" else torch.autocast(device_type="cpu"):
                outputs = model(images)
                if isinstance(outputs, dict):
                    logits = outputs["deepfake_logit"].squeeze(1)
                    if hasattr(criterion, 'forward') and 'outputs_dict' in criterion.forward.__code__.co_varnames:
                        loss = criterion(outputs, labels)
                    else:
                        loss = criterion(logits, labels)
                else:
                    logits = outputs[0].squeeze(1)
                    loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)

        running_loss += loss.item() * images.size(0)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probabilities.extend(probabilities.float().cpu().numpy().tolist())
        all_paths.extend(batch["path"])
        all_video_ids.extend(batch["video_id"])
        all_manipulations.extend(batch.get("manipulation", ["unknown"] * len(batch["path"])))

    metrics = calculate_binary_metrics(all_labels, all_probabilities)
    metrics["loss"] = running_loss / len(all_labels) if all_labels else 0.0

    predictions_df = pd.DataFrame({
        "image_path": all_paths,
        "video_id": all_video_ids,
        "manipulation": all_manipulations,
        "label": all_labels,
        "prob_fake": all_probabilities,
    })

    return metrics, predictions_df


def plot_training_curves(history_df, output_directory):
    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train Loss")
    plt.plot(history_df["epoch"], history_df["validation_loss"], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{config.MODEL_NAME} - Loss Curves ({config.TRAINING_STRATEGY})")
    plt.legend()
    plt.grid(alpha=0.3)
    loss_plot_path = output_directory / "loss_curves.png"
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    if "validation_roc_auc" in history_df.columns:
        plt.plot(history_df["epoch"], history_df["validation_roc_auc"], marker="o", label="Val ROC-AUC")
    plt.plot(history_df["epoch"], history_df["validation_accuracy"], marker="o", label="Val Accuracy")
    plt.plot(history_df["epoch"], history_df["validation_f1"], marker="o", label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{config.MODEL_NAME} - Validation Metrics ({config.TRAINING_STRATEGY})")
    plt.legend()
    plt.grid(alpha=0.3)
    metrics_plot_path = output_directory / "metrics_curves.png"
    plt.savefig(metrics_plot_path, dpi=150)
    plt.close()
    
    print(f"Generated metric plots in {output_directory}")


def compute_class_balanced_weights(dataframe, beta=0.999):
    class_counts = dataframe["label"].value_counts().to_dict()
    c0 = class_counts.get(0.0, 1)
    c1 = class_counts.get(1.0, 1)
    w0 = (1.0 - beta) / (1.0 - np.power(beta, c0))
    w1 = (1.0 - beta) / (1.0 - np.power(beta, c1))
    sum_w = w0 + w1
    w0 = (w0 / sum_w) * 2.0
    w1 = (w1 / sum_w) * 2.0
    return {0: w0, 1: w1}


def safe_torch_save(obj, path):
    """Save PyTorch object atomically using temporary file to prevent iostream / zip write errors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp_{os.getpid()}")
    try:
        torch.save(obj, tmp_path)
        tmp_path.replace(path)
    except Exception as e:
        print(f"Warning: atomic save to {path} failed ({e}); falling back to direct write.")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError as cleanup_err:
                print(f"Warning: could not remove stale temp file {tmp_path}: {cleanup_err}")
        torch.save(obj, path)


def recalibrate_bn_statistics(model, loader, device, num_batches=100):
    """Recalibrate BatchNorm running_mean and running_var after weight averaging using training data batches."""
    print("[+] Recalibrating BatchNorm running statistics on training data...")
    model.train()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    _ = model(images)
            else:
                _ = model(images)
    print("--> BatchNorm statistics recalibrated successfully.")


def average_checkpoints(checkpoint_paths, output_path, device):
    print(f"\n[+] Running Post-Training Checkpoint Averaging over {len(checkpoint_paths)} checkpoints...")
    if not checkpoint_paths:
        return

    first_ckpt = torch.load(checkpoint_paths[0], map_location=device, weights_only=False)
    first_state = first_ckpt["model_state_dict"]

    param_keys = set()
    for k, v in first_state.items():
        if v.is_floating_point() and "num_batches_tracked" not in k:
            param_keys.add(k)

    averaged_state = {k: v.clone() for k, v in first_state.items()}

    for path in checkpoint_paths[1:]:
        state = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
        for key in param_keys:
            if key in state:
                averaged_state[key] += state[key]

    num_checkpoints = len(checkpoint_paths)
    for key in param_keys:
        averaged_state[key] = averaged_state[key] / num_checkpoints

    first_ckpt["model_state_dict"] = averaged_state
    first_ckpt["averaged_checkpoints"] = [str(p) for p in checkpoint_paths]

    safe_torch_save(first_ckpt, output_path)
    print(f"--> Saved averaged model checkpoint to: {output_path}")


def build_optimizer(m, backbone_lr=None, classifier_lr=None, weight_decay=None):
    if backbone_lr is None:
        backbone_lr = getattr(config, "BACKBONE_LR", 1e-4)
    if classifier_lr is None:
        classifier_lr = getattr(config, "CLASSIFIER_LR", 1e-3)
    if weight_decay is None:
        weight_decay = getattr(config, "WEIGHT_DECAY", 5e-3)

    decay_backbone = [p for n, p in m.named_parameters() if p.requires_grad and not ("classifier" in n or "mlp" in n or "head" in n) and p.ndim >= 2]
    no_decay_backbone = [p for n, p in m.named_parameters() if p.requires_grad and not ("classifier" in n or "mlp" in n or "head" in n) and p.ndim < 2]
    decay_classifier = [p for n, p in m.named_parameters() if p.requires_grad and ("classifier" in n or "mlp" in n or "head" in n) and p.ndim >= 2]
    no_decay_classifier = [p for n, p in m.named_parameters() if p.requires_grad and ("classifier" in n or "mlp" in n or "head" in n) and p.ndim < 2]

    return torch.optim.AdamW(
        [
            {"params": decay_backbone, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": no_decay_backbone, "lr": backbone_lr, "weight_decay": 0.0},
            {"params": decay_classifier, "lr": classifier_lr, "weight_decay": weight_decay},
            {"params": no_decay_classifier, "lr": classifier_lr, "weight_decay": 0.0},
        ]
    )


def build_scheduler(optimizer):
    if config.SCHEDULER == "cosine":
        scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=config.WARMUP_EPOCHS
        )
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.NUM_EPOCHS - config.WARMUP_EPOCHS), eta_min=1e-6
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler1, scheduler2], milestones=[config.WARMUP_EPOCHS]
        )
    else:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )


def compute_optimal_f1_threshold(labels, probabilities):
    """Sweep precision-recall operating points and return the argmax-F1 threshold.

    Note: this is F1-optimal threshold selection (t* = argmax_t F1(t)), NOT Youden's J
    statistic (which would maximize TPR(t) - FPR(t) on the ROC curve). The two criteria
    generally select different thresholds; this project intentionally uses the F1-optimal
    criterion, so log messages/metadata should describe it as such.

    Returns (optimal_threshold, best_f1). Falls back to (0.50, 0.0) if no valid threshold
    can be computed (e.g. empty input).
    """
    from sklearn.metrics import precision_recall_curve

    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities).astype(float)

    if len(labels) == 0:
        # precision_recall_curve itself raises on empty input (a ValueError from an internal
        # broadcast, not a friendly message) rather than returning an empty/degenerate result,
        # so this must be guarded before calling it.
        return 0.50, 0.0

    prec_vals, rec_vals, thresh_vals = precision_recall_curve(labels, probabilities)
    f1_scores = 2 * (prec_vals * rec_vals) / (prec_vals + rec_vals + 1e-8)
    # precision_recall_curve returns thresh_vals of length len(prec_vals) - 1
    valid_mask = np.isfinite(thresh_vals) & (thresh_vals >= 0.0) & (thresh_vals <= 1.0)
    valid_thresholds = thresh_vals[valid_mask]
    valid_f1 = f1_scores[:-1][valid_mask]

    if len(valid_f1) > 0:
        best_idx = int(np.argmax(valid_f1))
        return float(valid_thresholds[best_idx]), float(valid_f1[best_idx])
    return 0.50, 0.0


def get_checkpoint_config_dict(manifest_path=None):
    import sys
    import subprocess
    commit_hash = "unknown"
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        commit_hash = res.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # Optional provenance metadata only (e.g. no git installed, or not a git checkout) -
        # not worth surfacing a warning for on every checkpoint save.
        pass
    return {
        "model_name": config.MODEL_NAME,
        "model_variant": config.MODEL_VARIANT,
        "branch_mode": getattr(config, "BRANCH_MODE", "fusion"),
        "image_size": config.IMAGE_SIZE,
        "training_strategy": config.TRAINING_STRATEGY,
        "seed": config.SEED,
        "batch_size": config.BATCH_SIZE,
        "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "backbone_lr": config.BACKBONE_LR,
        "classifier_lr": config.CLASSIFIER_LR,
        "weight_decay": config.WEIGHT_DECAY,
        "dropout": config.DROPOUT,
        "label_smoothing": config.LABEL_SMOOTHING,
        "focal_alpha": getattr(config, "FOCAL_ALPHA", 0.50),
        "focal_gamma": getattr(config, "FOCAL_GAMMA", 1.5),
        "balancing_strategy": config.BALANCING_STRATEGY,
        "use_mixup": getattr(config, "USE_MIXUP", False),
        "mixup_alpha": getattr(config, "MIXUP_ALPHA", 0.8),
        "mixup_prob": getattr(config, "MIXUP_PROB", 0.5),
        "freeze_percentage": config.FREEZE_PERCENT,
        "progressive_unfreeze": getattr(config, "PROGRESSIVE_UNFREEZE", True),
        "ema_decay": getattr(config, "EMA_DECAY", 0.999),
        "scheduler": config.SCHEDULER,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "git_commit": commit_hash,
        "manifest_path": str(manifest_path if manifest_path is not None else config.MANIFEST_PATH),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME, choices=["efficientnet_b0", "efficientnet_b4", "convnext_tiny", "resnet18", "resnet50", "mobilenet_v3_small", "shufflenet_v2", "densenet121"], help="Model backbone")
    parser.add_argument("--variant", type=str, default=config.MODEL_VARIANT, choices=["fusion", "rgb_only", "fusion_no_attn"], help="Model variant")
    parser.add_argument("--strategy", type=str, default=config.TRAINING_STRATEGY, choices=["clean", "standard", "degradation"], help="Training strategy")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS, help="Number of dataloader workers")
    parser.add_argument("--backbone_lr", type=float, default=config.BACKBONE_LR, help="Backbone learning rate")
    parser.add_argument("--classifier_lr", type=float, default=config.CLASSIFIER_LR, help="Classifier learning rate")
    parser.add_argument("--patience", type=int, default=config.PATIENCE, help="Early stopping patience")
    parser.add_argument("--limit_batches", type=int, default=None, help="Limit number of batches per epoch (for sanity check)")
    parser.add_argument("--find_lr", action="store_true", help="Run learning rate finder and exit")
    parser.add_argument("--fresh", action="store_true", help="Start training from scratch, removing existing checkpoints")
    parser.add_argument("--manifest", type=str, default=None, help="Path to a manifest CSV overriding config.MANIFEST_PATH (for isolated/test runs; does not mutate config)")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="Use pretrained ImageNet backbone weights")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Disable pretrained ImageNet backbone weights (useful for offline/smoke runs)")
    parser.set_defaults(pretrained=config.PRETRAINED)
    args = parser.parse_args()

    # Override config global values
    config.MODEL_NAME = args.model
    config.MODEL_VARIANT = args.variant
    config.TRAINING_STRATEGY = args.strategy
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.NUM_WORKERS = args.num_workers
    config.BACKBONE_LR = args.backbone_lr
    config.CLASSIFIER_LR = args.classifier_lr
    config.PATIENCE = args.patience
    config.PRETRAINED = args.pretrained

    # The manifest path is kept as a local variable (not written into config.MANIFEST_PATH) so
    # an override can't leak into other config-dependent code paths in the same process.
    manifest_path = Path(args.manifest) if args.manifest else config.MANIFEST_PATH

    # Dynamic image size adaptation
    if config.MODEL_NAME == "efficientnet_b4":
        config.IMAGE_SIZE = 380
        print(f"Adapting image size to {config.IMAGE_SIZE} for efficientnet_b4 backbone")
    else:
        config.IMAGE_SIZE = getattr(config, "IMAGE_SIZE", 256)
        print(f"Preserving native image size of {config.IMAGE_SIZE}x{config.IMAGE_SIZE} for frequency preservation")

    # Output experiment setup & paths upfront
    variant_suffix = f"_{config.MODEL_VARIANT}" if config.MODEL_VARIANT != "fusion" else ""
    experiment_name = f"{config.MODEL_NAME}{variant_suffix}_{config.TRAINING_STRATEGY}"
    experiment_directory = config.OUTPUT_ROOT / experiment_name
    experiment_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = experiment_directory / "best_model.pt"
    last_checkpoint_path = experiment_directory / "last_model.pt"
    best_single_path = experiment_directory / "best_single_model.pt"
    averaged_path = experiment_directory / "averaged_model.pt"
    history_path = experiment_directory / "history.csv"

    # Handle --fresh flag to clean stale experiment artifacts
    if args.fresh and experiment_directory.exists():
        print(f"--> --fresh flag specified. Cleaning training artifacts in {experiment_directory}...")
        for item in experiment_directory.glob("*"):
            if item.is_file() and (item.suffix in [".pt", ".csv", ".png", ".json", ".log"] or "best_model" in item.name or "last_model" in item.name):
                try:
                    item.unlink()
                except OSError as e:
                    print(f"Warning: could not remove stale artifact {item}: {e}")

    set_seed(config.SEED)

    if not manifest_path.exists():
        print(f"Manifest file not found at {manifest_path}. Please run extract_faces.py first.")
        return

    # Load data manifest and validate structure & splits upfront
    print(f"Loading data manifest from {manifest_path}...")
    manifest_df = pd.read_csv(manifest_path)
    if "split" not in manifest_df.columns:
        print("--> Manifest missing 'split' column. Assigning group splits...")
        manifest_df = assign_group_splits(manifest_df, seed=config.SEED)

    is_valid, val_msg = validate_manifest(manifest_df)
    if not is_valid:
        raise ValueError(f"Manifest validation failed before training: {val_msg}")
    print(f"[+] Dataset manifest validated successfully ({len(manifest_df)} samples).")

    # Get dataloaders
    print("Setting up dataloaders...")
    train_loader, val_loader, _ = get_dataloaders(manifest_df)
    print(f"Dataloaders initialized. Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build Model
    print(f"Building model: {config.MODEL_NAME} (pretrained={config.PRETRAINED})...")
    model = build_model(config.MODEL_NAME, pretrained=config.PRETRAINED, model_variant=config.MODEL_VARIANT).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params:,}")

    # Loss Configuration (including Class-Balanced Weights)
    class_weights = None
    if config.BALANCING_STRATEGY == "class_balanced_loss":
        train_df_labels = manifest_df[manifest_df["split"] == "train"].copy()
        class_weights = compute_class_balanced_weights(train_df_labels, beta=config.CB_BETA)
        print(f"--> Using Class-Balanced Loss weights: {class_weights}")

    loss_type = "bce"
    if config.BALANCING_STRATEGY == "focal_loss":
        loss_type = "focal"

    focal_alpha = getattr(config, "FOCAL_ALPHA", 0.50)
    focal_gamma = getattr(config, "FOCAL_GAMMA", 1.5)
    
    if config.MODEL_VARIANT == "modular_order":
        criterion = CoReDFLoss()
    else:
        criterion = DeepfakeLoss(
            loss_type=loss_type,
            alpha=focal_alpha,
            gamma=focal_gamma,
            smoothing=config.LABEL_SMOOTHING,
            class_weights=class_weights
        )

    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    # Smith-style Learning Rate Finder execution
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and getattr(config, "USE_MIXED_PRECISION", True) else None

    # TensorBoard setup
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(config.TENSORBOARD_DIR / experiment_name))
        print(f"TensorBoard logging initialized at: {config.TENSORBOARD_DIR / experiment_name}")
    except Exception as e:
        print(f"Could not load TensorBoard writer: {e}")

    # Auto-resume training if last checkpoint exists
    start_epoch = 1
    best_validation_auc = -np.inf
    best_validation_score = -np.inf
    using_auc_tracking = True
    epochs_without_improvement = 0
    history = []
    best_checkpoints = [] # list of (val_score, metric_name, file_path)

    resume_success = False
    if last_checkpoint_path.exists():
        print(f"Found existing last checkpoint at {last_checkpoint_path}. Resuming...")
        try:
            checkpoint = torch.load(
                last_checkpoint_path,
                map_location=device,
                weights_only=False,
            )

            # Prefer the checkpoint's own self-describing architecture metadata over whatever
            # the current CLI/config happens to specify, so resuming always reconstructs the
            # exact architecture the run was trained with (legacy checkpoints without this
            # metadata fall back to the current config, matching prior behavior).
            ckpt_model_name, ckpt_model_variant, ckpt_branch_mode = resolve_checkpoint_model_kwargs(checkpoint)
            current_branch_mode = getattr(config, "BRANCH_MODE", "fusion")
            if (ckpt_model_name, ckpt_model_variant, ckpt_branch_mode) != (config.MODEL_NAME, config.MODEL_VARIANT, current_branch_mode):
                print(
                    f"--> Checkpoint architecture ({ckpt_model_name}/{ckpt_model_variant}/{ckpt_branch_mode}) differs "
                    f"from current config ({config.MODEL_NAME}/{config.MODEL_VARIANT}/{current_branch_mode}); "
                    f"rebuilding model from checkpoint metadata."
                )
                model = build_model(
                    ckpt_model_name, pretrained=False, branch_mode=ckpt_branch_mode, model_variant=ckpt_model_variant
                ).to(device)

            model.load_state_dict(checkpoint["model_state_dict"])
            resume_success = True

            # Restore trainable parameter requires_grad state before rebuilding optimizer
            if "trainable_param_names" in checkpoint:
                trainable_set = set(checkpoint["trainable_param_names"])
                for name, param in model.named_parameters():
                    param.requires_grad = (name in trainable_set)
                print(f"--> Restored requires_grad states ({len(trainable_set)} trainable parameter tensors).")

            # Rebuild optimizer and scheduler bound to new optimizer instance
            optimizer = build_optimizer(model)
            if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            scheduler = build_scheduler(optimizer)
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            assert scheduler.optimizer is optimizer, "Resumed scheduler must be bound to resumed optimizer instance!"

            if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])

            start_epoch = checkpoint["epoch"] + 1
            best_validation_auc = checkpoint.get("best_validation_auc", -np.inf)
            best_validation_score = checkpoint.get("best_validation_score", best_validation_auc)
            using_auc_tracking = checkpoint.get("using_auc_tracking", True)
            history = checkpoint.get("history", [])
            epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

            if "best_checkpoints" in checkpoint and checkpoint["best_checkpoints"]:
                best_checkpoints = [(score, m_name, Path(p)) if len(tuple(item)) == 3 else (item[0], "video_roc_auc", Path(item[1])) for item in checkpoint["best_checkpoints"] if Path(item[-1]).exists()]
                print(f"--> Restored {len(best_checkpoints)} historical candidate best checkpoints for averaging.")

            tracking_desc = "Video AUC" if using_auc_tracking else "Balanced Accuracy"
            print(
                f"Resuming training from Epoch {start_epoch} | "
                f"Best AUC: {best_validation_auc:.4f} | "
                f"Best Score: {best_validation_score:.4f} (tracking {tracking_desc})"
            )
        except Exception as error:
            print(f"Could not load checkpoint ({error}). Starting training from scratch.")
            try:
                last_checkpoint_path.rename(last_checkpoint_path.with_name("last_model.incompatible.pt"))
            except OSError as rename_err:
                print(f"Warning: could not rename incompatible checkpoint {last_checkpoint_path}: {rename_err}")
            # Rebuild model, optimizer, and scheduler to ensure a clean slate because partial
            # restoration might have corrupted the requires_grad states. No checkpoint metadata
            # is available here (the load itself failed), so this uses the same
            # resolve_checkpoint_model_kwargs(None) config-fallback path as a fresh run.
            fresh_model_name, fresh_model_variant, fresh_branch_mode = resolve_checkpoint_model_kwargs(None)
            model = build_model(
                fresh_model_name,
                pretrained=config.PRETRAINED,
                branch_mode=fresh_branch_mode,
                model_variant=fresh_model_variant,
            ).to(device)
            optimizer = build_optimizer(model)
            scheduler = build_scheduler(optimizer)

    # Initialize EMA tracker
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config.EMA_DECAY))
    if resume_success and isinstance(checkpoint, dict) and "ema_shadow" in checkpoint and checkpoint["ema_shadow"] is not None:
        ema.load_state_dict(checkpoint["ema_shadow"])
        print("--> Restored EMA shadow weights from checkpoint.")

    # Determine num_freeze
    num_features = len(model.features) if hasattr(model, "features") else 0
    num_freeze = int(num_features * config.FREEZE_PERCENT)

    print(f"Starting training loop from Epoch {start_epoch} to {config.NUM_EPOCHS}...")
    training_start_time = time.time()
    metric_name = "Video ROC-AUC"
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        
        # Generic progressive unfreezing logic
        unfrozen_this_epoch = False
        if config.PROGRESSIVE_UNFREEZE and num_freeze > 0:
            unfreeze_interval = max(1, int((config.NUM_EPOCHS * 0.8) / num_freeze))
            if epoch > 1 and (epoch - 1) % unfreeze_interval == 0:
                unfreeze_idx = num_freeze - 1 - ((epoch - 1) // unfreeze_interval)
                if unfreeze_idx >= 0:
                    for param in model.features[unfreeze_idx].parameters():
                        param.requires_grad = True
                    print(f"--> Epoch {epoch}: Progressive Unfreezing block {unfreeze_idx} of backbone.")
                    unfrozen_this_epoch = True

        if unfrozen_this_epoch:
            existing_params = set()
            for group in optimizer.param_groups:
                existing_params.update(group["params"])

            newly_unfrozen_decay = [
                p for p in model.parameters()
                if p.requires_grad and p not in existing_params and p.ndim >= 2
            ]
            newly_unfrozen_no_decay = [
                p for p in model.parameters()
                if p.requires_grad and p not in existing_params and p.ndim < 2
            ]

            if newly_unfrozen_decay or newly_unfrozen_no_decay:
                current_backbone_lr = optimizer.param_groups[0]["lr"]
                if newly_unfrozen_decay:
                    optimizer.add_param_group({
                        "params": newly_unfrozen_decay,
                        "lr": current_backbone_lr,
                        "weight_decay": config.WEIGHT_DECAY,
                    })
                if newly_unfrozen_no_decay:
                    optimizer.add_param_group({
                        "params": newly_unfrozen_no_decay,
                        "lr": current_backbone_lr,
                        "weight_decay": 0.0,
                    })
                sync_scheduler_param_groups(scheduler, optimizer, config.BACKBONE_LR)

            # EMA updates parameters automatically
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"--> Updated number of trainable parameters: {trainable_params:,}")
        
        # Curriculum: update training transform severity each epoch (degradation strategy only)
        if (
            config.TRAINING_STRATEGY == "degradation"
            and getattr(config, "CURRICULUM_ENABLED", False)
        ):
            epoch_transform = get_degradation_transform_for_epoch(epoch, config.NUM_EPOCHS)
            train_loader.dataset.transform = epoch_transform
            if epoch == 1 or epoch == getattr(config, "CURRICULUM_RAMP_EPOCHS", 0):
                severity = getattr(config, "CURRICULUM_SEVERITY_START", 0.3) if epoch == 1 else 1.0
                print(f"---> Epoch {epoch}: Curriculum severity = {severity:.2f}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            limit_batches=args.limit_batches,
            ema=ema,
        )

        
        # Determine which model to evaluate
        eval_model = ema.module if ema is not None else model
        
        # Update BatchNorm stats for EMA before evaluation
        if ema is not None:
            torch.optim.swa_utils.update_bn(train_loader, ema, device=device)

        val_metrics, val_predictions_df = evaluate_model(
            model=eval_model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            limit_batches=args.limit_batches,
        )
        

        val_video_res = compute_video_level_metrics(val_predictions_df)
        val_video_auc = val_video_res.get("video_roc_auc", float("nan"))

        current_learning_rate = optimizer.param_groups[0]["lr"]
        if config.SCHEDULER == "plateau":
            scheduler.step(val_metrics["roc_auc"])
        else:
            scheduler.step()
        epoch_duration = time.time() - epoch_start_time

        epoch_results = {
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_roc_auc": train_metrics["roc_auc"],
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "validation_roc_auc": val_metrics["roc_auc"],
            "validation_video_roc_auc": val_video_auc,
        }
        history.append(epoch_results)

        history_df = pd.DataFrame(history)
        history_df.to_csv(history_path, index=False)
        plot_training_curves(history_df, experiment_directory)

        # TensorBoard Logging
        if writer is not None:
            writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
            writer.add_scalar("Loss/Val", val_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/Train", train_metrics["accuracy"], epoch)
            writer.add_scalar("Accuracy/Val", val_metrics["accuracy"], epoch)
            writer.add_scalar("BalancedAccuracy/Val", val_metrics["balanced_accuracy"], epoch)
            if not np.isnan(val_metrics["roc_auc"]):
                writer.add_scalar("ROC-AUC/Val_Frame", val_metrics["roc_auc"], epoch)
            if not np.isnan(val_video_auc):
                writer.add_scalar("ROC-AUC/Val_Video", val_video_auc, epoch)
            writer.add_scalar("LR/Backbone", current_learning_rate, epoch)

        vid_auc_str = f"{val_video_auc:.4f}" if not np.isnan(val_video_auc) else "N/A"
        print(
            f"Epoch {epoch:02d}/{config.NUM_EPOCHS} ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Frame AUC: {val_metrics['roc_auc']:.4f} | "
            f"Video AUC: {vid_auc_str} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Prioritize Video ROC-AUC for best model selection, fallback to Frame ROC-AUC / Balanced Accuracy
        current_auc = val_video_auc if not np.isnan(val_video_auc) else val_metrics["roc_auc"]
        current_bal_acc = val_metrics["balanced_accuracy"]

        if not np.isnan(current_auc):
            metric_name = "Video ROC-AUC" if not np.isnan(val_video_auc) else "Frame ROC-AUC"
            current_score = current_auc
            if not using_auc_tracking:
                using_auc_tracking = True
                best_validation_score = -np.inf
            is_best = (best_validation_score == -np.inf) or (current_score > best_validation_score)
        else:
            metric_name = "Balanced Accuracy (Fallback)"
            current_score = current_bal_acc
            if using_auc_tracking and best_validation_score == -np.inf:
                using_auc_tracking = False
                best_validation_score = -np.inf
            is_best = (not using_auc_tracking) and ((best_validation_score == -np.inf) or (current_score > best_validation_score))

        # Maintain true top N validation checkpoints overall for post-training averaging
        epoch_ckpt_path = experiment_directory / f"best_model_epoch_{epoch}.pt"
        total_training_time_so_far = time.time() - training_start_time

        

        safe_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": config.MODEL_NAME,
                "model_variant": config.MODEL_VARIANT,
                "training_strategy": config.TRAINING_STRATEGY,
                "image_size": config.IMAGE_SIZE,
                "best_validation_auc": current_auc,
                "best_validation_score": current_score,
                "selection_metric_used": metric_name,
                "using_auc_tracking": using_auc_tracking,
                "epoch": epoch,
                "trainable_parameters": trainable_params,
                "training_time_seconds": total_training_time_so_far,
                "configuration": get_checkpoint_config_dict(manifest_path),
            },
            epoch_ckpt_path,
        )

        

        best_checkpoints.append((current_score, metric_name, epoch_ckpt_path))
        # Filter best_checkpoints to ensure only candidates with matching selection metric type are kept
        valid_candidates = [item for item in best_checkpoints if item[1] == metric_name]
        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        keep_candidates = valid_candidates[:config.NUM_CHECKPOINTS_TO_AVERAGE]

        for item in list(best_checkpoints):
            if item not in keep_candidates:
                best_checkpoints.remove(item)
                worst_path = item[2]
                if worst_path.exists() and worst_path != checkpoint_path:
                    try:
                        worst_path.unlink()
                    except OSError as e:
                        print(f"Warning: could not remove superseded checkpoint {worst_path}: {e}")

        if is_best:
            best_validation_score = current_score
            if not np.isnan(current_auc):
                best_validation_auc = current_auc
            epochs_without_improvement = 0

            # Copy overall best checkpoint to best_single_path and initial checkpoint_path
            shutil.copyfile(epoch_ckpt_path, best_single_path)
            shutil.copyfile(epoch_ckpt_path, checkpoint_path)
            print(f"--> Saved overall best single checkpoint to: {best_single_path} ({metric_name}: {best_validation_score:.4f})")

            plot_diagnostic_curves(
                val_predictions_df["label"].to_list(),
                val_predictions_df["prob_fake"].to_list(),
                experiment_directory
            )
        else:
            epochs_without_improvement += 1

        # Save last checkpoint after metric evaluation and model selection updates
        safe_torch_save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "ema_shadow": ema.state_dict() if ema is not None else None,
                "best_validation_auc": best_validation_auc,
                "best_validation_score": best_validation_score,
                "using_auc_tracking": using_auc_tracking,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "best_checkpoints": [(score, m_name, str(p)) for score, m_name, p in best_checkpoints],
                "trainable_param_names": [n for n, p in model.named_parameters() if p.requires_grad],
                "configuration": get_checkpoint_config_dict(manifest_path),
            },
            last_checkpoint_path,
        )

        # Check Early Stopping Trigger
        if epochs_without_improvement >= config.PATIENCE:
            print(f"--> Early stopping triggered: validation {metric_name} did not improve for {config.PATIENCE} epochs.")
            break

    # Post-training Checkpoint Evaluation & Averaging
    if best_checkpoints:
        # Filter matching metric candidates for final averaging
        final_metric_candidates = [item for item in best_checkpoints if item[1] == metric_name]
        final_ckpt_paths = [path for _, _, path in final_metric_candidates]
        if len(final_ckpt_paths) > 1:
            average_checkpoints(final_ckpt_paths, averaged_path, device)

            # Recalibrate BatchNorm running statistics for averaged weights
            avg_ckpt_data = torch.load(averaged_path, map_location=device, weights_only=False)
            model.load_state_dict(avg_ckpt_data["model_state_dict"])
            recalibrate_bn_statistics(model, train_loader, device, num_batches=100)
            avg_ckpt_data["model_state_dict"] = model.state_dict()
            safe_torch_save(avg_ckpt_data, averaged_path)

            # Compare best single model vs averaged model on validation set
            print("\n[+] Comparing Best Single Model vs Averaged Model on Validation Set...")
            best_single_ckpt = torch.load(best_single_path, map_location=device, weights_only=False)
            model.load_state_dict(best_single_ckpt["model_state_dict"])
            single_metrics, single_preds_df = evaluate_model(model, val_loader, criterion, device)
            single_vid = compute_video_level_metrics(single_preds_df)
            single_score = single_vid["video_roc_auc"] if not np.isnan(single_vid["video_roc_auc"]) else single_metrics["roc_auc"]

            model.load_state_dict(avg_ckpt_data["model_state_dict"])
            avg_metrics, avg_preds_df = evaluate_model(model, val_loader, criterion, device)
            avg_vid = compute_video_level_metrics(avg_preds_df)
            avg_score = avg_vid["video_roc_auc"] if not np.isnan(avg_vid["video_roc_auc"]) else avg_metrics["roc_auc"]

            print(f"--> Best Single Model Val Score: {single_score:.4f}")
            print(f"--> Averaged Model Val Score:    {avg_score:.4f}")

            if avg_score > single_score:
                print("--> Selected Averaged Model as final best_model.pt (superior validation score).")
                shutil.copyfile(averaged_path, checkpoint_path)
            else:
                print("--> Selected Best Single Model as final best_model.pt (superior validation score).")
                shutil.copyfile(best_single_path, checkpoint_path)
        else:
            shutil.copyfile(best_single_path, checkpoint_path)
    else:
        print("Warning: No new best checkpoints saved during current run.")
        if not checkpoint_path.exists():
            if last_checkpoint_path.exists():
                shutil.copyfile(last_checkpoint_path, checkpoint_path)
            else:
                safe_torch_save({"model_state_dict": model.state_dict(), "configuration": {}}, checkpoint_path)

    # Post-averaging F1-optimal threshold calibration on val split (NOT Youden's J - see
    # compute_optimal_f1_threshold's docstring for the distinction).
    if checkpoint_path.exists():
        print("\n[+] Running post-averaging F1-optimal threshold calibration on val split...")
        try:
            averaged_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(averaged_ckpt["model_state_dict"])
            model.eval()
            _, calib_preds_df = evaluate_model(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
            calib_labels = calib_preds_df["label"].to_numpy().astype(int)
            calib_probs = calib_preds_df["prob_fake"].to_numpy().astype(float)

            optimal_threshold, best_f1 = compute_optimal_f1_threshold(calib_labels, calib_probs)

            print(f"---> Optimal threshold (F1={best_f1:.4f}): {optimal_threshold:.4f} (vs. fixed 0.5)")
            
            # Persist the threshold in the checkpoint's configuration dict
            averaged_ckpt.setdefault("configuration", {})
            averaged_ckpt["configuration"]["optimal_threshold"] = optimal_threshold
            averaged_ckpt["configuration"]["threshold_criterion"] = "f1"
            averaged_ckpt["configuration"]["threshold_f1"] = best_f1
            safe_torch_save(averaged_ckpt, checkpoint_path)
            print(f"---> Threshold saved to checkpoint: {checkpoint_path}")
        except Exception as calib_err:
            print(f"Warning: Threshold calibration failed ({calib_err}). Checkpoint left at threshold=0.5.")
    else:
        print("Warning: Checkpoint path does not exist for threshold calibration.")

    total_train_time = time.time() - training_start_time
    print(f"Training completed in {total_train_time:.1f}s. Best validation ROC-AUC: {best_validation_auc:.4f}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
