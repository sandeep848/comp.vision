import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from deepfake_robustness.configs import config
from deepfake_robustness.models.core_df import ModularExpertHead, OrderInferenceModule, OrderConditionedFusion, DynamicRouter

class MultiScaleSRMLayer(nn.Module):
    """
    Multi-Scale Spatial Rich Model (MS-SRM) High-Pass Frequency Residual Extractor.
    Extracts high-frequency noise residuals across 3x3, 5x5, and 7x7 spatial filter kernels.
    Kernels are mathematically constructed as exact zero-sum spatial derivative filters.
    """
    def __init__(self):
        super().__init__()
        # 1. Standard 3x3 SRM High-Pass Kernels (Zero-sum)
        f1_3 = [[0, 0, 0], [0, 1, -1], [0, 0, 0]]
        f2_3 = [[0, 1, 0], [0, -1, 0], [0, 0, 0]]
        f3_3 = [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]
        filters_3 = torch.tensor([f1_3, f2_3, f3_3], dtype=torch.float32)

        # 2. 3 Distinct 5x5 SRM High-Pass Filters (Exact Zero-Sum)
        f1_5 = [
            [ 0,  0, -1,  0,  0],
            [ 0, -1,  2, -1,  0],
            [-1,  2,  0,  2, -1],
            [ 0, -1,  2, -1,  0],
            [ 0,  0, -1,  0,  0]
        ]
        f2_5 = [
            [-1,  2, -2,  2, -1],
            [ 2, -4,  4, -4,  2],
            [-2,  4, -4,  4, -2],
            [ 2, -4,  4, -4,  2],
            [-1,  2, -2,  2, -1]
        ]
        f3_5 = [
            [-1,  0,  2,  0, -1],
            [ 0, -2,  4, -2,  0],
            [ 2,  4,-12,  4,  2],
            [ 0, -2,  4, -2,  0],
            [-1,  0,  2,  0, -1]
        ]
        filters_5 = torch.tensor([f1_5, f2_5, f3_5], dtype=torch.float32)

        # 3. 3 Distinct 7x7 SRM High-Pass Filters (Exact Zero-Sum)
        f1_7 = [
            [ 0,  0,  0, -1,  0,  0,  0],
            [ 0,  0, -1,  2, -1,  0,  0],
            [ 0, -1,  2, -4,  2, -1,  0],
            [-1,  2, -4, 12, -4,  2, -1],
            [ 0, -1,  2, -4,  2, -1,  0],
            [ 0,  0, -1,  2, -1,  0,  0],
            [ 0,  0,  0, -1,  0,  0,  0]
        ]
        f2_7 = [
            [-1,  1, -1,  2, -1,  1, -1],
            [ 1, -2,  2, -4,  2, -2,  1],
            [-1,  2, -3,  6, -3,  2, -1],
            [ 2, -4,  6, -8,  6, -4,  2],
            [-1,  2, -3,  6, -3,  2, -1],
            [ 1, -2,  2, -4,  2, -2,  1],
            [-1,  1, -1,  2, -1,  1, -1]
        ]
        f3_7 = [
            [-1,  0,  0,  2,  0,  0, -1],
            [ 0, -2,  0,  4,  0, -2,  0],
            [ 0,  0, -3,  6, -3,  0,  0],
            [ 2,  4,  6,-24,  6,  4,  2],
            [ 0,  0, -3,  6, -3,  0,  0],
            [ 0, -2,  0,  4,  0, -2,  0],
            [-1,  0,  0,  2,  0,  0, -1]
        ]
        filters_7 = torch.tensor([f1_7, f2_7, f3_7], dtype=torch.float32)

        # Build depthwise convolutions for RGB channels
        self.conv_3x3 = nn.Conv2d(3, 9, kernel_size=3, padding=1, bias=False, groups=3)
        self.conv_5x5 = nn.Conv2d(3, 9, kernel_size=5, padding=2, bias=False, groups=3)
        self.conv_7x7 = nn.Conv2d(3, 9, kernel_size=7, padding=3, bias=False, groups=3)

        # Weights assignment
        w3 = torch.zeros(9, 1, 3, 3)
        for i in range(3):
            for j in range(3):
                w3[i * 3 + j, 0, :, :] = filters_3[j] / (4.0 if j == 2 else 1.0)
        self.conv_3x3.weight.data.copy_(w3)
        self.conv_3x3.weight.requires_grad = False

        w5 = torch.zeros(9, 1, 5, 5)
        for i in range(3):
            for j in range(3):
                w5[i * 3 + j, 0, :, :] = filters_5[j] / max(1.0, torch.abs(filters_5[j]).sum() / 2.0)
        self.conv_5x5.weight.data.copy_(w5)
        self.conv_5x5.weight.requires_grad = False

        w7 = torch.zeros(9, 1, 7, 7)
        for i in range(3):
            for j in range(3):
                w7[i * 3 + j, 0, :, :] = filters_7[j] / max(1.0, torch.abs(filters_7[j]).sum() / 2.0)
        self.conv_7x7.weight.data.copy_(w7)
        self.conv_7x7.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res3 = self.conv_3x3(x)
        res5 = self.conv_5x5(x)
        res7 = self.conv_7x7(x)
        return torch.cat([res3, res5, res7], dim=1) # 27 channels output


class SpatialFrequencyCrossAttention(nn.Module):
    """
    Resolution-Preserving Spatial-Frequency Cross-Attention (SFCA) Module.
    Attends spatial RGB feature maps to MS-SRM frequency residual maps.
    """
    def __init__(self, spatial_dim: int, freq_dim: int, embed_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.query_proj = nn.Conv2d(spatial_dim, embed_dim, kernel_size=1)
        self.key_proj = nn.Conv2d(freq_dim, embed_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(freq_dim, embed_dim, kernel_size=1)
        self.scale = self.head_dim ** -0.5

        # 1x1 projection back to spatial_dim without spatial stride
        self.out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, spatial_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(spatial_dim),
        )
        self.norm = nn.BatchNorm2d(spatial_dim)

    def forward(self, rgb_map: torch.Tensor, freq_map: torch.Tensor) -> torch.Tensor:
        B, C_rgb, H_rgb, W_rgb = rgb_map.shape

        # Align frequency map to spatial RGB resolution if different
        if freq_map.shape[2:] != (H_rgb, W_rgb):
            f_map = F.interpolate(freq_map, size=(H_rgb, W_rgb), mode='bilinear', align_corners=False)
        else:
            f_map = freq_map
        q_map = rgb_map

        N = H_rgb * W_rgb

        Q = self.query_proj(q_map).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2).contiguous() # [B, heads, N, head_dim]
        K = self.key_proj(f_map).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2).contiguous()   # [B, heads, N, head_dim]
        V = self.value_proj(f_map).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2).contiguous() # [B, heads, N, head_dim]

        # Accelerated scaled dot product attention (PyTorch 2.0+ SDPA)
        out_attn = F.scaled_dot_product_attention(Q, K, V)                                                   # [B, heads, N, head_dim]
        context = out_attn.permute(0, 2, 1, 3).contiguous().view(B, N, self.embed_dim).permute(0, 2, 1).contiguous().view(B, self.embed_dim, H_rgb, W_rgb)
        attended = self.out_proj(context)                                                                     # [B, C_rgb, H_rgb, W_rgb]

        return self.norm(rgb_map + attended)


class EnhancedHead(nn.Module):
    def __init__(self, in_features, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fc1(x)
        # Avoid BatchNorm1d channel mean/variance crash when batch_size == 1 during training
        if x.size(0) > 1 or not self.training:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out, x


class DeepfakeModel(nn.Module):
    def __init__(self, model_name, pretrained=True, branch_mode="fusion", model_variant="fusion"):
        super().__init__()
        self.branch_mode = branch_mode
        self.model_variant = model_variant

        # Register static normalization buffers (ImageNet mean & std)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 1. RGB Backbone
        self.rgb_backbone = None
        rgb_features = 0
        if self.branch_mode in ["rgb", "fusion"] or self.model_variant == "rgb_only":
            sd_prob = getattr(config, "STOCHASTIC_DEPTH_PROB", 0.35)
            if model_name == "efficientnet_b0":
                weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                self.rgb_backbone = models.efficientnet_b0(weights=weights, stochastic_depth_prob=sd_prob).features
                rgb_features = 1280
            elif model_name == "efficientnet_b4":
                weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
                self.rgb_backbone = models.efficientnet_b4(weights=weights, stochastic_depth_prob=sd_prob).features
                rgb_features = 1792
            elif model_name == "convnext_tiny":
                weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
                self.rgb_backbone = models.convnext_tiny(weights=weights).features
                rgb_features = 768
            elif model_name == "mobilenet_v3_small":
                weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
                self.rgb_backbone = models.mobilenet_v3_small(weights=weights).features
                rgb_features = 576
            elif model_name == "resnet18":
                m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
                self.rgb_backbone = nn.Sequential(*list(m.children())[:-2])
                rgb_features = 512
            elif model_name == "resnet50":
                m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
                self.rgb_backbone = nn.Sequential(*list(m.children())[:-2])
                rgb_features = 2048
            elif model_name == "shufflenet_v2":
                m = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None)
                self.rgb_backbone = nn.Sequential(m.conv1, m.maxpool, m.stage2, m.stage3, m.stage4, m.conv5)
                rgb_features = 1024
            elif model_name == "densenet121":
                self.rgb_backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None).features
                rgb_features = 1024
            else:
                raise ValueError(f"Unsupported backbone: {model_name}")

        # 2. Resolution-Preserving Frequency Branch (MS-SRM + CNN outputting 16x16 feature maps)
        if self.branch_mode in ["freq", "fusion"] and self.model_variant != "rgb_only":
            self.srm = MultiScaleSRMLayer()
            self.srm_norm = nn.BatchNorm2d(27, affine=True)
            self.freq_convs = nn.Sequential(
                nn.Conv2d(27, 32, kernel_size=3, stride=2, padding=1, bias=False),   # 256 -> 128
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),   # 128 -> 64
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 64 -> 32
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False), # 32 -> 16
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            freq_features = 128
        else:
            freq_features = 0

        # Freeze backbone parameters if progressive unfreeze configured
        backbone_to_freeze = self.rgb_backbone if self.rgb_backbone is not None else getattr(self, "freq_convs", None)
        if backbone_to_freeze is not None:
            num_features = len(backbone_to_freeze)
            num_freeze = int(num_features * config.FREEZE_PERCENT) if getattr(config, "PROGRESSIVE_UNFREEZE", False) else 0
            for layer in backbone_to_freeze[:num_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Spatial-Frequency Cross-Attention Module for Fusion Mode
        if self.branch_mode == "fusion" and self.model_variant in ["fusion", "modular_order"]:
            self.sfca = SpatialFrequencyCrossAttention(spatial_dim=rgb_features, freq_dim=freq_features, embed_dim=256)
        else:
            self.sfca = None

        # CoRe-DF Modular Architecture Components
        if self.model_variant == "modular_order":
            self.num_experts = 6 # jpeg, downscale, motion_blur, gaussian_blur, gaussian_noise, sharpen
            
            # 1. Dynamic Routing Gating Network
            self.router = DynamicRouter(feature_dim=rgb_features + freq_features, num_experts=self.num_experts, top_k=2)
            
            # 2. Modular Expert Heads (taking SFCA-attended RGB map as input)
            self.expert_heads = nn.ModuleList([
                ModularExpertHead(in_channels=rgb_features, reduced_dim=64) for _ in range(self.num_experts)
            ])
            
            # 3. Order Inference Module (takes global pooled RGB+SRM features)
            self.order_inference = OrderInferenceModule(feature_dim=rgb_features + freq_features, num_experts=self.num_experts)
            
            # 4. Order-Conditioned Fusion
            self.order_fusion = OrderConditionedFusion(expert_feature_dim=64)
            
            # Final Classification Head takes the fused expert features (64-dim)
            total_features = 64
        else:
            # Standard Fusion & Classification Head
            if self.branch_mode == "rgb" or self.model_variant == "rgb_only":
                total_features = rgb_features
            elif self.branch_mode == "freq":
                total_features = freq_features
            else:
                total_features = rgb_features + freq_features

        self.head = EnhancedHead(total_features, config.DROPOUT)

    @property
    def features(self):
        return self.rgb_backbone if self.rgb_backbone is not None else self.freq_convs

    def forward(self, x):
        features = []

        if self.branch_mode == "rgb" or self.model_variant == "rgb_only":
            rgb_map = self.rgb_backbone(x)
            rgb_f = F.adaptive_avg_pool2d(rgb_map, 1).flatten(1)
            features.append(rgb_f)
        elif self.branch_mode == "freq":
            x_raw = x * self.std + self.mean
            freq_x = self.srm(x_raw)
            freq_x = self.srm_norm(freq_x)
            freq_map = self.freq_convs(freq_x)
            freq_f = F.adaptive_avg_pool2d(freq_map, 1).flatten(1)
            features.append(freq_f)
        elif self.branch_mode == "fusion":
            rgb_map = self.rgb_backbone(x)
            if self.model_variant == "fusion":
                x_raw = x * self.std + self.mean
                freq_x = self.srm(x_raw)
                freq_x = self.srm_norm(freq_x)
                freq_map = self.freq_convs(freq_x)
                attended_rgb_map = self.sfca(rgb_map, freq_map)

                rgb_f = F.adaptive_avg_pool2d(attended_rgb_map, 1).flatten(1)
                freq_f = F.adaptive_avg_pool2d(freq_map, 1).flatten(1)
                features.append(rgb_f)
                features.append(freq_f)
            elif self.model_variant == "fusion_no_attn":
                x_raw = x * self.std + self.mean
                freq_x = self.srm(x_raw)
                freq_x = self.srm_norm(freq_x)
                freq_map = self.freq_convs(freq_x)

                rgb_f = F.adaptive_avg_pool2d(rgb_map, 1).flatten(1)
                freq_f = F.adaptive_avg_pool2d(freq_map, 1).flatten(1)
                features.append(rgb_f)
                features.append(freq_f)

            elif self.model_variant == "modular_order":
                x_raw = x * self.std + self.mean
                freq_x = self.srm(x_raw)
                freq_x = self.srm_norm(freq_x)
                freq_map = self.freq_convs(freq_x)
                attended_rgb_map = self.sfca(rgb_map, freq_map)
                
                rgb_f = F.adaptive_avg_pool2d(attended_rgb_map, 1).flatten(1)
                freq_f = F.adaptive_avg_pool2d(freq_map, 1).flatten(1)
                global_fused = torch.cat([rgb_f, freq_f], dim=1)
                
                # 1. Dynamic Routing
                top_k_weights, top_k_indices, routing_weights = self.router(global_fused)
                
                # 2. Order Inference and Validity
                order_embs, order_logits, expert_validity_scores = self.order_inference(global_fused)
                
                # 3. Modular Expert Heads (Sparse Execution)
                B = x.size(0)
                expert_features = torch.zeros(B, self.num_experts, 64, device=x.device)
                for i in range(self.num_experts):
                    mask = (top_k_indices == i).any(dim=-1)
                    if mask.any():
                        expert_out = self.expert_heads[i](attended_rgb_map[mask])
                        expert_features[mask, i] = expert_out
                
                # 4. Order-Conditioned Fusion
                fused = self.order_fusion(expert_features, order_embs, top_k_weights, top_k_indices)
                
                # 5. Final Classification
                out, feat = self.head(fused)
                
                return {
                    "deepfake_logit": out,
                    "order_logits": order_logits,
                    "expert_validity_scores": expert_validity_scores,
                    "feat": feat
                }

        fused = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        out, feat = self.head(fused)
        return out, feat


# The only unambiguous model_variant -> branch_mode mapping (see resolve_checkpoint_model_kwargs
# docstring for why the reverse direction is NOT similarly inferred).
_NATURAL_BRANCH_MODE_FOR_VARIANT = {"rgb_only": "rgb", "fusion": "fusion", "fusion_no_attn": "fusion", "modular_order": "fusion"}


def resolve_checkpoint_model_kwargs(checkpoint=None):
    """Resolve (model_name, model_variant, branch_mode) for reconstructing a model.

    This is the single authoritative mechanism for deciding a model's architecture fields when
    reconstructing it from a checkpoint (training resume, evaluation, or recovery after a
    failed load). Fallback hierarchy, per field, in order:

      1. Top-level checkpoint key (e.g. checkpoint["model_variant"]), if present and not None.
      2. The nested checkpoint["configuration"] dict's matching key, if present and not None.
      3. For branch_mode ONLY: if the checkpoint specifies model_variant (via 1 or 2) but not
         branch_mode, infer branch_mode from that model_variant using the natural, unambiguous
         mapping {"rgb_only": "rgb", "fusion": "fusion", "fusion_no_attn": "fusion", "modular_order": "fusion"}. This is
         deliberately preferred over falling back to config.py's current BRANCH_MODE, which may
         belong to an unrelated experiment and would otherwise make resolution for the (very
         common) "checkpoint has model_variant but predates the branch_mode metadata field"
         case non-deterministic across machines/environments.
      4. config.py's current value (MODEL_NAME / MODEL_VARIANT / BRANCH_MODE), for whichever
         checkpoint fields are entirely absent (legacy checkpoints with no self-describing
         metadata at all, or no checkpoint given).

    Note the asymmetry: a missing model_variant is NOT inferred from a present branch_mode,
    because branch_mode='fusion' is inherently ambiguous between model_variant 'fusion' and
    'fusion_no_attn' - there is no unambiguous reverse mapping, so that case always falls
    through to config.py (step 4).

    This does not itself validate consistency between the resolved fields (e.g. a checkpoint
    could carry contradictory explicit model_variant/branch_mode metadata) - build_model()
    performs that validation and raises a clear ValueError, and a genuine architecture
    mismatch against the checkpoint's saved state_dict keys will raise loudly via
    model.load_state_dict()'s strict key checking. Neither path can silently produce a
    working-but-wrong model, since DeepfakeModel's forward() branches on the exact same
    (branch_mode, model_variant) pair used to decide which submodules get constructed.
    """
    cfg = {}
    if checkpoint is not None:
        cfg = checkpoint.get("configuration", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {}

    def _present(key):
        in_top_level = checkpoint is not None and key in checkpoint and checkpoint[key] is not None
        in_cfg = key in cfg and cfg[key] is not None
        return in_top_level or in_cfg

    def _get(key, default):
        if checkpoint is not None and key in checkpoint and checkpoint[key] is not None:
            return checkpoint[key]
        if key in cfg and cfg[key] is not None:
            return cfg[key]
        return default

    model_name = _get("model_name", getattr(config, "MODEL_NAME", "efficientnet_b0"))
    model_variant = _get("model_variant", getattr(config, "MODEL_VARIANT", "fusion"))

    if _present("branch_mode"):
        branch_mode = _get("branch_mode", getattr(config, "BRANCH_MODE", "fusion"))
    elif _present("model_variant"):
        branch_mode = _NATURAL_BRANCH_MODE_FOR_VARIANT.get(model_variant, getattr(config, "BRANCH_MODE", "fusion"))
    else:
        branch_mode = getattr(config, "BRANCH_MODE", "fusion")

    return model_name, model_variant, branch_mode


def build_model(model_name=None, pretrained=None, branch_mode=None, model_variant=None):
    if model_name is None:
        model_name = getattr(config, 'MODEL_NAME', 'efficientnet_b0')
    if pretrained is None:
        pretrained = getattr(config, 'PRETRAINED', True)
    if branch_mode is None:
        branch_mode = getattr(config, 'BRANCH_MODE', 'fusion')
    if model_variant is None:
        model_variant = getattr(config, 'MODEL_VARIANT', 'fusion')

    valid_branch_modes = {"rgb", "freq", "fusion"}
    valid_variants = {"fusion", "rgb_only", "fusion_no_attn", "modular_order"}

    if branch_mode not in valid_branch_modes:
        raise ValueError(f"Invalid branch_mode '{branch_mode}'. Allowed: {valid_branch_modes}")
    if model_variant not in valid_variants:
        raise ValueError(f"Invalid model_variant '{model_variant}'. Allowed: {valid_variants}")
    if branch_mode == "rgb" and model_variant not in {"rgb_only", "fusion"}:
        raise ValueError(f"Contradictory configuration: branch_mode='rgb' with model_variant='{model_variant}'")
    if model_variant == "rgb_only" and branch_mode == "freq":
        raise ValueError("Contradictory configuration: model_variant='rgb_only' cannot be paired with branch_mode='freq'")

    return DeepfakeModel(model_name, pretrained, branch_mode=branch_mode, model_variant=model_variant)

