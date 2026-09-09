import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig

class ModularExpertHead(nn.Module):
    """
    A lightweight expert head using a single MobileNetV3 block.
    Takes the spatial feature map, processes it, and outputs a 1D feature vector.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Config for a lightweight InvertedResidual block
        hidden_dim = in_channels * 4
        config = InvertedResidualConfig(in_channels, kernel=3, expanded_channels=hidden_dim, out_channels=in_channels, use_se=True, activation="RE", stride=1, dilation=1, width_mult=1.0)
        self.block = InvertedResidual(config, norm_layer=nn.BatchNorm2d)
        
    def forward(self, x):
        x = self.block(x)
        # Global average pool to 1D
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x

class OrderInferenceModule(nn.Module):
    """
    Infers the relative order of operators from the global pooled features.
    Outputs order embeddings and pairwise logits.
    """
    def __init__(self, feature_dim, num_experts, d_model=64, num_heads=4):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        
        self.proj = nn.Linear(feature_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.expert_tokens = nn.Parameter(torch.randn(1, num_experts, d_model))
        self.order_classifier = nn.Linear(d_model * 2, 1)

    def forward(self, global_features):
        B = global_features.size(0)
        context = self.proj(global_features).unsqueeze(1) # (B, 1, d_model)
        tokens = self.expert_tokens.expand(B, -1, -1) # (B, num_experts, d_model)
        
        x = torch.cat([context, tokens], dim=1) # (B, 1 + num_experts, d_model)
        x = self.transformer(x)
        
        expert_embs = x[:, 1:, :] # (B, num_experts, d_model)
        
        order_logits = torch.zeros(B, self.num_experts, self.num_experts, device=global_features.device)
        for i in range(self.num_experts):
            for j in range(self.num_experts):
                if i != j:
                    pair = torch.cat([expert_embs[:, i, :], expert_embs[:, j, :]], dim=1)
                    order_logits[:, i, j] = self.order_classifier(pair).squeeze(-1)
                    
        return expert_embs, order_logits

class OrderConditionedFusion(nn.Module):
    """
    Fuses the outputs of the multiple expert heads using Cross-Attention.
    """
    def __init__(self, expert_feature_dim, d_model=64):
        super().__init__()
        self.expert_feature_dim = expert_feature_dim
        self.d_model = d_model
        
        self.query_proj = nn.Linear(d_model, expert_feature_dim)
        self.attention = nn.MultiheadAttention(embed_dim=expert_feature_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(expert_feature_dim)
        
    def forward(self, expert_features, order_embeddings):
        # expert_features: (B, num_experts, expert_feature_dim)
        # order_embeddings: (B, num_experts, d_model)
        queries = self.query_proj(order_embeddings) # (B, num_experts, expert_feature_dim)
        attn_out, _ = self.attention(queries, expert_features, expert_features)
        
        out = self.norm(expert_features + attn_out)
        out = out.mean(dim=1) # (B, expert_feature_dim)
        
        return out
