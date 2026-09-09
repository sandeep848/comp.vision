import torch
import torch.nn as nn
import torch.nn.functional as F

class ModularExpertHead(nn.Module):
    """
    A lightweight expert head using depthwise-separable convolutions.
    Projects spatial features to 64 channels to keep parameters < 100K per expert.
    """
    def __init__(self, in_channels, reduced_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, reduced_dim, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(reduced_dim)
        self.relu = nn.ReLU(inplace=True)
        
        self.dw_conv = nn.Conv2d(reduced_dim, reduced_dim, kernel_size=3, padding=1, groups=reduced_dim, bias=False)
        self.dw_bn = nn.BatchNorm2d(reduced_dim)
        self.pw_conv = nn.Conv2d(reduced_dim, reduced_dim, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(reduced_dim)
        
    def forward(self, x):
        x = self.relu(self.bn_proj(self.proj(x)))
        x = self.relu(self.dw_bn(self.dw_conv(x)))
        x = self.relu(self.pw_bn(self.pw_conv(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x

class DynamicRouter(nn.Module):
    """
    Gating network that selects the top-k experts for each input.
    """
    def __init__(self, feature_dim, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(feature_dim, num_experts)
        
    def forward(self, global_features):
        logits = self.router(global_features)
        routing_weights = F.softmax(logits, dim=-1)
        
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return top_k_weights, top_k_indices, routing_weights

class OrderInferenceModule(nn.Module):
    """
    Infers the relative order of operators and learns expert validity.
    Outputs order embeddings, pairwise logits, and validity scores.
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
        
        # Learned reliability head (Issue 7)
        self.validity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, global_features):
        B = global_features.size(0)
        context = self.proj(global_features).unsqueeze(1) # (B, 1, d_model)
        tokens = self.expert_tokens.expand(B, -1, -1) # (B, num_experts, d_model)
        
        x = torch.cat([context, tokens], dim=1)
        x = self.transformer(x)
        
        expert_embs = x[:, 1:, :] # (B, num_experts, d_model)
        
        # Order Logits
        order_logits = torch.zeros(B, self.num_experts, self.num_experts, device=global_features.device)
        for i in range(self.num_experts):
            for j in range(self.num_experts):
                if i != j:
                    pair = torch.cat([expert_embs[:, i, :], expert_embs[:, j, :]], dim=1)
                    order_logits[:, i, j] = self.order_classifier(pair).squeeze(-1)
                    
        # Validity Scores
        validity_scores = torch.sigmoid(self.validity_head(expert_embs).squeeze(-1)) # (B, num_experts)
                    
        return expert_embs, order_logits, validity_scores

class OrderConditionedFusion(nn.Module):
    """
    Fuses the top-k expert outputs using sparse routing weights.
    """
    def __init__(self, expert_feature_dim=64, d_model=64):
        super().__init__()
        self.expert_feature_dim = expert_feature_dim
        self.d_model = d_model
        
        self.query_proj = nn.Linear(d_model, expert_feature_dim)
        self.attention = nn.MultiheadAttention(embed_dim=expert_feature_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(expert_feature_dim)
        
    def forward(self, expert_features, order_embeddings, top_k_weights, top_k_indices):
        # expert_features: (B, num_experts, expert_feature_dim)
        queries = self.query_proj(order_embeddings) # (B, num_experts, expert_feature_dim)
        
        attn_out, _ = self.attention(queries, expert_features, expert_features)
        attn_out = self.norm(expert_features + attn_out) # (B, num_experts, expert_feature_dim)
        
        # Sparse weighted fusion
        B = expert_features.size(0)
        fused = torch.zeros(B, self.expert_feature_dim, device=expert_features.device)
        for i in range(B):
            indices = top_k_indices[i]
            weights = top_k_weights[i]
            fused[i] = (attn_out[i, indices] * weights.unsqueeze(-1)).sum(dim=0)
            
        return fused
