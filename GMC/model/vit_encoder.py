import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

# 🔧 Graph Feature Enhancement Layer Using MLP-Based Attention
class GraphFeatureEnhancer(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GraphFeatureEnhancer, self).__init__()
        self.mlp1 = nn.Linear(in_dim, hidden_dim)
        self.mlp2 = nn.Linear(in_dim, hidden_dim)
        self.mlp3 = nn.Linear(in_dim, hidden_dim)
        self.mlp4 = nn.Linear(hidden_dim, in_dim)

    def forward(self, Vf):
        """
        Vf: [B, m, D] where m is number of region features (e.g., 36)
        """
        B, m, D = Vf.shape

        q = self.mlp1(Vf)                         # [B, m, H]
        k = self.mlp2(Vf)                         # [B, m, H]
        c = torch.bmm(q, k.transpose(1, 2)) / m   # [B, m, m], attention scores
        c = F.softmax(c, dim=-1)                  # Normalize edge weights

        agg = torch.bmm(c, Vf)                    # [B, m, D], weighted features
        out = self.mlp4(self.mlp3(agg))           # [B, m, D], transformed

        Ve = out + Vf                             # Residual connection
        return Ve


# ✅ Vision Transformer Encoder with Feature Enhancement
class ViTEncoder(nn.Module):
    def __init__(self, model_path='/public/atiqa/vit trandformer', proj_dim=512):
        super(ViTEncoder, self).__init__()

        # Load pretrained ViT from Hugging Face local path
        self.vit = ViTModel.from_pretrained(model_path)
        hidden_size = self.vit.config.hidden_size

        self.proj = nn.Linear(hidden_size, proj_dim)  # Linear projection to match feature dim
        self.enhancer = GraphFeatureEnhancer(proj_dim, proj_dim)  # Feature refinement layer

    def forward(self, pixel_values):
        """
        pixel_values: [B, 3, 224, 224] (preprocessed using ViTImageProcessor)
        Returns: enhanced patch token features [B, m, D]
        """
        outputs = self.vit(pixel_values=pixel_values)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # Remove CLS token, keep patch tokens [B, m, hidden]

        projected = self.proj(patch_tokens)        # [B, m, D]
        enhanced = self.enhancer(projected)        # [B, m, D]
        return enhanced
