import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftClusterGaussianHead(nn.Module):
    def __init__(self, input_dim, num_clusters=3):
        super(SoftClusterGaussianHead, self).__init__()
        self.num_clusters = num_clusters
        self.input_dim = input_dim

        # Linear layer to produce logits for cluster assignment
        self.assignment_layer = nn.Linear(input_dim, num_clusters)

    def forward(self, x):
        """
        x: Tensor of shape [B, N, D] where
           B = batch size,
           N = number of regions or tokens,
           D = feature dimension
        """
        B, N, D = x.shape

        # Cluster logits and soft assignments
        cluster_logits = self.assignment_layer(x)              # [B, N, K]
        cluster_probs = F.softmax(cluster_logits, dim=-1)      # [B, N, K]

        # Compute soft mean (mu_k)
        probs = cluster_probs.permute(0, 2, 1)                 # [B, K, N]
        x_exp = x.unsqueeze(1)                                 # [B, 1, N, D]
        mu_k = torch.sum(probs.unsqueeze(-1) * x_exp, dim=2)  # [B, K, D]

        # Compute soft variance (sigma_k)
        diff = x_exp - mu_k.unsqueeze(2)                       # [B, K, N, D]
        sigma_k = torch.sum(probs.unsqueeze(-1) * diff**2, dim=2)  # [B, K, D]

        # Reparameterization trick: sample z ~ N(mu, sigma)
        eps = torch.randn_like(sigma_k)
        z_k = mu_k + eps * torch.sqrt(sigma_k + 1e-6)         # [B, K, D]

        # Compute KL divergence with standard Gaussian N(0,1)
        kl = -0.5 * torch.sum(1 + torch.log(sigma_k + 1e-6) - mu_k**2 - sigma_k, dim=-1)  # [B, K]
        kl_loss = kl.mean()                                   # scalar

        # Optionally return mean over clusters as final embedding
        pooled_z = z_k.mean(dim=1)                            # [B, D]

        return pooled_z, kl_loss
