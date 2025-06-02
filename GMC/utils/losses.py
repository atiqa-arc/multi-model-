import torch
import torch.nn as nn
import torch.nn.functional as F

def kl_divergence(mu, sigma):
    """
    KL divergence between N(mu, sigma^2) and standard normal N(0,1)
    Formula: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    kl = -0.5 * torch.sum(1 + torch.log(sigma ** 2 + 1e-8) - mu ** 2 - sigma ** 2, dim=1)
    return kl.mean()
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_embeds, text_embeds):
        """
        image_embeds: (B, D)
        text_embeds:  (B, D)
        """
        batch_size = image_embeds.size(0)

        # Normalize vectors for cosine similarity
        image_embeds = F.normalize(image_embeds, dim=1)
        text_embeds = F.normalize(text_embeds, dim=1)

        # Similarity matrix (B x B)
        logits = torch.matmul(image_embeds, text_embeds.T) / self.temperature

        labels = torch.arange(batch_size).to(image_embeds.device)

        # Cross-entropy: correct pairs are diagonal (i = j)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2
