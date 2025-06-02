import torch
import torch.nn.functional as F
import numpy as np

def compute_cosine_sim_matrix(img_embeds, txt_embeds):
    """
    Compute cosine similarity matrix between all image and text embeddings.
    img_embeds: (N, D)
    txt_embeds: (N, D)
    Returns: sim_matrix: (N, N)
    """
    img_norm = F.normalize(img_embeds, dim=1)
    txt_norm = F.normalize(txt_embeds, dim=1)
    return torch.matmul(img_norm, txt_norm.T).cpu().numpy()

def retrieval_metrics(sim_matrix):
    """
    Compute Recall@1, Recall@5, Recall@10
    """
    def recall_at_k(sim_matrix, k):
        n = sim_matrix.shape[0]
        top_k = np.argsort(-sim_matrix, axis=1)[:, :k]
        correct = np.arange(n).reshape(-1, 1)
        return (top_k == correct).any(axis=1).mean()

    return {
        "R@1": round(recall_at_k(sim_matrix, 1) * 100, 2),
        "R@5": round(recall_at_k(sim_matrix, 5) * 100, 2),
        "R@10": round(recall_at_k(sim_matrix, 10) * 100, 2),
    }

def evaluate_retrieval(img_embeds, txt_embeds):
    """
    Evaluate image-to-text and text-to-image retrieval.
    """
    sim_i2t = compute_cosine_sim_matrix(img_embeds, txt_embeds)
    sim_t2i = compute_cosine_sim_matrix(txt_embeds, img_embeds)

    return {
        "Image→Text": retrieval_metrics(sim_i2t),
        "Text→Image": retrieval_metrics(sim_t2i)
    }
