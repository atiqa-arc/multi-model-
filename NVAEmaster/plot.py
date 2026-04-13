import os
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import utils
from model import AutoEncoder


COL_NORMAL = "#7D5FFF"      # purple
COL_SLIGHT = "#F39C34"      # orange
COL_STRONG = "#D94841"      # red-orange
COL_UNC = "black"


class ImageOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_root, image_size=256):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root

        if "chestxrays" in self.df.columns:
            self.image_col = "chestxrays"
        elif "images" in self.df.columns:
            self.image_col = "images"
        else:
            raise KeyError("Expected image column 'chestxrays' or 'images'.")

        self.transform = transforms.Compose([
            transforms.Resize(257),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.iloc[idx][self.image_col]
        full_path = os.path.join(self.image_root, rel_path)

        image = Image.open(full_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transform(image)
        return image, rel_path


def decode_mean(output):
    if hasattr(output, "mean") and callable(output.mean):
        return output.mean()
    if hasattr(output, "mean"):
        return output.mean
    if hasattr(output, "dist") and hasattr(output.dist, "mu"):
        return torch.clamp(output.dist.mu, -1.0, 1.0) / 2.0 + 0.5
    return output.sample()


def extract_mu_latent(model, x):
    logits, log_q, log_p, kl_all, kl_diag, _, mu_groups = model(
        x, return_latents=True, return_posterior_means=True
    )
    mu = torch.cat(
        [F.adaptive_avg_pool2d(mu_g, (1, 1)).flatten(1) for mu_g in mu_groups],
        dim=1,
    )
    return logits, mu


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    nvae_args = ckpt["args"]
    arch_instance = utils.get_arch_cells(nvae_args.arch_instance)

    model = AutoEncoder(nvae_args, writer=None, arch_instance=arch_instance).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    if hasattr(model, "set_mc_dropout"):
        model.set_mc_dropout(True)

    return model, ckpt, nvae_args


def reduce_to_2d(features, method="tsne", perplexity=30):
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        return reducer.fit_transform(features)

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto"
    )
    return reducer.fit_transform(features)


def minmax_normalize(x):
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < 1e-12:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def assign_margin_group(score):
    if 0.10 <= score <= 0.20:
        return "normal"
    if 0.30 <= score <= 0.60:
        return "slightly abnormal"
    if score >= 0.70:
        return "strongly abnormal"
    return None


@torch.no_grad()
def plot_svdd_boundary(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    nvae_args = ckpt["args"]
    arch_instance = utils.get_arch_cells(nvae_args.arch_instance)

    model = AutoEncoder(nvae_args, writer=None, arch_instance=arch_instance).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    svdd_center = ckpt["svdd_center"].to(device).float()

    ds = ImageOnlyDataset(args.csv_train, args.data_root, image_size=args.image_size)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feats = []
    for images, _ in dl:
        images = images.to(device, non_blocking=True)
        x = utils.pre_process(images, nvae_args.num_x_bits)
        _, z = extract_mu_latent(model, x)
        feats.append(z.cpu())

    feats = torch.cat(feats, dim=0).numpy()
    center = svdd_center.cpu().numpy().reshape(1, -1)

    pca = PCA(n_components=2, random_state=42)
    feats_2d = pca.fit_transform(feats)
    center_2d = feats_2d.mean(axis=0)


    dists_2d = np.sum((feats_2d - center_2d[None, :]) ** 2, axis=1)
    radius_2d = np.sqrt(np.percentile(dists_2d, args.boundary_percentile))

    plt.figure(figsize=(6, 6))

    circle = plt.Circle(
        center_2d,
        radius_2d,
        facecolor="skyblue",
        edgecolor="teal",
        linestyle="--",
        linewidth=1.5,
        alpha=0.35
    )
    plt.gca().add_patch(circle)

    plt.scatter(
        feats_2d[:, 0], feats_2d[:, 1],
        c="teal", s=10, alpha=0.8, edgecolors="none", label="Traning Data"
    )

    plt.scatter(
        center_2d[0], center_2d[1],
        c="black", s=80, marker="*", label="Center"
    )

    plt.xlim(center_2d[0] - radius_2d * 1.1, center_2d[0] + radius_2d * 1.1)
    plt.ylim(center_2d[1] - radius_2d * 1.1, center_2d[1] + radius_2d * 1.1)
    plt.gca().set_aspect("equal", adjustable="box")

    #plt.xlabel("PC 1")
    #plt.ylabel("PC 2")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(args.output_train, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {args.output_train}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model, ckpt, nvae_args = load_model(args.checkpoint, device)

    dataset = ImageOnlyDataset(
        csv_path=args.csv_test,
        image_root=args.data_root,
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    svdd_center = ckpt.get("svdd_center", None)
    if svdd_center is not None:
        svdd_center = svdd_center.to(device).float()

    alpha_score = float(getattr(nvae_args, "alpha_score", 1.0))
    beta_score = float(getattr(nvae_args, "beta_score", 1.0))
    gamma_score = float(getattr(nvae_args, "gamma_score", 1.0))

    all_features = []
    all_uncertainty = []
    all_fused_scores = []
    all_paths = []

    for images, image_paths in loader:
        images = images.to(device, non_blocking=True)
        x = utils.pre_process(images, nvae_args.num_x_bits)

        mu_passes = []
        xhat_passes = []

        for _ in range(args.mc_passes):
            logits, mu = extract_mu_latent(model, x)
            output = model.decoder_output(logits)

            xhat = decode_mean(output)
            mu_passes.append(mu)
            xhat_passes.append(xhat)

        mu_stack = torch.stack(mu_passes, dim=0)
        xhat_stack = torch.stack(xhat_passes, dim=0)

        mu_mean = mu_stack.mean(dim=0)
        xhat_mean = xhat_stack.mean(dim=0)

        recon_score = torch.mean((x - xhat_mean) ** 2, dim=(1, 2, 3))
        unc_score = (
            mu_stack.var(dim=0, unbiased=False).mean(dim=1) +
            xhat_stack.var(dim=0, unbiased=False).mean(dim=(1, 2, 3))
        )

        if svdd_center is not None:
            svdd_score = torch.sum((mu_mean - svdd_center.unsqueeze(0)) ** 2, dim=1)
        else:
            svdd_score = torch.zeros_like(recon_score)

        fused_score = (
            alpha_score * recon_score +
            beta_score * svdd_score +
            gamma_score * unc_score
        )

        all_features.append(mu_mean.cpu())
        all_uncertainty.append(unc_score.cpu())
        all_fused_scores.append(fused_score.cpu())
        all_paths.extend(list(image_paths))

    features = torch.cat(all_features, dim=0).numpy()
    uncertainty = torch.cat(all_uncertainty, dim=0).numpy()
    fused_scores_raw = torch.cat(all_fused_scores, dim=0).numpy()

    score_01 = minmax_normalize(fused_scores_raw)
    points_2d = reduce_to_2d(features, method=args.method, perplexity=args.perplexity)

    groups = np.array([assign_margin_group(s) for s in score_01], dtype=object)
    selected_mask = np.array([g is not None for g in groups])
    points_2d = points_2d[selected_mask]
    uncertainty = uncertainty[selected_mask]
    fused_scores_raw = fused_scores_raw[selected_mask]
    score_01 = score_01[selected_mask]
    groups = groups[selected_mask]
    all_paths = np.array(all_paths)[selected_mask]
    
    high_unc_mask = uncertainty >= np.percentile(uncertainty, args.uncertainty_percentile)
    normal_mask = groups == "normal"
    slight_mask = groups == "slightly abnormal"
    strong_mask = groups == "strongly abnormal"

    plt.figure(figsize=(6.4, 5.4))

    plt.scatter(points_2d[normal_mask, 0], points_2d[normal_mask, 1],
                c=COL_NORMAL, s=18, alpha=0.85, edgecolors="none", label="Normal")

    plt.scatter(points_2d[slight_mask, 0], points_2d[slight_mask, 1],
                c=COL_SLIGHT, s=20, alpha=0.90, edgecolors="none", label="Slightly abnormal")

    plt.scatter(points_2d[strong_mask, 0], points_2d[strong_mask, 1],
                c=COL_STRONG, s=22, alpha=0.95, edgecolors="none", label="Strongly abnormal")

    plt.scatter(points_2d[high_unc_mask, 0], points_2d[high_unc_mask, 1],
                c=COL_UNC, s=42, marker="x", linewidths=1.2, label="High uncertainty")

   
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    out_dir = os.path.dirname(args.output_test)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output_test, dpi=300, bbox_inches="tight")
    plt.close()

    csv_out = args.output_test.replace(".png", "_points.csv")
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path", "x", "y",
            "raw_score", "normalized_score",
            "uncertainty", "group", "high_uncertainty"
        ])
        for i in range(len(all_paths)):
            writer.writerow([
                all_paths[i],
                float(points_2d[i, 0]),
                float(points_2d[i, 1]),
                float(fused_scores_raw[i]),
                float(score_01[i]),
                float(uncertainty[i]),
                groups[i],
                int(high_unc_mask[i]),
            ])

    print(f"Saved figure to: {args.output_test}")
    print(f"Saved 2D points to: {csv_out}")
    print(f"Normal: {int(normal_mask.sum())}")
    print(f"Slightly abnormal: {int(slight_mask.sum())}")
    print(f"Strongly abnormal: {int(strong_mask.sum())}")
    print(f"High uncertainty: {int(high_unc_mask.sum())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NVAE score-margin 2D feature visualization")
    parser.add_argument("--checkpoint", default="/public/ATIQA/NVAEmaster/eval-exp/checkpoint.pt")
    parser.add_argument("--data_root", default="/public/ATIQA/Datasets/iu_xray/")
    
    parser.add_argument("--csv_test", default="/public/ATIQA/Datasets/iu_xray/test_pairs.csv")
    parser.add_argument("--csv_train", default="/public/ATIQA/Datasets/iu_xray/train_pairs.csv")

    parser.add_argument("--output_test", default="/public/ATIQA/NVAEmaster/eval-exp/test_feature_margin_2d.png")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mc_passes", type=int, default=10)
    parser.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--uncertainty_percentile", type=float, default=95.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output_train", default="/public/ATIQA/NVAEmaster/eval-exp/train_svdd_boundary.png")
    parser.add_argument("--boundary_percentile", type=float, default=85.0)
    args = parser.parse_args()
    #main(args)
    plot_svdd_boundary(args)

