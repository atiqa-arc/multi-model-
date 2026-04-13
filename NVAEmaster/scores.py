import argparse
import csv
import os

import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import utils
from model import AutoEncoder


class ScoreExportDataset(Dataset):
    def __init__(self, csv_path, image_root, image_size=256):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        if "chestxrays" in self.df.columns:
            self.image_col = "chestxrays"
        elif "images" in self.df.columns:
            self.image_col = "images"
        else:
            raise KeyError("Expected image column 'chestxrays' or 'images' in dataset CSV.")
        self.transform = transforms.Compose([
            transforms.Resize(257),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_rel_path = self.df.iloc[idx][self.image_col]
        image_path = os.path.join(self.image_root, image_rel_path)
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)
        return image, image_rel_path


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


@torch.no_grad()
def export_scores(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")

    nvae_args = checkpoint["args"]
    arch_instance = utils.get_arch_cells(nvae_args.arch_instance)
    model = AutoEncoder(nvae_args, writer=None, arch_instance=arch_instance).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    model.set_mc_dropout(True)

    svdd_center = checkpoint.get("svdd_center")
    if svdd_center is None:
        raise KeyError("Checkpoint is missing 'svdd_center'.")
    svdd_center = svdd_center.to(device).float()

    score_threshold = float(checkpoint.get("score_threshold", 0.0))
    alpha_score = float(getattr(nvae_args, "alpha_score", 1.0))
    beta_score = float(getattr(nvae_args, "beta_score", 1.0))
    gamma_score = float(getattr(nvae_args, "gamma_score", 1.0))
    mc_passes = args.mc_passes or int(getattr(nvae_args, "mc_passes", 5))

    split_to_csv = {
        "train": os.path.join(args.data_root, "train_pairs.csv"),
        "val": os.path.join(args.data_root, "val_pairs.csv"),
        "test": os.path.join(args.data_root, "test_pairs.csv"),
    }
    csv_path = args.csv_path or split_to_csv[args.split]

    dataset = ScoreExportDataset(csv_path=csv_path, image_root=args.data_root, image_size=256)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    rows = []
    for images, image_paths in loader:
        images = images.to(device, non_blocking=True)
        x = utils.pre_process(images, nvae_args.num_x_bits)

        mu_passes = []
        xhat_passes = []
        for _ in range(mc_passes):
            logits, mu = extract_mu_latent(model, x)
            output = model.decoder_output(logits)
            mu_passes.append(mu)
            xhat_passes.append(decode_mean(output))

        mu_stack = torch.stack(mu_passes, dim=0)
        xhat_stack = torch.stack(xhat_passes, dim=0)
        mu_mean = mu_stack.mean(dim=0)
        xhat_mean = xhat_stack.mean(dim=0)

        recon_score = torch.mean((x - xhat_mean) ** 2, dim=(1, 2, 3))
        svdd_score = torch.sum((mu_mean - svdd_center.unsqueeze(0)) ** 2, dim=1)
        unc_score = (
            mu_stack.var(dim=0, unbiased=False).mean(dim=1) +
            xhat_stack.var(dim=0, unbiased=False).mean(dim=(1, 2, 3))
        )
        fused_score = (
            alpha_score * recon_score +
            beta_score * svdd_score +
            gamma_score * unc_score
        )
        anomaly_flag = fused_score > score_threshold

        for idx, image_path in enumerate(image_paths):
            rows.append({
                "image_path": image_path,
                "recon_score": float(recon_score[idx].item()),
                "svdd_score": float(svdd_score[idx].item()),
                "uncertainty_score": float(unc_score[idx].item()),
                "fused_score": float(fused_score[idx].item()),
                "threshold": score_threshold,
                "anomaly_flag": int(anomaly_flag[idx].item()),
            })

    model.set_mc_dropout(False)

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "recon_score",
                "svdd_score",
                "uncertainty_score",
                "fused_score",
                "threshold",
                "anomaly_flag",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {args.output_csv}")


def main():
    parser = argparse.ArgumentParser("Export per-image NVAE anomaly scores to CSV")
    parser.add_argument("--checkpoint", default="/public/ATIQA/NVAEmaster/eval-exp/checkpoint.pt")
    parser.add_argument("--data_root", default="/public/ATIQA/Datasets/iu_xray/")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--csv_path", default="")
    parser.add_argument("--output_csv", default="/public/ATIQA/NVAEmaster/eval-exp/train_scores.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mc_passes", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    export_scores(args)


if __name__ == "__main__":
    main()
