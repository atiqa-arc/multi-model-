import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pyarrow as pa
from datasets import Dataset
from timm.layers import DropPath
from torchvision import transforms
from PIL import Image
from glob import glob


# ======================================================
# 🟢 1️⃣ DATASET LOADING
# ======================================================

dataset_path = "/public/ATIQA/my_code/train_pairs.csv"
val_dataset_path = "/public/ATIQA/my_code/val_pairs.csv"
test_dataset_path = "/public/ATIQA/my_code/test_pairs.csv"

train_df = pd.read_csv(dataset_path)
val_df = pd.read_csv(val_dataset_path)
test_df = pd.read_csv(test_dataset_path)

print("\n✅ Dataset loaded successfully!")
print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))
print("\nColumns:", train_df.columns.tolist())
print(train_df.head())
print(train_df.tail()) 

# Fix image paths by inserting 'images/' in path
train_df["image"] = train_df["image"].str.replace("/iu_xray/", "/iu_xray/images/")
val_df["image"] = val_df["image"].str.replace("/iu_xray/", "/iu_xray/images/")
test_df["image"] = test_df["image"].str.replace("/iu_xray/", "/iu_xray/images/")
print(train_df["image"].head())



# ======================================================
# 🧠 2️⃣ IMAGE PREPROCESSOR
# ======================================================

class ImagePreprocessor(nn.Module):
    """
    Converts X-ray images to tensors and patches.
    Equivalent to TextInputPreprocessor but for visual inputs.
    """
    def __init__(self, image_size=224, patch_size=16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def forward(self, image_paths):
        imgs = [self.transform(Image.open(p).convert("RGB")) for p in image_paths]
        x = torch.stack(imgs)  # (B, 3, H, W)
        return x


# ======================================================
# ⚙️ 3️⃣ IMAGE ENCODER MODULES
# ======================================================

class PatchEmbed(nn.Module):
    """Split image into patches and embed"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DisAttention(nn.Module):
    """Distributional Self-Attention for Vision Patches"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mu_proj = nn.Linear(int(dim / 2), dim)
        self.mu_proj_drop = nn.Dropout(proj_drop)
        self.logsig_proj = nn.Linear(int(dim / 2), dim)
        self.logsig_proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).reshape(B, N, 2, int(C / 2))
        mu = self.mu_proj(x[:, :, 0, :])
        logsigma = self.logsig_proj(x[:, :, 1, :])
        mu = self.mu_proj_drop(mu)
        logsigma = self.logsig_proj_drop(logsigma)
        return mu, logsigma, attn


class DisTrans(nn.Module):
    """Distributional Transformer Block for Images"""
    def __init__(self, dim=512, num_heads=8, mlp_ratio=4.0, qkv_bias=False,
                 qk_scale=None, drop=0.1, attn_drop=0.1, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = act_layer()
        self.norm1 = norm_layer(dim)
        self.attn = DisAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mu_mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.logsig_mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_ = self.norm1(self.act(self.fc(x)))
        mu, logsigma, attn = self.attn(x_)
        mu = x + self.drop_path(mu)
        mu = mu + self.drop_path(self.mu_mlp(self.norm2(mu)))
        logsigma = logsigma + self.drop_path(self.logsig_mlp(self.norm3(logsigma)))
        return mu, logsigma, attn


# ======================================================
# 🚀 4️⃣ MAIN EXECUTION
# ========
if __name__ == "__main__":
    preproc = ImagePreprocessor(image_size=224)
    patch_embed = PatchEmbed(img_size=224, patch_size=16, embed_dim=512)
    encoder = DisTrans(dim=512, num_heads=8)

    memory_bank = []  # Store feature representations

    # ⚙️ Loop over all images in the training set
    for idx, row in train_df.iterrows():
        image_path = row["image"]
        try:
            x = preproc([image_path])          # Preprocess single image
            x = patch_embed(x)                 # Convert to patch embeddings
            mu, logsigma, attn = encoder(x)    # Encode with transformer

            # Save this image’s representation
            memory_bank.append({
                "id": idx,
                "mu": mu.squeeze(0).detach().cpu(),         # [N, 512]
                "logsigma": logsigma.squeeze(0).detach().cpu(),  # [N, 512]
                "attn": attn.squeeze(0).detach().cpu()      # [8, N, N]
            })

            if (idx + 1) % 100 == 0:
                print(f"✅ Processed {idx + 1}/{len(train_df)} images")

        except Exception as e:
            print(f"⚠️ Skipped {image_path} due to error: {e}")

    # 💾 Save memory bank
    torch.save(memory_bank, "image_memory_bank.pt")
    print("\n💾 Saved full image memory bank as image_memory_bank.pt")

