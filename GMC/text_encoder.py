import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pyarrow as pa
from datasets import Dataset, DatasetDict
from functools import partial
from timm.layers import DropPath
from transformers import AutoTokenizer


# ======================================================
# 🟢 1️⃣ DATASET LOADING AND TOKENIZATION
# ======================================================

dataset_path = "/public/ATIQA/my_code/train_pairs.csv"
val_dataset_path = "/public/ATIQA/my_code/val_pairs.csv"
test_dataset_path = "/public/ATIQA/my_code/test_pairs.csv"
# Load datasets
train_df = pd.read_csv(dataset_path)
val_df = pd.read_csv(val_dataset_path)
test_df = pd.read_csv(test_dataset_path)

print("\n✅ Dataset loaded successfully!")
print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))
print("\nColumns:", train_df.columns.tolist())
print("\n🔹 First few rows of training data:")
print(train_df.head())

# Sampling
train_df = train_df.sample(n=min(2069, len(train_df)))
val_df = val_df.sample(n=min(296, len(val_df)))
test_df = test_df.sample(n=min(590, len(test_df)))

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\nSampled dataset sizes:")
print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

# ======================================================
# 🧠 2️⃣ TOKENIZER + HUGGINGFACE DATASETS
# ======================================================

train_dataset = Dataset(pa.Table.from_pandas(train_df[["captions"]]))
valid_dataset = Dataset(pa.Table.from_pandas(val_df[["captions"]]))
test_dataset = Dataset(pa.Table.from_pandas(test_df[["captions"]]))
#hg_datasets = DatasetDict({"train": train_dataset, "valid": valid_dataset, "test": test_dataset})

# ======================================================
# ⚙️ 3️⃣ ENCODER DEFINITION
# ======================================================


class TextInputPreprocessor(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased", hidden_dim=512, max_len=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_list, device=None):
        # 1️⃣ Tokenize and get attention mask
        tokens = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"]
        attn_mask = tokens["attention_mask"]  # 1=real token, 0=padding

        if device is not None:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

        # 2️⃣ Embedding + positional encoding
        x = self.embedding(input_ids)
        pos = self.pos_embed[:, :x.size(1), :].to(x.device)
        x = self.norm(x + pos)

        # 3️⃣ Reshape attention mask for attention layer  (B,1,1,N)
        attn_mask_4d = attn_mask[:, None, None, :]

        # 4️⃣ Return both
        return x, attn_mask_4d

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

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).reshape(B, N, 2, int(C / 2))

        mu = x[:, :, 0, :]
        logsigma = x[:, :, 1, :]
        mu = self.mu_proj(mu)
        mu = self.mu_proj_drop(mu)
        logsigma = self.logsig_proj(logsigma)
        logsigma = self.logsig_proj_drop(logsigma)
        return mu, logsigma, attn


class DisTrans(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 qk_scale=None, drop=0.1, attn_drop=0.1, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = act_layer()
        self.norm1 = norm_layer(dim)
        self.attn = DisAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mu_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                          act_layer=act_layer, drop=drop)
        self.logsig_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                              act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x_ = self.norm1(self.act(self.fc(x)))
        mu, logsigma, attn = self.attn(x_, mask=mask)
        mu = x + self.drop_path(mu)
        mu = mu + self.drop_path(self.mu_mlp(self.norm2(mu)))
        logsigma = logsigma + self.drop_path(self.logsig_mlp(self.norm3(logsigma)))
        return mu, logsigma, attn


# ======================================================
# 🚀 4️⃣ MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    preproc = TextInputPreprocessor(hidden_dim=512)
    encoder = DisTrans(dim=512, num_heads=8)

    # You can comment out or keep for testing small samples
    # sample_texts = train_df["captions"].sample(n=min(5, len(train_df))).tolist()
    # x, attn_mask = preproc(sample_texts)
    # mu, logsigma, attn = encoder(x, mask=attn_mask)
    # print("\n✅ Encoder run successful")
    # print("mu:", mu.shape)
    # print("logsigma:", logsigma.shape)
    # print("attn:", attn.shape)

    # -----------------------------------------
    # 💾 Build Memory Bank
    # -----------------------------------------
    memory_bank = []  # list to store all representations

    # ⚠️ You can loop over all train samples, but start small (e.g. 50)
    for idx, row in train_df.iterrows():
        text = [row["captions"]]
        x, attn_mask = preproc(text)
        mu, logsigma, attn = encoder(x, mask=attn_mask)

        # Store sample
        memory_bank.append({
            "id": idx,
            "mu": mu.squeeze(0).detach().cpu(),         # shape [N, 512]
            "logsigma": logsigma.squeeze(0).detach().cpu(),  # shape [N, 512]
            "attn": attn.squeeze(0).detach().cpu(),     # shape [8, N, N]
        })

        if (idx + 1) % 100 == 0:
            print(f"✅ Processed {idx + 1}/{len(train_df)} samples")

    # -----------------------------------------
    # Save Memory Bank
    # -----------------------------------------
    torch.save(memory_bank, "text_memory_bank.pt")
    print("💾 Memory bank saved as text_memory_bank.pt")
    # Check overall structure
print(type(memory_bank))
print(len(memory_bank))

# See the keys of the first item
print(memory_bank[0].keys())

# See tensor shapes
print("mu:", memory_bank[0]["mu"].shape)
print("logsigma:", memory_bank[0]["logsigma"].shape)
print("attn:", memory_bank[0]["attn"].shape)
