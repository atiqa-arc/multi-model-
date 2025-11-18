import torch
import torch
import torch.nn.functional as F 
# 🔹 Step 1: Load your saved memory bank
text_memory_bank = torch.load("text_memory_bank.pt", weights_only=True)

print(f"✅ Loaded text memory bank with {len(text_memory_bank)} samples")

# 🔹 Step 2: Separate mean (μ), variance (logσ), and attention (attn) vectors
mu_list = [item["mu"] for item in text_memory_bank]
logsigma_list = [item["logsigma"] for item in text_memory_bank]
attn_list = [item["attn"] for item in text_memory_bank]

print(f"✅ Extracted {len(mu_list)} mean, {len(logsigma_list)} variance, and {len(attn_list)} attention vectors")

# 🔹 Step 3: Save them separately (without stacking)
torch.save(mu_list, "text_mu_list.pt")
torch.save(logsigma_list, "text_logsigma_list.pt")
torch.save(attn_list, "text_attn_list.pt")
print("💾 Saved mean vectors → text_mu_list.pt")
print("💾 Saved variance vectors → text_logsigma_list.pt")
print("💾 Saved attention vectors → text_attn_list.pt")

# -----------------------------
# 1️⃣  Path to your file
# -----------------------------
file_path = "/public/ATIQA/my_code/text_logsigma_list.pt"
data = torch.load("/public/ATIQA/my_code/text_logsigma_list.pt", map_location="cuda:2")

# pad for consistent shapes check max and mini variance
max_len = max(t.shape[0] for t in data)
padded = [F.pad(t, (0,0,0,max_len - t.shape[0])) for t in data]
logsigma = torch.stack(padded)
variance = torch.exp(logsigma)

print("📊 Variance stats across dataset:")
print("Min:", variance.min().item())
print("Max:", variance.max().item())
print("Mean:", variance.mean().item())
# -----------------------------
# 2️⃣  Load your stored data
# -----------------------------
data = torch.load(file_path)
print(f"✅ Loaded file: {file_path}")
print(f"Type: {type(data)}, Length: {len(data)}")
# -----------------------------
# 2️⃣ Find the maximum sequence length
# -----------------------------
lengths = [t.shape[0] for t in data]
max_len = max(lengths)
hidden_dim = data[0].shape[1]
print(f"Longest sequence length: {max_len}")
# -----------------------------
# 3️⃣ Pad all tensors to max_len
# -----------------------------
padded_list = []
for t in data:
    pad_len = max_len - t.shape[0]
    if pad_len > 0:
        # pad format: (last_dim_left, last_dim_right, first_dim_left, first_dim_right)
        # here we pad only on sequence length dimension
        padded_t = F.pad(t, (0, 0, 0, pad_len), value=0.0)
    else:
        padded_t = t
    padded_list.append(padded_t)

logsigma = torch.stack(padded_list, dim=0)  # shape: [N, max_len, 512]
print(f"✅ logsigma shape after padding: {logsigma.shape}")
# -----------------------------
# 4️⃣ Convert log variance → variance
# -----------------------------
variance = torch.exp(logsigma)  # [N, max_len, 512]
print("✅ Converted log variance → variance")

# -----------------------------
# 1️⃣ Compute one uncertainty score per token
# -----------------------------
token_uncertainty = variance # [N, T]
print(f"Token uncertainty shape: {token_uncertainty.shape}")
# Now each value corresponds to how uncertain that token is.
# -----------------------------
# 2️⃣ Create a token-level mask for low-uncertainty tokens
# -----------------------------
threshold = 1.2
token_mask = (token_uncertainty < threshold)  # [N, T]
print(f"Token mask shape: {token_mask.shape}")
# -----------------------------
# 3️⃣ Select only low-uncertainty tokens per report
# -----------------------------
# We'll zero out (or remove) uncertain tokens for each report.
low_uncertainty_tokens = []
for i in range(variance.size(0)):  # for each report
    token_var = variance[i]              # [T, 512]
    mean_var = token_var.mean(dim=1)     # [T]
    mask = mean_var < threshold        # mask for tokens
    selected_tokens = token_var[mask]    # [T_selected, 512]
    low_uncertainty_tokens.append(selected_tokens)

print(f"✅ Example report 0: kept {low_uncertainty_tokens[0].shape[0]} / {variance.size(1)} tokens")

# -----------------------------
# 4️⃣ Save token-level results
# -----------------------------
torch.save(low_uncertainty_tokens, "/public/ATIQA/my_code/low_uncertainty_tokens.pt")
print("💾 Saved per-report low-uncertainty tokens → low_uncertainty_tokens.pt")


# 1️⃣ Load your file
path = "/public/ATIQA/my_code/low_uncertainty_tokens.pt"
data = torch.load(path, map_location="cuda:2")

print(f"✅ Total reports: {len(data)}")

# 2️⃣ Check data type
print(f"🔍 Type of first item: {type(data[0])}")

# 3️⃣ Check shape and dtype
print(f"🔍 Shape of first item: {data[0].shape}")
print(f"🔍 Data type: {data[0].dtype}")

# 3️⃣ Show first 3 token vectors
print("🔢 First 3 token vectors (first 10 dims each):")
for i in range(3):
    print(f"Token {i}:", data[0][i][:10])

import torch

# Load your file
path = "/public/ATIQA/my_code/low_uncertainty_tokens.pt"
data = torch.load(path, map_location="cuda:2")

# If it's a list of tensors
if isinstance(data, list) and isinstance(data[0], torch.Tensor):
    print("✅ File type: List of tensors")
    print("Number of tensors:", len(data))
    print("Shape of first tensor:", data[0].shape)
    print("Tensor dtype:", data[0].dtype)
    print("Device:", data[0].device)

# If it's a single tensor
elif isinstance(data, torch.Tensor):
    print("✅ File type: Single tensor")
    print("Shape:", data.shape)
    print("Tensor dtype:", data.dtype)
    print("Device:", data.device)

else:
    print("⚠️ Unknown data type:", type(data))

