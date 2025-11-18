import torch
import torch
import torch.nn.functional as F 
# 🔹 Step 1: Load your saved memory bank
image_memory_bank = torch.load("image_memory_bank.pt", weights_only=True)

print(f"✅ Loaded text memory bank with {len(image_memory_bank)} samples")

# 🔹 Step 2: Separate mean (μ), variance (logσ), and attention (attn) vectors
mu_list = [item["mu"] for item in image_memory_bank]
logsigma_list = [item["logsigma"] for item in image_memory_bank]
attn_list = [item["attn"] for item in image_memory_bank]

print(f"✅ Extracted {len(mu_list)} mean, {len(logsigma_list)} variance, and {len(attn_list)} attention vectors")
# 🔹 Step 3: Save them separately (without stacking)
torch.save(mu_list, "image_mu_list.pt")
torch.save(logsigma_list, "image_logsigma_list.pt")
torch.save(attn_list, "image_attn_list.pt")
print("💾 Saved mean vectors → image_mu_list.pt")
print("💾 Saved variance vectors → image_logsigma_list.pt")
print("💾 Saved attention vectors → image_attn_list.pt")

# -----------------------------
# 1️⃣  Path to your file
# -----------------------------
file_path = "/public/ATIQA/my_code/image_logsigma_list.pt"
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
# Stack them directly
logsigma_tensor = torch.stack(logsigma_list)            # [2069, 196, 512]
# -----------------------------
# 4️⃣ Convert log variance → variance
# -----------------------------
variance = torch.exp(logsigma_tensor)  # [N, max_len, 512]
print("✅ Converted log variance → variance")

# -----------------------------
# 1️⃣ Compute one uncertainty score per feature
# -----------------------------
feature_uncertainty = variance # [N, T]
print(f"feature uncertainty shape: {feature_uncertainty.shape}")

# Now each value corresponds to how uncertain that feature is.
# -----------------------------
# 2️⃣ Create a feature-level mask for low-uncertainty tokens
# -----------------------------
threshold = 1.0
feature_mask = (feature_uncertainty < threshold)  # [N, T]
print(f"feature mask shape: {feature_mask.shape}")

# -----------------------------
# 3️⃣ Select only low-uncertainty feature per image
# -----------------------------
# We'll zero out (or remove) uncertain tokens for each report.
low_uncertainty_features = []
for i in range(variance.size(0)):  # iterate over features
    mask = feature_mask[i]  # [T]
    selected_features = variance[i][mask]  # keep only low-uncertainty features
    low_uncertainty_features.append(selected_features)

print(f"✅ Example report 0: kept {low_uncertainty_features[0].shape[0]} / {variance.size(1)} tokens")

# -----------------------------
# 4️⃣ Save feature-level results
# -----------------------------
torch.save(low_uncertainty_features, "/public/ATIQA/my_code/low_uncertainty_features.pt")
print("💾 Saved per-feature low-uncertainty features → low_uncertainty_features.pt")
