import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import torch.nn as nn
import gc
import pandas as pd

# Metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
device = torch.device('cuda:3')
print(device)
# ======================================================
# 3️⃣ Load Image Features (certainty-based)
#     Assumed file: image_certain_features.pt
#     Shape: [N, T_img, 512]
# ======================================================
img_feat_path = "/public/ATIQA/my_code/low_uncertainty_features.pt"
image_feat = torch.load(img_feat_path, map_location=device)  # [N, T_img, 512]
print(f"✅ Loaded image features from {img_feat_path}")
# 1. Find max sequence length
max_img_len = max(feat.shape[0] for feat in image_feat)
hidden_dim = image_feat[0].shape[1]

# 2. PAD each tensor to max length
padded_img_feats = [
    F.pad(feat, (0, 0, 0, max_img_len - feat.shape[0]), value=0.0)
    for feat in image_feat
]
# 3. STACK into a single tensor
image_feats = torch.stack(padded_img_feats).to(device)

# ======================================================
# 1️⃣ Load Low-Uncertainty Token Embeddings
# ======================================================
tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
VOCAB_SIZE = tokenizer.vocab_size
path = "/public/ATIQA/my_code/low_uncertainty_tokens.pt"
data = torch.load(path, map_location=device)
print(f"✅ Loaded {len(data)} reports from {path}")
print(f"Example report shape: {data[0].shape}")

# Pad variable-length sequences to same length
max_len = max(t.shape[0] for t in data)
hidden_dim = data[0].shape[1]

padded = [F.pad(t, (0, 0, 0, max_len - t.shape[0]), value=0.0) for t in data]
memory_bank = torch.stack(padded).to(device)  # [B, max_len, hidden_dim]
print(f"🧠 Memory bank shape: {memory_bank.shape}")
# NOW tensor
N = memory_bank.size(0)
# Build padding mask from your original padded memory (all-zero rows are padding)
pad_mask = (memory_bank.abs().sum(dim=-1) == 0)   # [B, L] -> True where padde
# ======================================================
# 1️⃣ Load Dataset ( GT Reports)
# ======================================================
csv_path = "/public/ATIQA/my_code/train_pairs.csv"
# Load ground-truth
df = pd.read_csv(csv_path)
gt_texts = df["captions"].tolist()
assert len(gt_texts) == len(data), "Mismatch in dataset sizes!"
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
VOCAB_SIZE = tokenizer.vocab_size

# Convert GT reports → token IDs
encoded = tokenizer(
    gt_texts,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
)
labels = encoded["input_ids"].to(device)          # [N, L_label]
label_mask = encoded["attention_mask"].to(device) # not used now
print("GT Label shape:", labels.shape)
# PAD value
pad_id = tokenizer.pad_token_id
# ======================================================
# 2️⃣ Decoder Input Layer
# ======================================================
class DecoderInputFromEmbeddings(nn.Module):
    def __init__(self, hidden_dim=512, max_len=512):
        super().__init__()
        self.pos_embed = nn.Parameter(0.01 * torch.randn(1, max_len, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        # ⭐ Projection layer (VERY IMPORTANT)
        self.var_to_embed = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, embeddings):
        B, L, D = embeddings.shape

        # ⭐ Project variance into embedding space
        embeddings = self.var_to_embed(embeddings)

        # Add positional encoding
        pos_emb = self.pos_embed[:, :L, :].to(embeddings.device)
        x = embeddings + pos_emb

        return self.norm(x)

# ======================================================
# 3️⃣ FeedForward Network
# ======================================================
class FeedForward(nn.Module):
    def __init__(self, dim=512, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))

# ======================================================
# 4️⃣ Masked Multi-Head Self-Attention Block
# ======================================================
class MaskedSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1, ff_hidden=2048):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.ff = FeedForward(hidden_dim, ff_hidden, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # ⭐ LM Head (VERY IMPORTANT)
        self.lm_head = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, x, key_padding_mask=None):

        # ⭐ No causal mask for memory input
        attn_out, attn_probs = self.attn(
            x, x, x,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
        )

        x = self.ln1(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))

        # ⭐ PROBLEM FIXED — now we output logits!
        logits = self.lm_head(x)

        # RETURN LOGITS, NOT X
        return logits, attn_probs

# ======================================================
# 4️⃣ Cross-Attention Block  (query = text decoder, key/value = image features)
# ======================================================
class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, img_feats, img_mask=None):
        """
        x:         [B, L_text, D]   (queries)
        img_feats: [B, L_img,  D]   (keys/values)
        img_mask:  [B, L_img]       (True where padding)
        """

        attn_out, attn_weights = self.cross_attn(
            query=x,
            key=img_feats,
            value=img_feats,
            key_padding_mask=img_mask
        )

        x = self.ln(x + self.dropout(attn_out))  # Add + Norm
        return x, attn_weights
    
# ======================================================
# 5️⃣ Add + Norm → FeedForward → Add + Norm → Linear → Softmax
# ======================================================
class DecoderOutputBlock(nn.Module):
    def __init__(self, hidden_dim=512, ff_hidden=2048, dropout=0.1, vocab_size=30522):
        super().__init__()

        # Feed Forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, hidden_dim),
            nn.Dropout(dropout)
        )

        # LayerNorm layers
        self.ln_ff = nn.LayerNorm(hidden_dim)

        # Final LM head
        self.linear = nn.Linear(hidden_dim, vocab_size)

        # Softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: [B, L, D] from cross-attention output
        """

        # 1️⃣ Feed-forward block
        ff_out = self.ff(x)

        # 2️⃣ Add + Norm
        x = self.ln_ff(x + ff_out)

        # 3️⃣ Linear projection to vocabulary
        logits = self.linear(x)

        # 4️⃣ Softmax → token probabilities
        probs = self.softmax(logits)

        return logits, probs


# ======================================================
# ⭐ FULL DECODER MODULE (Combines ALL Blocks)
# ======================================================
class FullDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim=512,
        max_len=512,
        num_heads=8,
        ff_hidden=2048,
        dropout=0.1,
        vocab_size=30522
    ):
        super().__init__()

        # 1. Decoder Input Layer
        self.input_layer = DecoderInputFromEmbeddings(
            hidden_dim=hidden_dim,
            max_len=max_len
        )

        # 2. TEXT Self-Attention Layer
        self.self_attn = MaskedSelfAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            ff_hidden=ff_hidden
        )

        # 3. TEXT → IMAGE Cross-Attention Layer
        self.cross_attn = CrossAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 4. Final Output Block (FFN → AddNorm → Linear → Softmax)
        self.output_block = DecoderOutputBlock(
            hidden_dim=hidden_dim,
            ff_hidden=ff_hidden,
            dropout=dropout,
            vocab_size=vocab_size
        )

    def forward(self, txt_embeddings, img_features, txt_mask=None, img_mask=None):
        """
        txt_embeddings : [B, L_text, D] (variance-based text embeddings)
        img_features   : [B, L_img,  D] (image certainty features)
        txt_mask       : [B, L_text] (True for PAD tokens)
        img_mask       : [B, L_img]
        """

        # ----------------------------------------
        # 1️⃣ Decoder Input Layer
        # ----------------------------------------
        x = self.input_layer(txt_embeddings)

        # ----------------------------------------
        # 2️⃣ Self-Attention
        # ----------------------------------------
        logits_sa, _ = self.self_attn(x, key_padding_mask=txt_mask)
        # logits_sa is [B, L_text, vocab] → not used here
        # x is updated inside self_attn (x after LayerNorm)

        # ----------------------------------------
        # 3️⃣ Cross-Attention (TEXT query, IMAGE key/value)
        # ----------------------------------------
        x, cross_weights = self.cross_attn(x, img_features, img_mask)

        # ----------------------------------------
        # 4️⃣ FFN → AddNorm → Linear → Softmax
        # ----------------------------------------
        logits, probs = self.output_block(x)

        return logits, probs, cross_weights

# ======================================================
# 3️⃣ Instantiate Decoder, Optimizer, Loss
# ======================================================

decoder = FullDecoder(
    hidden_dim=512,
    num_heads=8,
    ff_hidden=2048,
    dropout=0.1,
    vocab_size=VOCAB_SIZE
).to(device)
optimizer = torch.optim.AdamW(decoder.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
# temperature for logits
#TEMPERATURE = 4
# hyperparameters
EPOCHS = 10
BATCH_SIZE = 4
N = memory_bank.size(0)
steps_per_epoch = N // BATCH_SIZE
print("Training settings:")
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("LR:", 5e-5)
# ======================================================
# 4️⃣ TRAIN LOOP
# ======================================================
decoder.train()
# 🌟 Initialize best loss tracker
best_loss = float("inf")
best_model_path = "/public/ATIQA/my_code/best_decoder_model.pt"

for epoch in range(EPOCHS):

    total_loss = 0.0

    for i in range(0, N, BATCH_SIZE):

        # slice batch
        txt_batch = memory_bank[i:i+BATCH_SIZE].to(device)
        img_batch = image_feats[i:i+BATCH_SIZE].to(device)
        lbl_batch = labels[i:i+BATCH_SIZE].to(device)

        txt_mask_batch = pad_mask[i:i+BATCH_SIZE]
        #img_mask_batch = img_pad_mask[i:i+BATCH_SIZE]

        # forward pass
        logits, probs, _ = decoder(
            txt_embeddings=txt_batch,
            img_features=img_batch,
            txt_mask=txt_mask_batch,
            #img_mask=img_mask_batch
        )

        # apply temperature
        #logits = logits / TEMPERATURE

        # reshape for CE loss
        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            lbl_batch.reshape(-1)
        )

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / steps_per_epoch
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")
    # 🔥🔥🔥 SAVE BEST MODEL HERE 🔥🔥🔥
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(decoder.state_dict(), best_model_path)
        print(f"🌟 Best model saved at epoch {epoch+1} with loss {best_loss:.4f}")

    # clean CUDA memory
    torch.cuda.empty_cache()
    gc.collect()

# ======================================================
# 5️⃣ SAVE MODEL
# ======================================================

model_path = "/public/ATIQA/my_code/full_decoder_model.pt"
torch.save(decoder.state_dict(), model_path)
print("Final model saved at:", model_path)
print("Best model saved at:", best_model_path)




