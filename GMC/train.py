import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer
from GMC.dataset.coco_loader import GMMImageTextDataset
from GMC.model.vit_encoder import ViTEncoder
from GMC.model.qwen_encoder import QwenEncoder
from GMC.model.gmm_module import SoftClusterGaussianHead
from GMC.utils.evaluation import evaluate_retrieval
import sys
import json

sys.path.append("/public/atiqa/Hygraph")
from config import Config

# === Triplet Loss Implementation ===
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.05):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = torch.nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.ranking_loss(anchor, positive, negative)

def train(config):
    tokenizer = AutoTokenizer.from_pretrained("/public/model/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = GMMImageTextDataset(
        data_path="/public/atiqa/coco_precomp",
        split="train",
        tokenizer_path="/public/model/Qwen2.5-1.5B-Instruct"
    ) 
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    vit = ViTEncoder(model_path=config.visual_model, proj_dim=config.embed_dim).to(config.device)
    qwen = QwenEncoder(model_name=config.text_model, proj_dim=config.embed_dim).to(config.device)

    gmm_image = SoftClusterGaussianHead(input_dim=config.embed_dim, num_clusters=config.num_clusters).to(config.device)
    gmm_text = SoftClusterGaussianHead(input_dim=config.embed_dim, num_clusters=config.num_clusters).to(config.device)

    triplet_loss_fn = TripletLoss(margin=0.3)
    optimizer = torch.optim.AdamW(
        list(qwen.parameters()) +
        list(vit.parameters()) +
        list(gmm_image.parameters()) +
        list(gmm_text.parameters()),
        lr=config.lr
    )

    qwen.train()
    vit.train()
    best_loss = float('inf')

    for epoch in range(config.epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")

        for images, input_ids, attention_mask, bboxes, idx, img_idx in pbar:

            images = images.to(config.device)  # [B, 36, 2048]
            input_ids = input_ids.to(config.device)

            B, R, D = images.shape  # Region-level features
            edge_index = torch.combinations(torch.arange(R), r=2).T
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(images.device)

            img_feat = vit(images, edge_index=edge_index)  # [B, R, embed_dim]
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            txt_feat = qwen(input_ids, attention_mask, edge_index=edge_index)  # [B, embed_dim]

            img_z, img_kl = gmm_image(img_feat)  # [B, embed_dim], scalar
            txt_z, txt_kl = gmm_text(txt_feat)  # [B, embed_dim], scalar

            img_z = F.normalize(img_z, dim=1)
            txt_z = F.normalize(txt_z, dim=1)

            anchor = img_z
            positive = txt_z
            negative = txt_z.roll(shifts=1, dims=0)

            loss = triplet_loss_fn(anchor, positive, negative) + img_kl + txt_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch + 1}] Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'qwen': qwen.state_dict(),
                'vit': vit.state_dict(),
                'gmm_image': gmm_image.state_dict(),
                'gmm_text': gmm_text.state_dict()
            }, "best_checkpoint.pt")
            print(f"\n✅ New best model saved with Average Loss: {avg_loss:.4f}")

def test_model(config, checkpoint_path):
    print("\n🔍 Running evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(config.text_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_dataset = GMMImageTextDataset(
        data_path=config.data_path,
        split="test",
        tokenizer_path=config.text_model
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    vit = ViTEncoder(model_path=config.visual_model, proj_dim=config.embed_dim).to(config.device)
    qwen = QwenEncoder(model_name=config.text_model, proj_dim=config.embed_dim).to(config.device)

    gmm_image = GMMEmbedding(input_dim=config.embed_dim, num_clusters=config.num_clusters).to(config.device)
    gmm_text = GMMEmbedding(input_dim=config.embed_dim, num_clusters=config.num_clusters).to(config.device)

    ckpt = torch.load(checkpoint_path)
    vit.load_state_dict(ckpt['vit'])
    qwen.load_state_dict(ckpt['qwen'])
    gmm_image.load_state_dict(ckpt['gmm_image'])
    gmm_text.load_state_dict(ckpt['gmm_text'])

    vit.eval()
    qwen.eval()

    all_img_feats, all_txt_feats = [], []

    with torch.no_grad():
        for images, input_ids, _, _, _ in tqdm(test_loader, desc="Testing"):
            images = images.to(config.device)
            input_ids = input_ids.to(config.device)

            B, R, D = images.shape
            edge_index = torch.combinations(torch.arange(R), r=2).T
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(images.device)

            img_feat = vit(images, edge_index=edge_index)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            txt_feat = qwen(input_ids, attention_mask, edge_index=edge_index)

            img_z, _ = gmm_image(img_feat)
            txt_z, _ = gmm_text(txt_feat)

            img_z = F.normalize(img_z, dim=1)
            txt_z = F.normalize(txt_z, dim=1)

            all_img_feats.append(img_z)
            all_txt_feats.append(txt_z)

    all_img_feats = torch.cat(all_img_feats, dim=0)
    all_txt_feats = torch.cat(all_txt_feats, dim=0)

    evaluate_retrieval(all_img_feats, all_txt_feats)

if __name__ == "__main__":
    config = Config()
    train(config)
    test_model(config, checkpoint_path="best_checkpoint.pt")
