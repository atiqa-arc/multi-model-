import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from GMC.dataset.coco_loader import CocoDataset, default_transform
from GMC.model.vit_encoder import ViTEncoder
from GMC.model.qwen_encoder import QwenEncoder
from GMC.model.gmm_module import GMMEmbedding
from GMC.utils.evaluation import evaluate_retrieval
import numpy as np
from config import Config
from transformers import AutoTokenizer
import json


def test_model(config, checkpoint_path, save_preds_path="predictions.json"):
    device = config.device
    tokenizer = AutoTokenizer.from_pretrained(config.text_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = CocoDataset(
        img_folder=config.image_path,
        ann_file=config.caption_path,
        tokenizer=tokenizer,
        transform=default_transform
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # === Load Model ===
    vit = ViTEncoder(proj_dim=config.embed_dim).to(device)
    qwen = QwenEncoder(model_name=config.text_model, proj_dim=config.embed_dim).to(device)
    gmm_image = GMMEmbedding(input_dim=config.embed_dim, num_clusters=config.num_clusters).to(device)
    gmm_text = GMMEmbedding(input_dim=config.embed_dim, num_clusters=config.num_clusters).to(device)

    checkpoint = torch.load("/public/atiqa/Hygraph/checkpoint_epoch1.pt", map_location=device)
    vit.load_state_dict(checkpoint['vit'])
    qwen.load_state_dict(checkpoint['qwen'])
    gmm_image.load_state_dict(checkpoint['gmm_image'])
    gmm_text.load_state_dict(checkpoint['gmm_text'])

    vit.eval()
    qwen.eval()

    all_img, all_txt, img_caps = [], [], []
    with torch.no_grad():
        for images, input_ids, attention_mask, caps in loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            batch_size = images.size(0)
            edge_index = torch.combinations(torch.arange(batch_size), r=2).T
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_index = edge_index.to(device)

            img_feat = vit(images, edge_index)
            txt_feat = qwen(input_ids, attention_mask, edge_index)

            img_z, _ = gmm_image(img_feat)
            txt_z, _ = gmm_text(txt_feat)

            all_img.append(img_z.cpu())
            all_txt.append(txt_z.cpu())
            img_caps.extend(caps)

    img_embeds = torch.cat(all_img, dim=0)
    txt_embeds = torch.cat(all_txt, dim=0)

    # === Compute similarity ===
    sim_matrix = F.cosine_similarity(img_embeds.unsqueeze(1), txt_embeds.unsqueeze(0), dim=2)

    # === Compute recall metrics ===
    def compute_recall(sim, topk):
        correct = 0
        for i in range(sim.size(0)):
            top_indices = torch.topk(sim[i], topk).indices
            if i in top_indices:
                correct += 1
        return correct / sim.size(0) * 100

    recall1 = compute_recall(sim_matrix, 1)
    recall5 = compute_recall(sim_matrix, 5)

    print(f"\n[TEST] Recall@1: {recall1:.2f}% | Recall@5: {recall5:.2f}%")

    # === Save predictions ===
    predictions = {}
    for i in range(sim_matrix.size(0)):
        top_idx = torch.argmax(sim_matrix[i]).item()
        onehot = [0] * sim_matrix.size(1)
        onehot[top_idx] = 1
        predictions[f"image_{i}"] = onehot

    with open(save_preds_path, 'w') as f:
        json.dump(predictions, f)
    print(f"✅ Predictions saved to {save_preds_path}")


if __name__ == "__main__":
    config = Config()
    test_model(config, checkpoint_path="checkpoint_epoch1.pt", )

