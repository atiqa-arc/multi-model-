# config_args.py

# config.py

import torch

class Config:
    data_path = "/public/atiqa/coco_precomp"
    text_model = "/public/model/Qwen2.5-1.5B-Instruct"
    visual_model = "/public/atiqa/vit trandformer"  # ✅ Add this line
    embed_dim = 512
    num_clusters = 3
    batch_size = 4
    lr = 2e-5
    epochs = 5
    kl_weight = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

