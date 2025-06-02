import torch  
from torch.utils.data import Dataset
import numpy as np
import json
from transformers import AutoTokenizer

# ✅ Helper function to flatten any nested caption list into a single string
def flatten_caption(caption):
    result = []

    def recurse(item):
        if isinstance(item, list):
            for sub in item:
                recurse(sub)
        else:
            result.append(str(item))  # Ensure everything is string

    recurse(caption)
    return " ".join(result)


class GMMImageTextDataset(Dataset):
    def __init__(self, data_path, split='train', tokenizer_path="/public/model/Qwen2.5-1.5B-Instruct"):
        self.data_path = data_path
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load image region features: shape [N, 36, 2048]
        self.img_feats = np.load(f"{self.data_path}/{self.split}_ims_bbx.npy")

        # Load image size info (used to normalize dummy bounding boxes)
        self.sizes = np.load(f"{self.data_path}/{self.split}_ims_size.npy", allow_pickle=True).tolist()

        # Load captions (may be nested)
        with open(f"{self.data_path}/{self.split}_caps.json", "r") as f:
            self.captions = json.load(f)

        self.num_samples = len(self.captions)
        self.img_per_caption = 5 if self.img_feats.shape[0] != self.num_samples else 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_idx = idx // self.img_per_caption

        # Load image features
        image = torch.tensor(self.img_feats[img_idx], dtype=torch.float32)

        # Generate dummy bounding boxes
        imsize = self.sizes[img_idx]
        bboxes = self.fake_bboxes(image.shape[0], imsize)

        # Load and flatten caption
        raw_caption = self.captions[idx]
        caption = flatten_caption(raw_caption)

        # Tokenize caption
        tokenized = self.tokenizer(
            caption,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=64 
        )
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)

        # ✅ Return everything needed for training
        return image, input_ids, attention_mask, bboxes, idx, img_idx

    def fake_bboxes(self, k, imsize):
        width, height = imsize['image_w'], imsize['image_h']
        boxes = torch.rand((k, 4))
        boxes[:, 0] *= width
        boxes[:, 1] *= height
        boxes[:, 2] *= width
        boxes[:, 3] *= height
        return boxes / torch.tensor([width, height, width, height], dtype=torch.float32)


# ✅ Run a test case
if __name__ == "__main__":
    dataset = GMMImageTextDataset("/public/atiqa/coco_precomp", split="train")
    img, input_ids, attention_mask, bboxes, idx, img_idx = dataset[0]
    print("Image Feature Shape:", img.shape)         # [36, 2048]
    print("Caption Token IDs:", input_ids.shape)     # [64]
    print("Attention Mask:", attention_mask.shape)   # [64]
    print("BBox Shape:", bboxes.shape)               # [36, 4]
    print("Caption Index:", idx, "| Image Index:", img_idx)
