import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MedicalImageDataset(Dataset):
    def __init__(self, csv_path, image_root, image_size=224, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.image_size = image_size

        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_col = "chestxrays" if "chestxrays" in self.df.columns else "images"
        relative_path = self.df.iloc[idx][image_col]
        img_path = os.path.join(self.image_root, relative_path)

        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transform(image).clone().contiguous()

        if image.dim() != 3 or image.shape[0] != 3:
            raise ValueError(f"[MedicalImageDataset] Bad image shape {image.shape} at {img_path}")

        dummy_label = torch.tensor(0, dtype=torch.long)
        return image, dummy_label


class MedicalTextDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=256):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]["captions"])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0).clone().contiguous()
        attention_mask = encoding["attention_mask"].squeeze(0).clone().contiguous()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class CombinedDataset(Dataset):
    def __init__(
        self,
        image_csv_path,
        text_csv_path,
        image_root,
        tokenizer,
        image_size=224,
        max_length=256,
        image_transform=None,
        svdd_score_map=None,
        svdd_threshold=0.0,

    ):
        self.image_df = pd.read_csv(image_csv_path)
        self.text_df = pd.read_csv(text_csv_path)

        assert len(self.image_df) == len(self.text_df), "The number of images and texts should match."

        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.svdd_score_map = svdd_score_map or {}
        self.svdd_threshold = float(svdd_threshold)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_root = image_root

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, idx):
        image_col = "chestxrays" if "chestxrays" in self.image_df.columns else "images"
        image_relative_path = self.image_df.iloc[idx][image_col]
        image_path = os.path.join(self.image_root, image_relative_path)
        image = Image.open(image_path)
        


        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.image_transform(image).clone().contiguous()

        caption = str(self.text_df.iloc[idx]["captions"])

        encoding = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt", )
        
        score_key = str(image_relative_path).strip()
        score = float(self.svdd_score_map.get(score_key, 0.0))
        input_ids = encoding["input_ids"].squeeze(0).clone().contiguous()
        attention_mask = encoding["attention_mask"].squeeze(0).clone().contiguous()

        return {
            "image": image,
            "image_path": image_relative_path,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "svdd_score": torch.tensor(score, dtype=torch.float32),
            "is_abnormal": torch.tensor(score > self.svdd_threshold, dtype=torch.long),
}

