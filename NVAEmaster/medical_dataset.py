import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MedicalImageDataset(Dataset):
    def __init__(self, csv_path, image_root, image_size=256, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.image_size = image_size

        if "chestxrays" in self.df.columns:
            self.image_col = "chestxrays"
        elif "images" in self.df.columns:
            self.image_col = "images"
        else:
            raise KeyError("Expected image column 'chestxrays' or 'images' in dataset CSV.")

        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_rel_path = self.df.iloc[idx][self.image_col]
        img_path = os.path.join(self.image_root, image_rel_path)

        image = Image.open(img_path)

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transform(image)

        if image.dim() != 3 or image.shape[0] != 3:
            raise ValueError(
                f"[MedicalImageDataset] Bad image shape {image.shape} at {img_path}"
            )

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
        text = self.df.iloc[idx]["captions"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
