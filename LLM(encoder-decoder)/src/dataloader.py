import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class ReportTextDataset(Dataset):
    def __init__(self, csv_path, text_column="captions"):
        self.df = pd.read_csv(csv_path)
        self.text_column = text_column

        if text_column not in self.df.columns:
            raise ValueError(f"Column '{text_column}' not found in {csv_path}")

        self.texts = self.df[text_column].fillna("").astype(str).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx]}


def get_report_dataloaders(data_root, batch_size=16, text_column="captions", num_workers=0):
    train_csv = os.path.join(data_root, "train_pairs.csv")
    val_csv = os.path.join(data_root, "val_pairs.csv")
    test_csv = os.path.join(data_root, "test_pairs.csv")

    train_dataset = ReportTextDataset(train_csv, text_column=text_column)
    val_dataset = ReportTextDataset(val_csv, text_column=text_column)
    test_dataset = ReportTextDataset(test_csv, text_column=text_column)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def collect_reports(data):
    reports = []

    if isinstance(data, DataLoader):
        for batch in data:
            batch_text = batch.get("text", [])
            if isinstance(batch_text, str):
                reports.append(batch_text)
            else:
                reports.extend([str(text) for text in batch_text])
        return reports

    if isinstance(data, Dataset):
        for idx in range(len(data)):
            sample = data[idx]
            if isinstance(sample, dict) and "text" in sample:
                reports.append(str(sample["text"]))
        return reports

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                reports.append(item)
            elif isinstance(item, dict) and "text" in item:
                reports.append(str(item["text"]))
        return reports

    raise ValueError("Unsupported data format for report collection")


def _clean_report_text(report):
    report = str(report).strip().lower()
    report = report.replace("\n", " ")
    report = re.sub(r"\s+", " ", report)
    return report


def _split_report_to_sentences(report):
    # Split on common sentence boundaries in medical reports.
    parts = re.split(r"[.!?]+|\s{2,}", report)
    return [part.strip() for part in parts if part and part.strip()]


def _tokenize_sentence(sentence):
    # Keep words, numbers, and lightweight punctuation as separate tokens.
    return re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*|[(),:;.%]", sentence)


def data2sentences(data, name="medical"):
    if name not in {"medical", "radiology", "iu_xray"}:
        raise NotImplementedError(f"Dataset '{name}' is not supported by this dataloader")

    sentence_data = []
    reports = collect_reports(data)

    for report in reports:
        cleaned_report = _clean_report_text(report)
        if not cleaned_report:
            continue

        for sentence in _split_report_to_sentences(cleaned_report):
            tokens = _tokenize_sentence(sentence)
            if len(tokens) >= 3:
                sentence_data.append(tokens)

    return sentence_data


def preprocess_text(name, dataset, text_field, max_len=50, plot=False):
    sentences_raw = data2sentences(dataset, name=name)

    if plot:
        sentence_length = [len(tokens) for tokens in sentences_raw]
        plt.hist(sentence_length, bins=50)
        plt.title("Sentence Length Distribution - Before Filter")
        plt.show()

    sentences_filtered = [tokens for tokens in sentences_raw if 3 <= len(tokens) <= max_len]

    if plot:
        sentence_length = [len(tokens) for tokens in sentences_filtered]
        plt.hist(sentence_length, bins=50)
        plt.title("Sentence Length Distribution - After Filter")
        plt.show()

    sentences = text_field.pad(sentences_filtered)
    sentences = text_field.numericalize(sentences)
    return sentences[:, 1:]


def pad_token_sen(token_sen, max_sen_len, pad_idx, device):
    pad_add = torch.full((1, max_sen_len - len(token_sen) + 2), pad_idx, device=device)
    token_sen = token_sen.transpose(0, 1)
    out = torch.cat((token_sen, pad_add), 1)
    out = out.transpose(1, 0)
    return out


def create_masks(batch, pad_token):
    pad_mask = (batch == pad_token).transpose(0, 1)

    seq_len = batch.shape[0]
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=batch.device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))

    return pad_mask, mask


def calc_kl(mu, logvar, reduce="mean"):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    if reduce == "sum":
        kl = torch.sum(kl)
    elif reduce == "mean":
        kl = torch.mean(kl)
    return kl


def cyc_beta_scheduler(epochs, warmup_epochs, beta_min, beta_max, period, ratio):
    beta_warmup = np.ones(warmup_epochs) * beta_min
    beta_cyc = np.ones(epochs - warmup_epochs) * beta_max
    n_cycle = int(np.floor((epochs - warmup_epochs) / period))
    step = (beta_max - beta_min) / (period * ratio)
    for cycle in range(n_cycle):
        curr_beta, i = beta_min, 0
        while curr_beta <= beta_max and (int(i + cycle * period) < epochs - warmup_epochs):
            beta_cyc[int(i + cycle * period)] = curr_beta
            curr_beta += step
            i += 1
    beta = np.concatenate((beta_warmup, beta_cyc), axis=0)
    return beta
