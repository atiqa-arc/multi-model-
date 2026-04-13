# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""Code for getting the data loaders."""
import os
import torch
import csv

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.medical_data import MedicalImageDataset , MedicalTextDataset
from src.medical_data import CombinedDataset
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_loaders(args):
    """Get data loaders for required dataset."""
    return get_loaders_sep(args.dataset, args, args.tokenizer)

def get_loaders_sep(dataset, args, tokenizer):
    """Get medical dataset loaders."""
    if dataset== 'medical':
        num_classes = 1
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])
        
        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])
        
        # Define CSV paths
        train_csv = os.path.join(args.data, "train_pairs.csv")
        val_csv = os.path.join(args.data, "val_pairs.csv")
        test_csv = os.path.join(args.data, "test_pairs.csv")
        
        # Create datasets for images
        train_data = MedicalImageDataset( csv_path=train_csv, image_root=args.data,image_size=256, transform=train_transform, )
        valid_data = MedicalImageDataset(csv_path=val_csv, image_root=args.data,image_size=256, transform=valid_transform, )
        test_data = MedicalImageDataset( csv_path=test_csv,image_root=args.data,image_size=256,transform=valid_transform,)
        # Create DataLoaders for images
        train_queue = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0, drop_last=True)
        valid_queue = DataLoader(valid_data, batch_size=args.batch_size,shuffle=True,pin_memory=True, num_workers=0, drop_last=False )
        test_queue = DataLoader(test_data, batch_size=args.batch_size,shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
        
        # Create datasets for text (captions)
        train_data_cap = MedicalTextDataset(
            csv_path=train_csv,
            tokenizer=tokenizer,  # Pass tokenizer here
            max_length=256
        )

        valid_data_cap = MedicalTextDataset(
            csv_path=val_csv,
            tokenizer=tokenizer,  # Pass tokenizer here
            max_length=256
        )

        test_data_cap = MedicalTextDataset(
            csv_path=test_csv,
            tokenizer=tokenizer,  # Pass tokenizer here
            max_length=256
        )
        
        # Create DataLoaders for text (captions)
        train_data_cap_loader = DataLoader(
            train_data_cap,  # Pass the train text dataset here
            batch_size=args.batch_size,
            shuffle=False,  # Typically no shuffle in the training text
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )
        
        valid_data_cap_loader = DataLoader(
            valid_data_cap,  # Pass the valid text dataset here
            batch_size=args.batch_size,
            shuffle=False,  # Typically no shuffle in the validation text
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
        
        test_data_cap_loader = DataLoader(
            test_data_cap,  # Pass the test text dataset here
            batch_size=args.batch_size,
            shuffle=False,  # No shuffle for the test set
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
        
        return train_queue, valid_queue, test_queue, train_data_cap_loader, valid_data_cap_loader, test_data_cap_loader, train_transform, valid_transform, test_transform, num_classes
    
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not supported")

def get_loaders(args):
    """Get data loaders for the required dataset."""
    return get_loaders_com(args.dataset, args, args.tokenizer)

def get_loaders_com(dataset, args, tokenizer):
    """Get medical dataset loaders."""
    if dataset != "medical":
        raise NotImplementedError(f"Dataset '{dataset}' not supported")

    num_classes = 1
    train_image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    eval_image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_csv = os.path.join(args.data, "train_pairs.csv")
    val_csv = os.path.join(args.data, "val_pairs.csv")
    test_csv = os.path.join(args.data, "test_pairs.csv")

    combined_train_data = CombinedDataset(
        image_csv_path=train_csv,
        text_csv_path=train_csv,
        image_root=args.data,
        tokenizer=tokenizer,
        image_size=256,
        max_length=256,
        image_transform=train_image_transform,
    )

    combined_valid_data = CombinedDataset(
        image_csv_path=val_csv,
        text_csv_path=val_csv,
        image_root=args.data,
        tokenizer=tokenizer,
        image_size=256,
        max_length=256,
        image_transform=eval_image_transform,
    )

    combined_test_data = CombinedDataset(
        image_csv_path=test_csv,
        text_csv_path=test_csv,
        image_root=args.data,
        tokenizer=tokenizer,
        image_size=256,
        max_length=256,
        image_transform=eval_image_transform,
    )

    train_loader = DataLoader(combined_train_data, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=0, drop_last=True)
    valid_loader = DataLoader(combined_valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(combined_test_data, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=0, drop_last=False)

    return train_loader, valid_loader, test_loader, num_classes
