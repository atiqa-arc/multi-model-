# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""Code for getting the data loaders."""
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from medical_dataset import MedicalImageDataset
from medical_dataset import MedicalTextDataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
def get_loaders(args):
    """Get data loaders for required dataset."""
    return get_loaders_eval(args.dataset, args, tokenizer)

import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from medical_dataset import MedicalImageDataset, MedicalTextDataset

def get_loaders_eval(dataset, args, tokenizer):
    """Get medical dataset loaders."""
    if dataset == 'medical':
        num_classes = 1
        num_workers = getattr(args, "num_workers", 0)
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize(257),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        valid_transform = transforms.Compose([
            transforms.Resize(257),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(257),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
        
        # Define CSV paths
        train_csv = os.path.join(args.data, "train_pairs.csv")
        val_csv = os.path.join(args.data, "val_pairs.csv")
        test_csv = os.path.join(args.data, "test_pairs.csv")
        
        # Create datasets for images
        train_data = MedicalImageDataset(
            csv_path=train_csv,
            image_root=args.data,
            image_size=256,
            transform=train_transform,
        )
        
        valid_data = MedicalImageDataset(
            csv_path=val_csv,
            image_root=args.data,
            image_size=256,
            transform=valid_transform,
        )
        
        test_data = MedicalImageDataset(
            csv_path=test_csv,
            image_root=args.data,
            image_size=256,
            transform=valid_transform,
        )

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
        
        # Create DataLoaders for images
        train_sampler, valid_sampler , test_sampler = None, None , None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)
            test_sampler  = torch.utils.data.distributed.DistributedSampler(test_data)
    
        train_queue = DataLoader(
            train_data, batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler, pin_memory=True, num_workers=num_workers, drop_last=True
        )
    
        valid_queue = DataLoader(
            valid_data, batch_size=args.batch_size,
            shuffle=(valid_sampler is None),
            sampler=valid_sampler, pin_memory=True, num_workers=num_workers, drop_last=False
        )
        
        test_queue = DataLoader(
            test_data, batch_size=args.batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=num_workers, pin_memory=True, drop_last=False
        )
        
        # Create DataLoaders for text (captions)
        train_data_cap_loader = DataLoader(
            train_data_cap,  # Pass the train text dataset here
            batch_size=args.batch_size,
            shuffle=False,  # Typically no shuffle in the training text
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        valid_data_cap_loader = DataLoader(
            valid_data_cap,  # Pass the valid text dataset here
            batch_size=args.batch_size,
            shuffle=False,  # Typically no shuffle in the validation text
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        test_data_cap_loader = DataLoader(
            test_data_cap,  # Pass the test text dataset here
            batch_size=args.batch_size,
            shuffle=False,  # No shuffle for the test set
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return train_queue, valid_queue, num_classes
    
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not supported")
