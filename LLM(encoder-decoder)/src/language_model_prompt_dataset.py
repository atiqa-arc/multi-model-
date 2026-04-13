# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.medical_data import MedicalTextDataset
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)


def collate(samples, pad_idx, eos_idx, prefix=False, sep_idx=None, prompt=None):
    if len(samples) == 0:
        return {}

    # Helper function to create a sentence with prompt and source
    def make_sentence(prompt, source):
        # Ensure EOS token is not part of the sentence if it's the last one
        if source[-1] == eos_idx:
            source = source[:-1]
        if prompt is None:
            return source  # Only source is used
        if prefix:
            sep = torch.LongTensor([sep_idx])
            return torch.cat([prompt, source, sep], dim=0)  # Concatenate prompt, source, and separator
        return torch.cat([source, prompt], dim=0)  # Concatenate source and prompt

    # Helper function to merge tokens
    def merge(tokens, pad_idx, eos_idx, move_eos_to_beginning=False):
        # Flatten the list of tokens into a single tensor
        merged_tokens = torch.cat(tokens, dim=0)

        # If we need to move the EOS token to the beginning
        if move_eos_to_beginning:
            eos_pos = (merged_tokens == eos_idx).nonzero(as_tuple=True)[0]
            if eos_pos.numel() > 0:  # If EOS token is present
                eos_token = merged_tokens[eos_pos]
                merged_tokens = torch.cat([eos_token, merged_tokens[:eos_pos[0]], merged_tokens[eos_pos+1:]])

        # Add padding token at the end (if needed)
        merged_tokens = torch.cat([merged_tokens, torch.tensor([pad_idx], dtype=torch.long)])

        return merged_tokens

    # Process each sample (only source, no separate target)
    source_tokens = []
    source_lengths = []
    for s in samples:
        source_tokens.append(make_sentence(prompt, s["source"]))
    
    # Calculate the lengths of the source tokens
    source_lengths = [t.ne(pad_idx).long().sum() for t in source_tokens]

    # Merge source tokens into a batch
    source = merge(source_tokens, pad_idx, eos_idx)

    # Get the lengths in tensor format
    source_lengths = torch.LongTensor(source_lengths)

    # Create the batch dictionary (no separate target)
    batch = {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "nsentences": len(samples),
        "ntokens": source_lengths.sum().item(),
        "net_input": {
            "src_tokens": source,  # Only source tokens as input
            "src_lengths": source_lengths,  # Source lengths
        },
        "target": source,  # Use the same source as target for text generation
    }
    return batch



class LanguageModelPromptDataset(MedicalTextDataset):
    def __init__(self, src, src_sizes, dictionary=None,  eos=None,shuffle=True, max_source_length=256, max_length=None, prompt_length=None):
        self.max_source_length = max_source_length
        self.max_length = max_length if max_length is not None else 256
        self.src = src  
        self.seq_sep = None
        self.dictionary = dictionary if dictionary else AutoTokenizer.from_pretrained("microsoft/biogpt")
        self.shuffle = shuffle
        self.eos = eos if eos is not None else self.dictionary.eos_token_id
        self.max_source_length = max_source_length
        self.max_target_length = self.max_length - self.max_source_length
        self.src_sizes = [self.max_source_length for _ in range(len(self.src))]  # We only need src_sizes
        self.sizes = np.array(self.src_sizes)  # Only source sizes are needed
        
    def get_batch_shapes(self):
        return None  # Not using buckets

    def __getitem__(self, index):
        # Access tokenized data from MedicalTextDataset
        src_item = self.src[index]["input_ids"]

        # Truncate if necessary
        if len(src_item) > self.max_source_length:
            src_item = src_item[:self.max_source_length]

        # Add EOS token to the source (target is the same as source)
        src_item[-1] = self.dict.eos()

        # Return the example as source and target are the same
        return {
            "id": index,
            "source": torch.LongTensor(src_item),
            "target": torch.LongTensor(src_item),  # Target is the same as source for text generation
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        # No target dataset, handle only source
        return collate(
            samples, 
            pad_idx=self.dictionary.pad_token_id,
            eos_idx=self.dictionary.eos_token_id,
           
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample (same as source tokens)."""
        return self.sizes[index]

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices."""
        sizes = self.sizes[indices]
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices for batching."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        return indices[np.argsort(self.sizes[indices], kind="mergesort")]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False)

    def prefetch(self, indices):
        self.src.prefetch(indices)  