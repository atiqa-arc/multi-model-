# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .language_model_prompt_dataset import LanguageModelPromptDataset
from omegaconf import II


logger = logging.getLogger(__name__)

class LanguageModelingPromptConfig:
    max_source_positions: Optional[int] = field(default=384, metadata={"help": "Max number of tokens in the source sequence, excluding EOS."})
   

class LanguageModelingPromptTask:
    def __init__(self, args, model_name="microsoft/biogpt"):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.prompt = args.manual_prompt
        self.prompt_length = len(self.prompt) if self.prompt else 0
        self.max_source_length = args.max_source_positions
        self.max_target_length = args.max_length - self.max_source_length - self.prompt_length
        logger.info("Model and tokenizer initialized.")

    def load_dataset(self, dataset):
        return LanguageModelPromptDataset(dataset)

    def inference_step(self, input_text, max_length=512, num_beams=5):
        if self.prompt:
            input_text = self.prompt + " " + input_text  # Add prompt to the input

        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                inputs['input_ids'], 
                max_length=max_length, 
                num_beams=num_beams,
                no_repeat_ngram_size=2,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    def build_generator(self, models, args):
        beam_size = getattr(args, "beam", 5)
        max_length = getattr(args, "max_length", 1024)
        
        def generate(input_ids):
            outputs = models.generate(
                input_ids,
                num_beams=beam_size,
                max_length=max_length,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            return outputs
        return generate

     

  
