
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision import models, transforms
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import ViTModel, ViTImageProcessor
logger = logging.getLogger(__name__)
import torch
from torch import nn
from PIL import Image
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class Tokenizer:
    def __init__(self, max_text_length=512, max_summary_length=128, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side="left")

        # GPT-2 has no pad token by default, so set it
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        self.max_text_length = max_text_length
        self.max_summary_length = max_summary_length
        self.vocab_size = len(self.tokenizer)

    def __call__(self, text, is_target=False, padding=True, truncation=True):
        max_len = self.max_summary_length if is_target else self.max_text_length

        return self.tokenizer(
            text,
            padding="max_length" if padding else False,
            truncation=truncation,
            max_length=max_len,
            return_tensors="pt",
        )

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, token_ids_batch, skip_special_tokens=True):
        return self.tokenizer.batch_decode(token_ids_batch, skip_special_tokens=skip_special_tokens)



class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim, max_seq_length):
        super().__init__()
        # Create fixed sinusoidal positional embeddings
        pe = torch.zeros(max_seq_length, emb_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term) # even-index columns
        pe[:, 1::2] = torch.cos(position * div_term) # odd-index columns
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]  # Adjust to match the input size
        
        return x
    
class AttentionHead(nn.Module):
    def __init__(self, emb_dim, head_size, causal=False):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.causal = causal

    def forward(self, x, kv=None, k_mask=None):
        Q = self.query(x)                          # (B, q_len, head_size)
        K = self.key(kv if kv is not None else x) # (B, k_len, head_size)
        V = self.value(kv if kv is not None else x)

        attention = Q @ K.transpose(-2, -1)       # (B, q_len, k_len)
        attention = attention / (self.head_size ** 0.5)

        # Key padding mask for both self-attn and cross-attn
        if k_mask is not None:
            k_mask = k_mask.unsqueeze(1)          # (B, 1, k_len)
            attention = attention.masked_fill(k_mask == 0, float("-inf"))

        # Causal mask only for self-attention (not cross-attention)
        if self.causal and kv is None:
            q_len = Q.size(1)
            k_len = K.size(1)
            c_mask = torch.tril(
                torch.ones(q_len, k_len, device=x.device, dtype=torch.bool)
            )
            attention = attention.masked_fill(~c_mask, float("-inf"))

        attention = torch.softmax(attention, dim=-1)
        output = attention @ V               
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, causal=False):
       super().__init__()
       assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
       self.head_size = emb_dim // n_heads
       self.W_o = nn.Linear(emb_dim, emb_dim, bias=False)
       self.causal = causal
       self.heads = nn.ModuleList([AttentionHead(emb_dim, self.head_size, causal=self.causal) for _ in range(n_heads)])

    def forward(self, x, kv=None, k_mask=None):
      # Combine attention heads
      out = torch.cat([head(x, kv, k_mask=k_mask) for head in self.heads], dim=-1)
      out = self.W_o(out)

      return out
  
class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, n_heads, r_mlp=4, ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads

        # Sub-Layer 1 Normalization
        self.ln1 = nn.LayerNorm(emb_dim)

        # Multi-Head Attention
        self.mha = MultiHeadAttention(emb_dim, n_heads)
        # Sub-Layer 2 Normalization
        self.ln2 = nn.LayerNorm(emb_dim)

        # Multilayer Perceptron
        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * r_mlp),
            nn.GELU(),
            nn.Linear(self.emb_dim * r_mlp, self.emb_dim)
        )
 

    def forward(self, x, src_mask=None):
        # Residual Connection After Sub-Layer 1 (MHA)
        x = x + self.mha(self.ln1(x), k_mask=src_mask)
        # Residual Connection After Sub-Layer 2 (MLP)
        x = x + self.mlp(self.ln2(x))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, emb_dim, n_heads, r_mlp=4, ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads

        self.ln1 = nn.LayerNorm(emb_dim)
        self.masked_mha = MultiHeadAttention(emb_dim, n_heads, causal=True)
        

        self.ln2 = nn.LayerNorm(emb_dim)
        self.cross_attention = MultiHeadAttention(emb_dim, n_heads)
       

        self.ln3 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * r_mlp),
            nn.GELU(),
            nn.Linear(self.emb_dim * r_mlp, self.emb_dim)
        )

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = x + self.masked_mha(self.ln1(x), k_mask=tgt_mask)
        x = x + self.cross_attention(self.ln2(x), kv=encoder_output, k_mask=src_mask)
        x = x + self.mlp(self.ln3(x))
        return x

class ViTImageEncoder(nn.Module):
    def __init__(self, device,model_name="google/vit-base-patch16-224",emb_dim=768,n_heads=8,n_layers=2,
        dropout=0.1,):
        super().__init__()

        self.device = device
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.vit_model = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
        self.vit_model.to(self.device)

        vit_dim = self.vit_model.config.hidden_size
        self.proj = nn.Linear(vit_dim, emb_dim) if vit_dim != emb_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        # Reuse your existing TransformerEncoder block
        self.layers = nn.ModuleList(
            [TransformerEncoder(emb_dim, n_heads,) for _ in range(n_layers)]
        )
        self.to(self.device)

    def forward(self, images, src_mask=None):
        """
        images: list of PIL images or tensor batch
        """

        # Ensure images are in the correct format
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()
            images = images.permute(0, 2, 3, 1)  # [batch_size, channels, height, width]
            images = (images * 255.0).to(torch.uint8)
            images = [Image.fromarray(image.numpy()) for image in images]

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        x= self.vit_model(**inputs).last_hidden_state 
        
        for layer in self.layers:
            x = layer(x, src_mask=src_mask) 

        return x

class TextDecoder(nn.Module):
    def __init__(self, device, model_name="gpt2", tokenizer=None,):
        super().__init__()

        self.device = device
        config = GPT2Config.from_pretrained(model_name)
        config.add_cross_attention = True
        config.is_decoder = True
        self.decoder = GPT2LMHeadModel.from_pretrained( model_name, config=config )
        self.tokenizer = tokenizer if tokenizer else GPT2Tokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        self.decoder.config.pad_token_id = self.tokenizer.pad_token_id
    def forward(self, tgt, encoder_outputs, src_mask=None, tgt_mask=None):
        if isinstance(tgt, dict):
            input_ids = tgt["input_ids"]
            if tgt_mask is None:
                tgt_mask = tgt.get("attention_mask")
        else:
            input_ids = tgt

        if tgt_mask is None:
            tgt_mask = (input_ids != self.tokenizer.pad_token_id).long()

        labels = input_ids.clone()
        labels[tgt_mask == 0] = -100  # Ignore padding tokens in loss.

        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=tgt_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=src_mask,
            labels=labels,
        )
        return outputs

class EncoderDecoder(nn.Module):
    def __init__(self, emb_dim=768, max_text_length=600, n_heads=8, device=None, tokenizer=None):
        super().__init__()
        # Correct the initialization of ViTImageEncoder
        self.encoder = ViTImageEncoder(device=device, model_name="google/vit-base-patch16-224", emb_dim=emb_dim, n_heads=n_heads, n_layers=2)
        self.decoder = TextDecoder(device=device, model_name="gpt2", tokenizer=tokenizer)
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.max_text_length = max_text_length

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_outputs = self.encoder(src, src_mask=src_mask)
        decoder_outputs = self.decoder(tgt, encoder_outputs, src_mask=src_mask, tgt_mask=tgt_mask)
        return decoder_outputs
    def generate(self, src, src_mask=None, max_length=128, num_beams=3):
        encoder_outputs = self.encoder(src, src_mask=src_mask)
        batch_size = encoder_outputs.size(0)
        eos_token_id = self.decoder.tokenizer.eos_token_id
        pad_token_id = self.decoder.tokenizer.pad_token_id

        # GPT-2 has no BOS token by default; EOS is commonly used to start decoding.
        start_tokens = torch.full(
            (batch_size, 1),
            fill_value=eos_token_id,
            device = encoder_outputs.device,
            dtype=torch.long,
        )

        # Use HF generate() to enable KV-cache decoding and better decoding quality.
        return self.decoder.decoder.generate(
            input_ids=start_tokens,
            attention_mask=torch.ones_like(start_tokens),
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=src_mask,
            max_new_tokens=max_length,
            do_sample=False,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
