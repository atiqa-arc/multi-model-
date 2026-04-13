import sys
from pathlib import Path
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from llama_ross_attn import LlamaForCausalLMWithCrossAttention



NVAE_ROOT = Path("/public/ATIQA/NVAEmaster")
BIOGPT_ROOT = Path("/public/ATIQA/BioGPTLLM")


def _load_module_from_path(module_name: str, file_path: Path):
    nvae_root = str(file_path.parent)
    if nvae_root not in sys.path:
        sys.path.insert(0, nvae_root)
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_nvae_utils_module(nvae_root: Path):
    utils_path = nvae_root / "utils.py"
    return _load_module_from_path("nvae_utils_local", utils_path)


def _load_autoencoder_class(nvae_root: Path):
    model_path = nvae_root / "model.py"
    module = _load_module_from_path("nvae_model_local", model_path)
    return module.AutoEncoder


AutoEncoder = _load_autoencoder_class(NVAE_ROOT)


class LatentMemoryAdapter(nn.Module):
    def __init__(self, latent_dim, hidden_size=768, dropout=0.1):
        super().__init__()
        self.mean_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        #self.var_proj = nn.Sequential(nn.Linear(latent_dim, hidden_size),nn.GELU(),nn.Dropout(dropout),nn.LayerNorm(hidden_size),)
        
        #self.scalar_proj = nn.Sequential( nn.Linear(1, hidden_size),nn.GELU(),nn.Dropout(dropout),nn.LayerNorm(hidden_size),)
        # 10 memory types: latent mean, latent variance, reconstruction variance,
        # reconstruction score, SVDD score, uncertainty, fused score, margin,
        # soft anomaly flag, and score threshold.
        self.type_embed = nn.Embedding(1, hidden_size)

    #def _scalar_state(self, value, type_idx): state = self.scalar_proj(value.unsqueeze(-1)).unsqueeze(1) return state + self.type_embed.weight[type_idx].view(1, 1, -1)

    def forward(
        self,
        z_mean_tokens,
        #z_var_tokens,
        #recon_var_tokens,
        #recon_score,
        #svdd_score,
        #unc_score,
        #fused_score,
        #margin,
        #soft_flag,
        #score_threshold,
    ):
        mean_states = self.mean_proj(z_mean_tokens) + self.type_embed.weight[0].view(1, 1, -1)
        #var_states = self.var_proj(z_var_tokens) + self.type_embed.weight[1].view(1, 1, -1)
        #recon_var_states = self.scalar_proj(recon_var_tokens) + self.type_embed.weight[2].view(1, 1, -1)
        #recon_state = self._scalar_state(recon_score, 3)
        #svdd_state = self._scalar_state(svdd_score, 4)
        #unc_state = self._scalar_state(unc_score, 5)
        #fused_state = self._scalar_state(fused_score, 6)
        #margin_state = self._scalar_state(margin, 7)
        #flag_state = self._scalar_state(soft_flag, 8)
        #threshold_state = self._scalar_state(score_threshold, 9)
        enc_states = mean_states
        #var_states, #recon_var_states, #recon_state,#svdd_state,#unc_state,#fused_state, #margin_state,#flag_state,   #threshold_state, ], dim=1, )

        enc_mask = torch.ones(
            enc_states.size(0),
            enc_states.size(1),
            dtype=torch.long,
            device=enc_states.device,
        )
        return enc_states, enc_mask


class LatentMemoryBridgeModel(nn.Module):
    """
    Frozen NVAE + trainable LLM bridge using latent memory.
    """

    def __init__(
        self,
        nvae_ckpt: str,
        text_model_name: str = "meta-llama/Meta-Llama-3-8B",
        hidden_size: int = None,
        mc_passes: int = 10,
    ):
        super().__init__()


        load_map = "cpu"
        try:
            ckpt = torch.load(nvae_ckpt, map_location=load_map, weights_only=False)
        except TypeError:
            ckpt = torch.load(nvae_ckpt, map_location=load_map)

        nvae_args = ckpt["args"]
        self.nvae_utils = _load_nvae_utils_module(NVAE_ROOT)
        arch_instance = self.nvae_utils.get_arch_cells(nvae_args.arch_instance)

        self.nvae = AutoEncoder(nvae_args, writer=None, arch_instance=arch_instance)
        self.nvae.load_state_dict(ckpt["state_dict"], strict=False)

        self.mc_passes = int(mc_passes)
        self.num_x_bits = int(nvae_args.num_x_bits)
        self.num_latent_groups = sum(self.nvae.groups_per_scale)
        self.latent_dim = int(nvae_args.num_latent_per_group)

        #self.alpha_score = float(getattr(nvae_args, "alpha_score", 1.0))
        #self.beta_score = float(getattr(nvae_args, "beta_score", 1.0))
        #self.gamma_score = float(getattr(nvae_args, "gamma_score", 1.0))

        """svdd_center = ckpt.get("svdd_center")
        if svdd_center is None:
            svdd_center = torch.zeros(self.num_latent_groups * self.latent_dim)
        self.register_buffer("svdd_center", svdd_center.float(), persistent=False)
        self.register_buffer(
            "score_threshold",
            torch.tensor(float(ckpt.get("score_threshold", 0.0)), dtype=torch.float32),
            persistent=False,
        )"""

        """self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})"""

        """config = GPT2Config.from_pretrained(text_model_name)
        config.add_cross_attention = True
        config.is_decoder = True
        self.decoder = GPT2LMHeadModel.from_pretrained(text_model_name, config=config)
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        self.decoder.config.pad_token_id = self.tokenizer.pad_token_id"""

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, padding_side="left", local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(text_model_name, local_files_only=True)
        decoder_dtype = torch.bfloat16
        self.decoder = LlamaForCausalLMWithCrossAttention.from_pretrained(
            text_model_name,
            config=config,
            local_files_only=True,
           dtype=decoder_dtype,
            low_cpu_mem_usage=True,
        )
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        self.decoder.config.pad_token_id = self.tokenizer.pad_token_id
        adapter_hidden_size = hidden_size or self.decoder.config.hidden_size
        self.adapter = LatentMemoryAdapter(
            latent_dim=self.latent_dim,
            hidden_size=adapter_hidden_size,
        )
    

    def _decode_mean(self, output):
        if hasattr(output, "mean") and callable(output.mean):
            return output.mean()
        if hasattr(output, "mean"):
            return output.mean
        if hasattr(output, "dist") and hasattr(output.dist, "mu"):
            return torch.clamp(output.dist.mu, -1.0, 1.0) / 2.0 + 0.5
        return output.sample()

    def _extract_memory(self, images):
        x = self.nvae_utils.pre_process(images, self.num_x_bits)

        mean_passes = []
        #xhat_passes = []

        self.nvae.eval()
        self.nvae.set_mc_dropout(True)
        with torch.no_grad():
            for _ in range(self.mc_passes):
                logits, _, _, _, _, _, posterior_means = self.nvae(
                    x, return_latents=True, return_posterior_means=True
                )
                mean_tokens = torch.stack(
                    [F.adaptive_avg_pool2d(mu_g, (1, 1)).flatten(1) for mu_g in posterior_means],
                    dim=1,
                )
                mean_passes.append(mean_tokens)
                #xhat_passes.append(self._decode_mean(self.nvae.decoder_output(logits)))
        self.nvae.set_mc_dropout(False)

        mean_stack = torch.stack(mean_passes, dim=0)
       # xhat_stack = torch.stack(xhat_passes, dim=0)

        z_mean_tokens = mean_stack.mean(dim=0)
        #z_mean_flat = z_mean_tokens.flatten(1)
        #z_var_tokens = mean_stack.var(dim=0, unbiased=False)

        #recon_var_map = xhat_stack.var(dim=0, unbiased=False)
        #recon_var_tokens = F.adaptive_avg_pool2d(recon_var_map, (2, 2))
        #recon_var_tokens = recon_var_tokens.mean(dim=1).flatten(1).unsqueeze(-1)

        #center = self.svdd_center.to(z_mean_flat.device).unsqueeze(0)
        #svdd_score = torch.sum((z_mean_flat - center) ** 2, dim=1)

        #xhat_mean = xhat_stack.mean(dim=0)
        #recon_score = torch.mean((x - xhat_mean) ** 2, dim=(1, 2, 3))

        #latent_unc = z_var_tokens.mean(dim=(1, 2))
        #recon_unc = xhat_stack.var(dim=0, unbiased=False).mean(dim=(1, 2, 3))
        #total_uncer_score = latent_unc + recon_unc

        #fused_score = ( self.alpha_score * recon_score+ self.beta_score * svdd_score+ self.gamma_score * total_uncer_score)
        #threshold = self.score_threshold.to(fused_score.device).expand_as(fused_score)
        #margin = fused_score - threshold
        #soft_flag = torch.sigmoid(margin)

        enc_states, enc_mask = self.adapter(
            z_mean_tokens=z_mean_tokens,
            #z_var_tokens=z_var_tokens,
            #recon_var_tokens=recon_var_tokens,
            #recon_score=recon_score,
            #svdd_score=svdd_score,
            #unc_score=total_uncer_score,
            #fused_score=fused_score,
            #margin=margin,
            #soft_flag=soft_flag,
            #score_threshold=threshold,
        )

        metrics = {
            "z_mean_tokens": z_mean_tokens,
            #"recon_score": recon_score,
            #"svdd_score": svdd_score,
            #"unc_score": total_uncer_score,
            #"fused_score": fused_score,
            #"margin": margin,
            #"anomaly_flag": soft_flag,
            #"score_threshold": threshold,
            #"z_mean_tokens": z_mean_tokens,
            #"z_var_tokens": z_var_tokens,
        }
        return enc_states, enc_mask, metrics

    def forward(self, images, input_ids, attention_mask):
        enc_states, enc_mask, _ = self._extract_memory(images)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=enc_states,
            encoder_attention_mask=enc_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, images, max_new_tokens=56, num_beams=4, use_cache=False):
        with torch.amp.autocast("cuda", enabled=images.device.type == "cuda"):
            enc_states, enc_mask, _ = self._extract_memory(images)

        bsz = images.size(0)
        bos_id = self.decoder.config.bos_token_id
        if bos_id is None:
            bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            bos_id = self.decoder.config.eos_token_id

        bos = torch.full((bsz, 1), bos_id, dtype=torch.long, device=images.device)


        eos_id = self.decoder.config.eos_token_id
        if eos_id is None:
            eos_id = self.tokenizer.eos_token_id
            

        return self.decoder.generate(
           input_ids=bos,
            attention_mask=torch.ones_like(bos),
            encoder_hidden_states=enc_states,
            encoder_attention_mask=enc_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            no_repeat_ngram_size=2,
            early_stopping=True,
            use_cache=use_cache,
            eos_token_id=eos_id,
            pad_token_id=self.decoder.config.pad_token_id,
        )

    @torch.no_grad()
    def extract_scores(self, images):
        _, _, metrics = self._extract_memory(images)
        return metrics
