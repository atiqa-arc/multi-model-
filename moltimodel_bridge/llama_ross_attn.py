import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
except Exception:
    LlamaRMSNorm = nn.LayerNorm


class LlamaForCausalLMWithCrossAttention(LlamaForCausalLM):
    """
    Lightweight LLaMA wrapper that injects encoder memory through a cross-attention
    fusion step on the final decoder hidden states before the LM head.

    This keeps the same interface as your current bridge:
      - encoder_hidden_states
      - encoder_attention_mask
    """

    def __init__(self, config):
        super().__init__(config)

        hidden_size = config.hidden_size
        num_heads = getattr(config, "num_attention_heads", 8)
        norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.cross_attn_norm = LlamaRMSNorm(hidden_size, eps=norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=getattr(config, "attention_dropout", 0.0),
            batch_first=True,
        )
        # A less tiny initialization helps the bridge signal influence decoding early.
        self.cross_attn_gate = nn.Parameter(torch.tensor(0.2))
    
    

    def _fuse_encoder_memory(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if encoder_hidden_states is None:
            return hidden_states

        query = self.cross_attn_norm(hidden_states)
        encoder_hidden_states = encoder_hidden_states.to(
            device=query.device,
            dtype=query.dtype,
        )


        key_padding_mask = None
        if encoder_attention_mask is not None:
            key_padding_mask = encoder_attention_mask == 0

        cross_out, _ = self.cross_attn(
            query=query,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        return hidden_states + self.cross_attn_gate * cross_out

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = model_outputs[0]
        if encoder_hidden_states is not None:
            hidden_states = self._fuse_encoder_memory(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + model_outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )

        if "encoder_hidden_states" in kwargs:
            model_inputs["encoder_hidden_states"] = kwargs["encoder_hidden_states"]
        if "encoder_attention_mask" in kwargs:
            model_inputs["encoder_attention_mask"] = kwargs["encoder_attention_mask"]

        return model_inputs
