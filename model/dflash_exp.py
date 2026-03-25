from typing import Optional, Callable
from typing_extensions import Unpack, Tuple
import torch
import time
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from .kvcache import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from .utils import build_target_layer_ids, extract_context_feature, sample, build_hybrid_attention_mask

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        
        '''
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        '''
        # Q projection (only for hidden_states)
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        
        kv_input = torch.cat([target_hidden, hidden_states], dim=1)
        k = self.k_proj(kv_input).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = self.v_proj(kv_input).view(bsz, ctx_len + q_len, -1, self.head_dim)
        
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # basically we don't have kv cache
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.default_target_layer_ids = list(range(config.num_target_layers))
        self.target_layer_ids = self.config.dflash_config.get("target_layer_ids", self.default_target_layer_ids)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.default_target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        # Statistics tracking
        self._last_decode_stats = None
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden)) # turn on when concat on feature
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)
    
    def get_last_decode_stats(self):
        """Return statistics from the last generation call"""
        return self._last_decode_stats
    
    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
    ):
        self.eval() 
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # Prefill stage
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature)
        target_hidden = extract_context_feature(output.hidden_states, self.default_target_layer_ids)[:, -1:, :]

        # Decode stage
        acceptance_lengths = []
        target_total_time = 0.0
        draft_total_time = 0.0
        steps = 0
        start = input_ids.shape[1]
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            
            # Draft model forward pass
            draft_start_time = time.time()
            draft_hidden = self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, start-1: start + block_size],
                past_key_values=None,
                use_cache=True,
                is_causal=False,
            )
            draft_logits = target.lm_head(draft_hidden[:, -block_size+1:, :])
            draft_total_time += time.time() - draft_start_time
            # past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)

            # Target model verification
            target_start_time = time.time()
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            target_total_time += time.time() - target_start_time

            posterior = sample(output.logits, temperature)
            acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(output.hidden_states, self.default_target_layer_ids)[:, acceptance_length:acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length+1)
            steps += 1
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
            ):
                break
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]
        
        # Store statistics
        self._last_decode_stats = {
            "accept_lengths": acceptance_lengths,
            "target_total_time": target_total_time,
            "draft_total_time": draft_total_time,
            "steps": steps,
        }
                
        return output_ids
    

    @torch.inference_mode()
    def naive_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
    ):
        self.eval() 
        bsz, num_input_tokens = input_ids.shape
        max_length = num_input_tokens + max_new_tokens

        output_ids = torch.full(
            (bsz, max_length),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(max_length, device=target.device).unsqueeze(0)

        past_key_values_target = DynamicCache()

        # Prefill target model on the full prompt once.
        target_start_time = time.time()
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=False,
        )
        target_total_time = time.time() - target_start_time

        output_ids[:, :num_input_tokens] = input_ids
        next_token = sample(output.logits, temperature)
        output_ids[:, num_input_tokens : num_input_tokens + 1] = next_token

        # Autoregressive decode with target model only.
        draft_total_time = 0.0
        steps = 1
        cur_len = num_input_tokens + 1
        group_size = 16
        num_layers = len(target.model.layers)
        token_hidden_index = 0
        layer_group_cosine_similarity = []
        current_group_refs = None
        stop_token_ids_tensor = (
            torch.tensor(stop_token_ids, device=output_ids.device) if stop_token_ids is not None else None
        )
        hit_stop = stop_token_ids_tensor is not None and torch.isin(next_token.view(-1), stop_token_ids_tensor).any()

        while cur_len < max_length and not hit_stop:
            step_input_ids = output_ids[:, cur_len - 1 : cur_len]
            step_position_ids = position_ids[:, cur_len - 1 : cur_len]
            target_start_time = time.time()
            output = target(
                step_input_ids,
                position_ids=step_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                logits_to_keep=1,
                output_hidden_states=True,
            )
            target_total_time += time.time() - target_start_time

            # Track cosine similarity of per-layer hidden states in groups of 16 tokens.
            # For each group, compare each token's hidden state against the first token in that group.
            group_idx = token_hidden_index // group_size
            group_offset = token_hidden_index % group_size
            if group_offset == 0:
                current_group_refs = []
                layer_group_cosine_similarity.append(
                    {
                        "group_index": group_idx,
                        "start_token_index": token_hidden_index,
                        "cosine_by_layer": [[] for _ in range(num_layers)],
                    }
                )

            group_record = layer_group_cosine_similarity[group_idx]
            for layer_idx in range(num_layers):
                layer_hidden = output.hidden_states[layer_idx + 1][0, -1, :].float()
                if group_offset == 0:
                    current_group_refs.append(layer_hidden)
                    sim = 1.0
                else:
                    ref_hidden = current_group_refs[layer_idx]
                    sim = torch.nn.functional.cosine_similarity(layer_hidden, ref_hidden, dim=0).item()
                group_record["cosine_by_layer"][layer_idx].append(sim)
            token_hidden_index += 1

            next_token = sample(output.logits, temperature)
            output_ids[:, cur_len : cur_len + 1] = next_token
            cur_len += 1
            steps += 1
            hit_stop = stop_token_ids_tensor is not None and torch.isin(next_token.view(-1), stop_token_ids_tensor).any()

        output_ids = output_ids[:, :cur_len]
        if stop_token_ids_tensor is not None:
            stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]
        
        # Store statistics
        self._last_decode_stats = {
            "accept_lengths": [],
            "target_total_time": target_total_time,
            "draft_total_time": draft_total_time,
            "steps": steps,
            "hidden_similarity_group_size": group_size,
            "layer_group_cosine_similarity": layer_group_cosine_similarity,
        }
                
        return output_ids
    
    @torch.inference_mode()
    # with native hybrid kv cache. Training aligns with inference
    def spec_generate_with_nkv(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        draft_kv_len: Optional[int] = 0,
    ):
        self.eval() 
        bsz, num_input_tokens = input_ids.shape
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)

        past_key_values_target = DynamicCache()

        if draft_kv_len:
            past_key_values_draft = DynamicCache()

        # Prefill stage
        # ----- target model forward -----
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature)
        target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)

        # Decode stage
        acceptance_lengths = []
        target_total_time = 0.0
        draft_total_time = 0.0
        steps = 0
        start = input_ids.shape[1]

        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            
            # Draft model forward pass
            # Build hybrid attention mask here
            if draft_kv_len > 0:
                attention_mask = build_hybrid_attention_mask(bsz, start, block_size)


            draft_start_time = time.time()
            draft_hidden = self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                attention_mask=attention_mask if draft_kv_len else None,
                position_ids=position_ids[:, start: start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )

            draft_logits = target.lm_head(draft_hidden[:, -block_size+1:, :])
            draft_total_time += time.time() - draft_start_time

            if draft_kv_len > 0:
                past_key_values_draft.keep(draft_kv_len, block_size)

            block_output_ids[:, 1:] = sample(draft_logits)

            # Target model verification
            target_start_time = time.time()
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            target_total_time += time.time() - target_start_time

            posterior = sample(output.logits, temperature)
            acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)[:, :acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length+1)
            steps += 1
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
            ):
                break
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]
        
        # Store statistics
        self._last_decode_stats = {
            "accept_lengths": acceptance_lengths,
            "target_total_time": target_total_time,
            "draft_total_time": draft_total_time,
            "steps": steps,
        }
                
        return output_ids
