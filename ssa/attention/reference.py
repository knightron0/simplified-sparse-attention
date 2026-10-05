"""Eager reference attention helpers shared by the Qwen2 and Llama SSA models.

Moved verbatim from the model modules; the model-specific (drifted) eager
prefill implementations remain in ssa/models/{qwen2,llama}.py.
"""

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from ssa.utils import Global_data


@torch.no_grad()
def top_p_from_logits(logits, p=0.90):
    """
    Exact nucleus sampling mask from logits.

    logits: [..., V]
    returns:
        indices: [..., V] with rejected = -1
        keep_mask: same shape, True where kept
    """

    # sort descending
    sorted_logits, sorted_idx = torch.sort(logits.float(), descending=True, dim=-1)

    # softmax on sorted logits
    probs = torch.softmax(sorted_logits, dim=-1)

    # cumulative prob
    cum = torch.cumsum(probs, dim=-1)

    # first position where cum >= p
    cutoff = cum >= p

    # find first True per row
    first = cutoff.int().argmax(dim=-1)
    never = ~cutoff.any(dim=-1)
    if never.any():
        first = torch.where(never, logits.new_full(first.shape, logits.size(-1) - 1, dtype=first.dtype), first)

    # build mask: positions <= first
    arange = torch.arange(logits.size(-1), device=logits.device)
    keep = arange <= first.unsqueeze(-1)

    # padded output indices (GPU friendly)
    out_idx = sorted_idx.masked_fill(~keep, -1)

    return out_idx, keep

@torch.no_grad()
def top_k_from_logits(logits, k=1):

    _, top_k = torch.topk(logits, k=k, dim=-1)

    logits_topk = logits.gather(dim=-1, index=top_k)
    keep = logits_topk > float("-inf")
    
    out_idx = top_k.masked_fill(~keep, -1)

    return out_idx, keep


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def eager_attention_forward_decoding(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    B, H, _, d = query.shape  # q_len = 1 for decoding
    _, _, k_len, _ = key_states.shape

    # -------------------------------------------------------------------------
    # Hierarchical Multi-Level Gist/Sparse Attention (Decoding)
    # -------------------------------------------------------------------------
    
    input_ids = kwargs.get("tmp_ids", None)
    gist_ids = Global_data.gist_token_id
    PAD_ID = Global_data.pad_token_id
    sink_size = Global_data.sink_size
    causal_mask = kwargs.get("causal_mask", None)
    top_k = copy.deepcopy(Global_data.top_k)
    if Global_data.comp_factor > 0:
        comp_factor = Global_data.comp_factor
    else:  # The same comp_factor as the compressed context
        comp_factor = Global_data.chunk_size[0] if len(gist_ids) == 1 else Global_data.chunk_size[0] * Global_data.chunk_size[1]
    top_k[0] = k_len // comp_factor // module.num_key_value_groups // Global_data.chunk_size[0] + 1
    top_k[1] = top_k[0]
    top_p = Global_data.top_p

    device = query.device
    k_idx = torch.arange(k_len, device=device)

    # -------------------------------------------------------------------------
    # Identify gist tokens at all levels (from lowest to highest)
    # -------------------------------------------------------------------------
    num_levels = len(gist_ids)
    is_gist_per_level = []  # [B, k_len] for each level
    for level_idx in range(num_levels):
        gist_m = (input_ids == gist_ids[level_idx])
        is_gist_per_level.append(gist_m)

    # Find the last gist token at any level to determine suffix start
    last_gist = torch.where(is_gist_per_level[-1], k_idx, torch.full_like(k_idx, -1)).amax(dim=1)  # [B]

    # Sink mask on keys (optional)
    start_idx = (input_ids != PAD_ID).int().argmax(dim=1)  # [B]

    # Keys after last gist always visible
    after_last_key = (k_idx[None, :] > last_gist[:, None])  # [B, k]
    compressed = (~after_last_key & (k_idx[None, :] >= start_idx[:, None]))

    # Attention mask processing
    attention_mask_bool = (attention_mask > -1e-4).bool()  # [B, 1, 1, k_len]

    # -------------------------------------------------------------------------
    # Hierarchical Tree Search: Top-k selection from highest to lowest level
    # For decoding, q_len=1, so this is simpler than prefill
    # -------------------------------------------------------------------------
    keep_mask = torch.ones(B, H, 1, k_len, dtype=torch.bool, device=device) & compressed[:, None, None, :]
    all_selected_gist_pos = torch.zeros_like(keep_mask, dtype=torch.bool)
    pos = k_idx[None, None, None, :].expand(B, H, 1, k_len)  # [B, H, 1, k]

    # Start from highest level (last index) and work down to level 0
    for level_idx in range(num_levels - 1, -1, -1):
        is_gist_current = is_gist_per_level[level_idx]  # [B, k_len]
        Gmax_level = int(is_gist_current.sum(dim=1).max().item())
        if Gmax_level == 0:
            # No gist tokens at this level, skip
            continue

        # Build padded gist positions for current level [B, Gmax_level]
        gist_cumsum = is_gist_current.cumsum(dim=1)  # [B, k_len]
        gather_indices = torch.arange(1, Gmax_level + 1, device=device)[None, :].expand(B, Gmax_level)
        matches = (gist_cumsum[:, :, None] == gather_indices[:, None, :])  # [B, k_len, Gmax_level]
        gist_pos_level = matches.to(torch.float32).argmax(dim=1)  # [B, Gmax_level]
        gist_valid_level = gist_pos_level > 0
        gist_valid_level = gist_valid_level[:, None, None, :]  # [B, 1, 1, Gmax_level]

        gist_pos_safe = gist_pos_level.clamp(0, k_len - 1)
        gist_gather = gist_pos_safe[:, None, :, None].expand(B, H, Gmax_level, d)
        gist_pos_level = gist_pos_level[:, None, None, :].expand(B, H, 1, Gmax_level)
        is_gist_current = is_gist_current[:, None, None, :]
        gist_cumsum = gist_cumsum[:, None, None, :]

        # Compute attention scores for this level's gist tokens
        K_gist_level = key_states.gather(dim=2, index=gist_gather)  # [B, H, Gmax_level, d]
        gist_scores_level = torch.matmul(query.double(), K_gist_level.transpose(2, 3).double())  # [B, H, 1, Gmax_level]

        is_gist_current_selected = is_gist_current & keep_mask

        if level_idx < num_levels - 1:
            Gmax_ = int(is_gist_current_selected.sum(dim=-1).max().item())
            pos_masked = pos.masked_fill(~is_gist_current_selected, k_len)  # [B, k]
            pos_sorted = pos_masked.sort(dim=-1).values  # [B, k]
            gist_pos_level_ = pos_sorted[..., :Gmax_]  # [B, Gmax_]
            gist_valid_level_ = gist_pos_level_ < k_len # [B, Gmax_]
            gist_pos_mask = ((gist_pos_safe[:, None, None, :].expand(B, H, 1, Gmax_level).unsqueeze(-2) == gist_pos_level_.unsqueeze(-1)) & gist_valid_level_.unsqueeze(-1)).any(dim=-2)
            gist_valid_level = gist_valid_level.masked_fill(~gist_pos_mask, False)

        gist_scores_level = gist_scores_level.masked_fill(~gist_valid_level, float("-inf"))

        if top_p <= 0:
            kk_level = min(top_k[level_idx], max(1, Gmax_level // 4))
            top_in_gist_level, top_k_mask = top_k_from_logits(gist_scores_level, k=kk_level)  # [B, H, qs_max, kk_level]
        else:
            top_in_gist_level, top_p_mask = top_p_from_logits(gist_scores_level, p=top_p)
        top_in_gist_level_safe = top_in_gist_level.clamp(min=0)
        selected_chunk_mask = torch.zeros((B, H, 1, Gmax_level + 1), device=device, dtype=torch.bool)
        selected_chunk_mask.scatter_(dim=3, index=top_in_gist_level_safe, value=1)
        selected_chunk_mask[..., 0] = (top_in_gist_level == 0).any(dim=3).to(selected_chunk_mask.dtype)
        chunk_ids_parent_adj = torch.where(is_gist_current, gist_cumsum - 1, gist_cumsum)
        chunk_ids_parent_adj = torch.clamp(chunk_ids_parent_adj, min=0)
        chunk_ids_view = chunk_ids_parent_adj.expand(B, H, 1, k_len)
        in_selected_chunks = selected_chunk_mask.gather(dim=3, index=chunk_ids_view).to(torch.bool)

        # Update keep_mask to only include children of selected gist at this level
        keep_mask &= in_selected_chunks
        # all_selected_gist_pos &= ~in_selected_chunks
        all_selected_gist_pos = all_selected_gist_pos | (in_selected_chunks & is_gist_current)

    keep_mask = keep_mask | all_selected_gist_pos

    # -------------------------------------------------------------------------
    # Group-union: union and deduplicate selected gist positions across all levels within each GQA group
    # -------------------------------------------------------------------------
    use_nsa_gqa = getattr(Global_data, 'use_nsa_gqa', False)

    if use_nsa_gqa and hasattr(module, 'num_key_value_groups') and module.num_key_value_groups > 1:
        heads_per_group = module.num_key_value_groups
        num_kv_heads = H // heads_per_group
        keep_mask_grouped = keep_mask.view(B, num_kv_heads, heads_per_group, 1, k_len)

        # For each KV group, take the union of selected positions across all query heads
        # that share the same KV projection
        keep_mask_union = keep_mask_grouped.any(dim=2, keepdim=True)  # [B, num_kv_heads, 1, 1, k_len]

        # Broadcast back to all heads in the group
        keep_mask_union = keep_mask_union.expand(B, num_kv_heads, heads_per_group, 1, k_len)

        # Reshape back to original shape
        keep_mask = keep_mask_union.reshape(B, H, 1, k_len)

    # -------------------------------------------------------------------------
    # Finalize the keep mask with additional constraints
    # -------------------------------------------------------------------------
    gist_token_tensor = torch.tensor(Global_data.gist_token_id, device=device)
    is_gist = torch.isin(input_ids, gist_token_tensor)
    attention_mask_bool = attention_mask_bool & ~(is_gist[:, None, None, :])
    keep_mask = keep_mask | attention_mask_bool  # [B, H, 1, k_len]
    # keep_mask = keep_mask & ~(is_gist[:, None, None, :])

    # ######################################################
    # # Visualization / Logging
    # mean_select_num = (keep_mask & ~after_last_key[:, None, None, :]).sum(-1).float().mean().int().item()
    # max_select_num = (keep_mask & ~after_last_key[:, None, None, :]).sum(-1).max().item()
    # compressed_length = compressed.sum().item()
    # max_ratio = (max_select_num / compressed_length)
    # mean_ratio = (mean_select_num / compressed_length)

    # max_gain = compressed_length // 4 + 30 - max_select_num
    # mean_gain = compressed_length // 4 + 30 - mean_select_num

    # print(
    #     f"max_select_ratio: {max_ratio:.4f}, "
    #     f"mean_select_ratio: {mean_ratio:.4f}, "
    #     f"compressed_length: {compressed_length}, "
    #     f"{'max_gain:'} {max_gain:<4}, "
    #     f"{'mean_gain:'} {mean_gain:<4}"
    # )
    # ######################################################

    # Convert to float attention mask
    attn_mask_float = causal_mask.masked_fill(~keep_mask, float('-inf'))

    attn_output_select = F.scaled_dot_product_attention(
        query, key_states, value_states,
        attn_mask=attn_mask_float,
        dropout_p=dropout if module.training else 0.0,
        is_causal=False,
        scale=scaling,
    )
    attn_output_select = attn_output_select.transpose(1, 2).contiguous()

    return attn_output_select, None
