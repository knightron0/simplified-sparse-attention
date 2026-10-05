"""Distillation losses shared by the SSA models."""

import torch.nn.functional as F


def masked_kl_vocab(teacher_logits, student_logits, mask, temperature=1.0):
    """
    KL( teacher || student ) averaged over token positions where mask==1.
    teacher_logits, student_logits: [B, t_keep, V]
    mask: [B, t_keep] (bool or 0/1)
    """
    # teacher probs
    p_t = F.softmax(teacher_logits / temperature, dim=-1)
    # student log-probs
    logp_s = F.log_softmax(student_logits / temperature, dim=-1)

    # tokenwise KL: [B, t_keep]
    kl_tok = F.kl_div(logp_s, p_t, reduction="none").sum(dim=-1)

    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    kl = (kl_tok * mask_f).sum() / denom

    # standard distillation scaling
    return kl * (temperature ** 2)


def masked_smoothl1_vocab(h1, h2, mask):
    loss = F.smooth_l1_loss(h1, h2, reduction="none").sum(dim=-1)

    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    smooth_l1 = (loss * mask_f).sum() / denom

    # standard distillation scaling
    return smooth_l1
