import torch

def pool(hidden, ws1, ws2, mask):
    """
    hidden: float64 [B, T, D]
    ws1:    float64 [A, D]
    ws2:    float64 [R, A]
    mask:   bool    [B, T]  (True = valid)
    """
    B, T, D = hidden.shape
    A = ws1.shape[0]
    R = ws2.shape[0]

    # Validate inputs
    if B == 0 or T == 0 or D == 0 or A == 0 or R == 0:
        raise ValueError("Dimensions must be positive")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("Each batch item must have at least one valid token")

    # ssa_mask_input: h0 = where(mask[..., None], hidden, 0)
    h0 = torch.where(mask.unsqueeze(-1), hidden, torch.zeros_like(hidden))

    # ssa_projection: features[b,t,a] = tanh(sum_d h0[b,t,d] * ws1[a,d])
    # h0: [B, T, D], ws1: [A, D] -> features: [B, T, A]
    features = torch.tanh(torch.einsum('btd,ad->bta', h0, ws1))

    # ssa_hop_scores: raw[b,r,t] = sum_a features[b,t,a] * ws2[r,a]
    # features: [B, T, A], ws2: [R, A] -> raw: [B, R, T]
    raw = torch.einsum('bta,ra->brt', features, ws2)

    # scores = where(mask[:, None, :], raw, 0)
    scores = torch.where(mask.unsqueeze(1), raw, torch.zeros_like(raw))

    # ssa_attention: masked softmax over token axis (axis 2)
    # For each b, r: z = max_{t valid} raw[b,r,t]
    # attention[b,r,t] = exp(raw[b,r,t] - z) / sum_{s valid} exp(raw[b,r,s] - z) if mask[b,t] else 0

    # Create a mask for raw: [B, R, T]
    mask_expanded = mask.unsqueeze(1)  # [B, 1, T] -> broadcast to [B, R, T]

    # Set invalid positions to -inf for max and softmax
    neg_inf = torch.tensor(float('-inf'), dtype=raw.dtype, device=raw.device)
    raw_masked = torch.where(mask_expanded, raw, neg_inf)

    # z = max over valid tokens for each (b, r)
    z = raw_masked.max(dim=2, keepdim=True).values  # [B, R, 1]

    # exp(raw - z) for valid positions, 0 for invalid
    exp_vals = torch.exp(raw_masked - z)  # [B, R, T]
    # For invalid positions, raw_masked is -inf, so exp(-inf - z) = 0 (or nan if z is -inf, but z is finite since at least one valid token)
    # Actually, if all tokens are invalid for some (b,r), z would be -inf, but we guaranteed at least one valid token per batch item,
    # so for each b, there's at least one valid t, hence z is finite for all r.
    
    # Sum over valid tokens
    denom = exp_vals.sum(dim=2, keepdim=True)  # [B, R, 1]
    
    attention = exp_vals / denom  # [B, R, T]
    
    # Ensure invalid positions are exactly 0
    attention = torch.where(mask_expanded, attention, torch.zeros_like(attention))

    # ssa_pool: output[b,r,d] = sum_t attention[b,r,t] * h0[b,t,d]
    # attention: [B, R, T], h0: [B, T, D] -> output: [B, R, D]
    output = torch.einsum('brt,btd->brd', attention, h0)

    return {
        'features': features,
        'scores': scores,
        'attention': attention,
        'output': output,
    }
