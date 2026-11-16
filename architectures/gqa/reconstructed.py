import math
import torch

def attention(q, k, v, cache_k, cache_v):
    B, H, T, D = q.shape
    G = k.shape[1]
    P = cache_k.shape[2]

    # 1. gqa_append: Concatenate cache and new k/v along token axis (dim 2)
    # Ensure fresh storage as per state aliasing rules
    new_k = torch.cat([cache_k, k], dim=2)
    new_v = torch.cat([cache_v, v], dim=2)

    # 2. gqa_group: Determine group mapping
    # r = H / G
    r = H // G

    # 3. gqa_scores: Compute scores
    # s[b, h, t, j] = sum_d q[b, h, t, d] * new_k[b, floor(h/r), j, d] / sqrt(D)

    # We need to map each query head h to its corresponding KV head g = h // r.
    # Instead of looping over heads, we can expand the KV heads to match query heads or use indexing.
    # Since H is divisible by G, we can create an index tensor for KV heads.
    # kv_indices: [H] where kv_indices[h] = h // r
    kv_indices = torch.arange(H, device=q.device) // r

    # new_k shape: [B, G, P+T, D]
    # We need to gather the correct KV head for each query head.
    # new_k_expanded: [B, H, P+T, D] where new_k_expanded[:, h, :, :] = new_k[:, kv_indices[h], :, :]
    # Using advanced indexing: new_k[:, kv_indices, :, :]
    # This results in shape [B, H, P+T, D]
    new_k_for_q = new_k[:, kv_indices, :, :]
    new_v_for_q = new_v[:, kv_indices, :, :]

    # Compute scores: q [B, H, T, D] @ new_k_for_q [B, H, P+T, D]^T
    # Result s: [B, H, T, P+T]
    scale = 1.0 / math.sqrt(D)
    s = torch.matmul(q, new_k_for_q.transpose(-2, -1)) * scale

    # 4. gqa_causal: Apply causal mask
    # Query t has absolute position P + t.
    # Allow exactly 0 <= j <= P + t.
    # j is the index in the concatenated sequence (0 to P+T-1).
    # t is the index in the new query sequence (0 to T-1).
    # Condition: j <= P + t

    # Create mask of shape [T, P+T]
    # t_indices: [T] -> 0, 1, ..., T-1
    # j_indices: [P+T] -> 0, 1, ..., P+T-1
    t_indices = torch.arange(T, device=q.device).unsqueeze(1) # [T, 1]
    j_indices = torch.arange(P + T, device=q.device).unsqueeze(0) # [1, P+T]

    # allowed_mask[t, j] is True if j <= P + t
    allowed_mask = j_indices <= (P + t_indices) # [T, P+T]

    # Expand mask to [B, H, T, P+T] for broadcasting
    # mask shape: [1, 1, T, P+T]
    mask = allowed_mask.unsqueeze(0).unsqueeze(0)

    # Set disallowed positions to -inf
    s = s.masked_fill(~mask, float('-inf'))

    # 5. gqa_softmax: Softmax over allowed j
    # p = exp(s - max_allowed(s)) / sum_allowed exp(s - max_allowed(s))
    # PyTorch softmax handles -inf correctly by treating them as 0 in the sum and exp(-inf)=0.
    # However, if a row is all -inf (which shouldn't happen here because j=0 is always allowed for t>=0 since 0 <= P+t),
    # it would result in NaN. But P>=0 and t>=0, so P+t >= 0, so j=0 is always allowed.
    p = torch.softmax(s, dim=-1)

    # 6. gqa_output: Weighted sum
    # y[b, h, t, d] = sum_j p[b, h, t, j] * new_v_for_q[b, h, j, d]
    # p: [B, H, T, P+T]
    # new_v_for_q: [B, H, P+T, D]
    y = torch.matmul(p, new_v_for_q)

    return y, new_k, new_v
