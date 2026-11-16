import torch

def relative_position_attention(q, k, v, relative_key, relative_value, allowed, radius):
    B, H, N, D = q.shape
    
    # 1. Compute relative indices
    # delta[i, j] = j - i
    # index[i, j] = max(-radius, min(radius, delta[i, j])) + radius
    i_idx = torch.arange(N, device=q.device).unsqueeze(1) # [N, 1]
    j_idx = torch.arange(N, device=q.device).unsqueeze(0) # [1, N]
    delta = j_idx - i_idx # [N, N]
    
    # Clip to [-radius, radius]
    delta_clipped = torch.clamp(delta, min=-radius, max=radius)
    
    # Shift to [0, 2*radius]
    indices = delta_clipped + radius # [N, N]
    
    # 2. Lookup relative keys and values
    # relative_key and relative_value are [2*radius+1, D]
    # We need aK[i, j, d] = relative_key[index[i, j], d]
    # indices is [N, N]. We can use it to index the first dimension of relative_key.
    
    # Expand indices to [N, N, 1] to match D dimension for gathering
    indices_expanded = indices.unsqueeze(-1) # [N, N, 1]
    
    # relative_key is [2*radius+1, D]. We need to gather along dim 0.
    # To use torch.gather, we need to expand relative_key to [N, N, 2*radius+1, D] or similar?
    # Actually, simpler: 
    # aK[i, j, d] = relative_key[indices[i, j], d]
    
    # Let's use advanced indexing or gather.
    # relative_key: [L, D] where L = 2*radius + 1
    # indices: [N, N]
    
    # We can expand relative_key to [1, 1, L, D] and indices to [N, N, 1, 1]?
    # Or just use torch.gather on an expanded tensor.
    
    # Let's expand relative_key to [N, N, L, D] is too big.
    # Better: 
    # relative_key_expanded = relative_key.unsqueeze(0).unsqueeze(0) # [1, 1, L, D]
    # indices_expanded = indices.unsqueeze(-1).unsqueeze(-1) # [N, N, 1, 1]
    # This doesn't work directly with gather because gather requires the index tensor to have the same number of dimensions as the input tensor.
    
    # Alternative: 
    # aK = relative_key[indices] # This works if indices is [N, N] and relative_key is [L, D].
    # Result shape: [N, N, D]. This is exactly what we want.
    
    aK = relative_key[indices] # [N, N, D]
    aV = relative_value[indices] # [N, N, D]
    
    # 3. Compute content score
    # content[b, h, i, j] = sum_d q[b, h, i, d] * k[b, h, j, d]
    # q: [B, H, N, D]
    # k: [B, H, N, D]
    
    # We can use einsum or matmul.
    # q: [B, H, N, D]
    # k: [B, H, N, D]
    # content: [B, H, N, N]
    
    content = torch.einsum('bhid,bhjd->bhij', q, k)
    
    # 4. Compute relative key score
    # relative[b, h, i, j] = sum_d q[b, h, i, d] * aK[i, j, d]
    # q: [B, H, N, D]
    # aK: [N, N, D]
    # relative: [B, H, N, N]
    
    # We need to broadcast aK to [B, H, N, N, D] or use einsum.
    # aK is shared across B and H.
    
    relative = torch.einsum('bhid,ijd->bhij', q, aK)
    
    # 5. Scale, mask, softmax
    # logits = (content + relative) / sqrt(D)
    scale = torch.sqrt(torch.tensor(D, dtype=q.dtype, device=q.device))
    logits = (content + relative) / scale
    
    # masked[b, h, i, j] = logits[b, h, i, j] if allowed[b, i, j] else -inf
    # allowed: [B, N, N]
    # We need to broadcast allowed to [B, H, N, N]
    
    allowed_expanded = allowed.unsqueeze(1) # [B, 1, N, N]
    
    # Create mask tensor with -inf where allowed is False
    # Use torch.where
    neg_inf = torch.tensor(float('-inf'), dtype=q.dtype, device=q.device)
    masked_logits = torch.where(allowed_expanded, logits, neg_inf)
    
    # probabilities = softmax(masked_logits, dim=-1)
    # Note: The equation says exp(masked - max_j(masked)) / sum_j(exp(masked - max_j(masked)))
    # This is standard softmax. PyTorch's softmax handles -inf correctly (results in 0).
    
    probabilities = torch.softmax(masked_logits, dim=-1)
    
    # 6. Value aggregation
    # output[b, h, i, d] = sum_j probabilities[b, h, i, j] * (v[b, h, j, d] + aV[i, j, d])
    
    # v: [B, H, N, D]
    # aV: [N, N, D]
    # probabilities: [B, H, N, N]
    
    # We need to compute v[b, h, j, d] + aV[i, j, d]
    # v_expanded: [B, H, N, 1, D] (broadcast over i)
    # aV_expanded: [1, 1, N, N, D] (broadcast over B, H)
    
    # Let's use einsum for the final aggregation.
    # output[b, h, i, d] = sum_j probabilities[b, h, i, j] * (v[b, h, j, d] + aV[i, j, d])
    
    # Term 1: sum_j probabilities[b, h, i, j] * v[b, h, j, d]
    # This is standard attention output.
    out_v = torch.einsum('bhij,bhjd->bhid', probabilities, v)
    
    # Term 2: sum_j probabilities[b, h, i, j] * aV[i, j, d]
    # aV is [N, N, D]
    # probabilities is [B, H, N, N]
    # We need to sum over j.
    # out_rel = torch.einsum('bhij,ijd->bhid', probabilities, aV)
    
    out_rel = torch.einsum('bhij,ijd->bhid', probabilities, aV)
    
    output = out_v + out_rel
    
    return {
        'output': output,
        'logits': logits,
        'probabilities': probabilities,
        'indices': indices
    }