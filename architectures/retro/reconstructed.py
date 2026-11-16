import torch


def retro_cca(h, e, q, k, v, position_bias, chunk_size):
    # Validate chunk_size
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError("chunk_size must be a Python int, not bool")
    m = chunk_size
    if m <= 0:
        raise ValueError("chunk_size must be positive")

    # Validate tensor types and basic properties
    tensors = [h, e, q, k, v, position_bias]
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise ValueError("All inputs must be torch.Tensor")
        if t.dtype != torch.float64:
            raise ValueError("All tensors must be torch.float64")
        if t.device.type != 'cpu':
            raise ValueError("All tensors must be on CPU")
        if not torch.isfinite(t).all():
            raise ValueError("All tensors must contain finite values")

    # Validate ranks
    if h.dim() != 3:
        raise ValueError("h must be 3D [B,N,D]")
    if e.dim() != 5:
        raise ValueError("e must be 5D [B,L,K,R,D]")
    if q.dim() != 2:
        raise ValueError("q must be 2D [D,D]")
    if k.dim() != 2:
        raise ValueError("k must be 2D [D,D]")
    if v.dim() != 2:
        raise ValueError("v must be 2D [D,D]")
    if position_bias.dim() != 2:
        raise ValueError("position_bias must be 2D [m,R]")

    # Extract dimensions
    B, N, D = h.shape
    B_e, L, K, R, D_e = e.shape
    D_q, D_q2 = q.shape
    D_k, D_k2 = k.shape
    D_v, D_v2 = v.shape
    m_pb, R_pb = position_bias.shape

    # Validate dimension consistency
    if B_e != B:
        raise ValueError("Batch dimension mismatch between h and e")
    if D_e != D:
        raise ValueError("Feature dimension mismatch between h and e")
    if D_q != D or D_q2 != D:
        raise ValueError("q must be [D,D]")
    if D_k != D or D_k2 != D:
        raise ValueError("k must be [D,D]")
    if D_v != D or D_v2 != D:
        raise ValueError("v must be [D,D]")
    if m_pb != m:
        raise ValueError("position_bias first dimension must equal chunk_size m")
    if R_pb != 2 * m:
        raise ValueError("position_bias second dimension must equal 2*m")
    if R != 2 * m:
        raise ValueError("R must equal 2*m")
    if N % m != 0:
        raise ValueError("N must be divisible by m")
    if L != N // m:
        raise ValueError("L must equal N/m")

    # Check for empty dimensions
    if B == 0 or N == 0 or D == 0 or L == 0 or K == 0 or R == 0:
        raise ValueError("Dimensions must be positive")

    # Compute chunk indices
    # chunk_index[t] = -1 if t < m-1, else floor((t-(m-1))/m)
    t_indices = torch.arange(N, dtype=torch.int64, device='cpu')
    active_mask = t_indices >= (m - 1)
    chunk_index = torch.full((N,), -1, dtype=torch.int64, device='cpu')
    if m > 1:
        active_t = t_indices[active_mask]
        chunk_index[active_mask] = (active_t - (m - 1)) // m
    else:
        # m == 1: active for t >= 0, chunk_index = t
        chunk_index = t_indices.clone()

    # Initialize outputs
    logits = torch.zeros((B, N, K, R), dtype=torch.float64, device='cpu')
    probabilities = torch.zeros((B, N, K, R), dtype=torch.float64, device='cpu')
    delta = torch.zeros((B, N, D), dtype=torch.float64, device='cpu')

    # Projections
    # query[b,t,c] = sum_a h[b,t,a]*q[a,c] -> h @ q
    query = torch.matmul(h, q)  # [B, N, D]

    # key[b,u,j,r,c] = sum_a e[b,u,j,r,a]*k[a,c] -> e @ k
    key = torch.matmul(e, k)  # [B, L, K, R, D]

    # value[b,u,j,r,d] = sum_a e[b,u,j,r,a]*v[a,d] -> e @ v
    value = torch.matmul(e, v)  # [B, L, K, R, D]

    # Process active positions
    if m > 1:
        active_t_indices = t_indices[active_mask]
        if len(active_t_indices) > 0:
            # For each active t, compute attention
            # We can vectorize over batch and active positions
            # active_t_indices: [N_active]
            # u(t) = chunk_index[active_t_indices]  # [N_active]
            # i(t) = (t - (m-1)) % m  # [N_active]
            
            u_active = chunk_index[active_mask]  # [N_active]
            i_active = (active_t_indices - (m - 1)) % m  # [N_active]
            
            # query_active: [B, N_active, D]
            query_active = query[:, active_mask, :]  # [B, N_active, D]
            
            # For each active position, we need key[b, u(t), :, :, :] and value[b, u(t), :, :, :]
            # u_active is [N_active], so we gather along L dimension
            # key_gathered: [B, N_active, K, R, D]
            # We need to index key[b, u_active[t], :, :, :] for each t
            # Use advanced indexing
            
            # Create index tensors for gathering
            # For each b in [0, B), t_idx in [0, N_active):
            #   key[b, u_active[t_idx], :, :, :]
            
            # batch_idx: [B, N_active] -> repeat u_active for each b
            # Actually, we can do:
            # key_gathered[b, t_idx, j, r, c] = key[b, u_active[t_idx], j, r, c]
            
            # Use torch.gather or advanced indexing
            # Let's use advanced indexing:
            # b_idx: [B, N_active] = torch.arange(B).unsqueeze(1).expand(B, N_active)
            # u_idx: [B, N_active] = u_active.unsqueeze(0).expand(B, N_active)
            
            b_idx = torch.arange(B, device='cpu').unsqueeze(1).expand(B, len(active_t_indices))  # [B, N_active]
            u_idx = u_active.unsqueeze(0).expand(B, len(active_t_indices))  # [B, N_active]
            
            # key_gathered: [B, N_active, K, R, D]
            # key[b, u_idx[b,t], :, :, :]
            # We can do: key[b_idx, u_idx, :, :, :]
            # But this gives [B, N_active, K, R, D] if we expand properly
            
            # Actually, key is [B, L, K, R, D]
            # key[b_idx, u_idx] gives [B, N_active, K, R, D] because the last 3 dims are broadcast
            key_gathered = key[b_idx, u_idx]  # [B, N_active, K, R, D]
            value_gathered = value[b_idx, u_idx]  # [B, N_active, K, R, D]
            
            # Now compute logits: [B, N_active, K, R]
            # logits[b, t_idx, j, r] = sum_c query_active[b, t_idx, c] * key_gathered[b, t_idx, j, r, c] + position_bias[i_active[t_idx], r]
            
            # query_active: [B, N_active, D]
            # key_gathered: [B, N_active, K, R, D]
            # Dot product over D: [B, N_active, K, R]
            content_logits = torch.einsum('btd,btjrd->btjr', query_active, key_gathered)
            
            # position_bias: [m, R]
            # i_active: [N_active]
            # pos_bias_gathered: [N_active, R]
            pos_bias_gathered = position_bias[i_active]  # [N_active, R]
            
            # Add position bias: [B, N_active, K, R] + [1, N_active, 1, R]
            logits_active = content_logits + pos_bias_gathered.unsqueeze(0).unsqueeze(2)  # [B, N_active, K, R]
            
            # Softmax over K and R jointly
            # z[b, t] = max over j, r of logits[b, t, j, r]
            z = logits_active.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # [B, N_active, 1, 1]
            
            # probabilities = exp(logits - z) / sum(exp(logits - z))
            exp_logits = torch.exp(logits_active - z)  # [B, N_active, K, R]
            sum_exp = exp_logits.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)  # [B, N_active, 1, 1]
            probabilities_active = exp_logits / sum_exp  # [B, N_active, K, R]
            
            # delta[b, t, d] = sum_{j,r} probabilities[b, t, j, r] * value[b, u(t), j, r, d]
            # probabilities_active: [B, N_active, K, R]
            # value_gathered: [B, N_active, K, R, D]
            delta_active = torch.einsum('btjr,btjrd->btd', probabilities_active, value_gathered)  # [B, N_active, D]
            
            # Store results
            logits[:, active_mask, :, :] = logits_active
            probabilities[:, active_mask, :, :] = probabilities_active
            delta[:, active_mask, :] = delta_active
    else:
        # m == 1: all positions active, u(t) = t, i(t) = 0
        # u_active = t_indices (all positions)
        # i_active = zeros
        u_active = t_indices  # [N]
        i_active = torch.zeros(N, dtype=torch.int64, device='cpu')
        
        query_active = query  # [B, N, D]
        
        b_idx = torch.arange(B, device='cpu').unsqueeze(1).expand(B, N)  # [B, N]
        u_idx = u_active.unsqueeze(0).expand(B, N)  # [B, N]
        
        key_gathered = key[b_idx, u_idx]  # [B, N, K, R, D]
        value_gathered = value[b_idx, u_idx]  # [B, N, K, R, D]
        
        content_logits = torch.einsum('btd,btjrd->btjr', query_active, key_gathered)
        
        pos_bias_gathered = position_bias[i_active]  # [N, R]
        
        logits_active = content_logits + pos_bias_gathered.unsqueeze(0).unsqueeze(2)  # [B, N, K, R]
        
        z = logits_active.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # [B, N, 1, 1]
        
        exp_logits = torch.exp(logits_active - z)  # [B, N, K, R]
        sum_exp = exp_logits.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)  # [B, N, 1, 1]
        probabilities_active = exp_logits / sum_exp  # [B, N, K, R]
        
        delta_active = torch.einsum('btjr,btjrd->btd', probabilities_active, value_gathered)  # [B, N, D]
        
        logits = logits_active
        probabilities = probabilities_active
        delta = delta_active

    # Residual
    output = h + delta

    return {
        'output': output,
        'delta': delta,
        'logits': logits,
        'probabilities': probabilities,
        'chunk_index': chunk_index
    }