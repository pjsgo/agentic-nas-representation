import torch

def retention(x, wq, wk, wv, state=None, mode="recurrent"):
    B, T, D = x.shape
    H, K = wq.shape[1], wq.shape[2]
    V = wv.shape[2]
    
    if K % 2 != 0:
        raise ValueError("K must be even")
    
    if state is None:
        kv = torch.zeros(B, H, K, V, dtype=torch.float64)
        pos = 0
    else:
        kv = state['kv'].clone()
        pos = state['position']

    # 1. Projections
    # q0: [B, T, D] @ [D, H, K] -> [B, T, H, K]
    q0 = torch.einsum('btd,dhk->bthk', x, wq)
    # k0: [B, T, D] @ [D, H, K] -> [B, T, H, K] / sqrt(K)
    k0 = torch.einsum('btd,dhk->bthk', x, wk) / (K ** 0.5)
    # v: [B, T, D] @ [D, H, V] -> [B, T, H, V]
    v = torch.einsum('btd,dhv->bthv', x, wv)

    # 2. Position Rotation
    L = K // 2
    if L > 1:
        j = torch.arange(L, dtype=torch.float64)
        omega = 10000.0 ** (-j / (L - 1))
    else:
        omega = torch.tensor([1.0], dtype=torch.float64)
    
    # Positions: [T]
    t_indices = torch.arange(T, dtype=torch.float64)
    n = pos + t_indices # [T]
    
    # Angles: [T, L]
    angles = n.unsqueeze(1) * omega.unsqueeze(0)
    
    cos_a = torch.cos(angles) # [T, L]
    sin_a = torch.sin(angles) # [T, L]
    
    # Reshape q0 and k0 to [B, T, H, L, 2]
    q0_pairs = q0.reshape(B, T, H, L, 2)
    k0_pairs = k0.reshape(B, T, H, L, 2)
    
    a_q = q0_pairs[..., 0] # [B, T, H, L]
    b_q = q0_pairs[..., 1] # [B, T, H, L]
    a_k = k0_pairs[..., 0] # [B, T, H, L]
    b_k = k0_pairs[..., 1] # [B, T, H, L]
    
    # Broadcast cos/sin to [1, T, 1, L]
    cos_b = cos_a.unsqueeze(0).unsqueeze(2)
    sin_b = sin_a.unsqueeze(0).unsqueeze(2)
    
    # Rotation
    q_rot_0 = a_q * cos_b - b_q * sin_b
    q_rot_1 = a_q * sin_b + b_q * cos_b
    
    k_rot_0 = a_k * cos_b - b_k * sin_b
    k_rot_1 = a_k * sin_b + b_k * cos_b
    
    # Interleave back to [B, T, H, K]
    q = torch.stack([q_rot_0, q_rot_1], dim=-1).reshape(B, T, H, K)
    k = torch.stack([k_rot_0, k_rot_1], dim=-1).reshape(B, T, H, K)

    # 3. Decay
    h_indices = torch.arange(H, dtype=torch.float64)
    gamma = 1.0 - (2.0 ** (-5.0 - h_indices)) # [H]
    
    # 4. Core Retention
    if mode == "recurrent":
        S = kv.clone() # [B, H, K, V]
        outputs = torch.empty(B, T, H, V, dtype=torch.float64)
        
        for t in range(T):
            # S_t = gamma * S_{t-1} + outer(k_t, v_t)
            # gamma is [H], S is [B, H, K, V]. Broadcast gamma to [1, H, 1, 1]
            S = S * gamma.view(1, H, 1, 1) + torch.einsum('bhi,bhv->bhiv', k[:, t, :, :], v[:, t, :, :])
            
            # output_t = q_t @ S_t
            # q_t: [B, H, K], S_t: [B, H, K, V] -> [B, H, V]
            outputs[:, t, :, :] = torch.einsum('bhi,bhiv->bhv', q[:, t, :, :], S)
        
        final_kv = S
    elif mode == "parallel":
        # output[b,t,h,j] = gamma^(t+1) * (q_t @ S_in) + sum_{s=0}^t gamma^(t-s) * (q_t @ k_s) * v_s
        
        # Precompute q @ S_in: [B, H, V]
        # Wait, q is [B, T, H, K]. We need q_t for each t.
        # Let's do it step by step or vectorized.
        
        # Term 1: gamma^(t+1) * (q_t @ S_in)
        # q_t: [B, T, H, K]
        # S_in: [B, H, K, V]
        # qS_in_t: [B, T, H, V]
        qS_in_t = torch.einsum('bthk,bhkv->bthv', q, kv)
        
        # Powers of gamma: [T, H]
        t_idx = torch.arange(T, dtype=torch.float64)
        # gamma_pow_t_plus_1: [T, H]
        gamma_pow_t_plus_1 = gamma.view(1, H) ** (t_idx.view(T, 1) + 1.0)
        
        term1 = qS_in_t * gamma_pow_t_plus_1.view(1, T, H, 1)
        
        # Term 2: sum_{s=0}^t gamma^(t-s) * (q_t @ k_s) * v_s
        # q_t @ k_s: [B, T, H, K] @ [B, T, H, K] -> [B, T, T, H]
        # qk = torch.einsum('bthk,bshk->btsh', q, k)
        
        # We need sum_s gamma^(t-s) * qk[b,t,s,h] * v[b,s,h,j]
        # Let's compute weights W[t,s] = gamma^(t-s) if s<=t else 0
        # W: [T, T, H]
        
        # Create matrix of t-s
        t_col = t_idx.view(T, 1) # [T, 1]
        s_row = t_idx.view(1, T) # [1, T]
        diff = t_col - s_row # [T, T]
        
        # Mask for s <= t
        mask = (s_row <= t_col).float() # [T, T]
        
        # gamma^(diff) * mask
        # gamma is [H]
        # We need gamma[h]^(diff[t,s])
        # log_gamma = log(gamma) [H]
        # exp(diff * log_gamma) [T, T, H]
        
        log_gamma = torch.log(gamma) # [H]
        # diff: [T, T] -> [T, T, 1]
        # log_gamma: [H] -> [1, 1, H]
        exp_term = torch.exp(diff.unsqueeze(2) * log_gamma.unsqueeze(0).unsqueeze(0)) # [T, T, H]
        
        # Apply mask
        exp_term = exp_term * mask.unsqueeze(2) # [T, T, H]
        
        # Now compute sum_s exp_term[t,s,h] * qk[b,t,s,h] * v[b,s,h,j]
        # qk: [B, T, T, H] (b, t, s, h)
        # v: [B, T, H, V] (b, s, h, v)
        
        # qk = torch.einsum('bthk,bshk->btsh', q, k)
        qk = torch.einsum('bthk,bshk->btsh', q, k)
        
        # Weighted sum over s
        # term2[b,t,h,j] = sum_s exp_term[t,s,h] * qk[b,t,s,h] * v[b,s,h,j]
        # = sum_s (exp_term[t,s,h] * qk[b,t,s,h]) * v[b,s,h,j]
        
        # Let's combine exp_term and qk first
        # exp_term: [T, T, H] -> [1, T, T, H]
        # qk: [B, T, T, H]
        # weighted_qk = exp_term * qk -> [B, T, T, H]
        weighted_qk = qk * exp_term.unsqueeze(0)
        
        # Now sum over s (dim 2) and multiply with v (dim 1 is s in v? No, v is [B, T, H, V] where T is s)
        # We need sum_s weighted_qk[b,t,s,h] * v[b,s,h,j]
        # This is a contraction over s.
        # weighted_qk: [B, T, T, H] (b, t, s, h)
        # v: [B, T, H, V] (b, s, h, v)
        
        # term2 = torch.einsum('btsh,bs hv->bthv', weighted_qk, v)
        term2 = torch.einsum('btsh,bs hv->bthv', weighted_qk, v)
        
        outputs = term1 + term2
        
        # Final State
        # final_kv = gamma^T * S_in + sum_{s=0}^{T-1} gamma^(T-1-s) * outer(k_s, v_s)
        
        # Term A: gamma^T * S_in
        # gamma^T: [H] -> [1, H, 1, 1]
        gamma_T = gamma ** T
        termA = kv * gamma_T.view(1, H, 1, 1)
        
        # Term B: sum_s gamma^(T-1-s) * outer(k_s, v_s)
        # Weights for s: gamma^(T-1-s)
        # s_idx: [T]
        # diff_final = T - 1 - s_idx
        s_idx = torch.arange(T, dtype=torch.float64)
        diff_final = T - 1 - s_idx # [T]
        
        # weights_final: [T, H]
        weights_final = torch.exp(diff_final.unsqueeze(1) * log_gamma.unsqueeze(0)) # [T, H]
        
        # outer(k_s, v_s): [B, T, H, K, V]
        # k: [B, T, H, K], v: [B, T, H, V]
        # outer = torch.einsum('bthk,bthv->bthkv', k, v)
        outer_kv = torch.einsum('bthk,bthv->bthkv', k, v)
        
        # Weighted sum over s
        # weights_final: [T, H] -> [1, T, H, 1, 1]
        # outer_kv: [B, T, H, K, V]
        # termB = sum_s weights_final[s,h] * outer_kv[b,s,h,k,v]
        termB = torch.einsum('th,bthkv->bhkv', weights_final, outer_kv)
        
        final_kv = termA + termB
    else:
        raise ValueError("mode must be 'recurrent' or 'parallel'")

    return {
        'q': q,
        'k': k,
        'v': v,
        'output': outputs,
        'state': {
            'kv': final_kv,
            'position': pos + T
        }
    }
