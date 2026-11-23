import torch

def time_mix(x, params, state=None):
    B, T, D = x.shape
    C = params['wk'].shape[1]

    # Initialize state if None
    if state is None:
        last_x = torch.zeros(B, D, dtype=torch.float64, device=x.device)
        a = torch.zeros(B, C, dtype=torch.float64, device=x.device)
        b = torch.zeros(B, C, dtype=torch.float64, device=x.device)
        p = torch.full((B, C), float('-inf'), dtype=torch.float64, device=x.device)
    else:
        last_x = state['last_x'].clone()
        a = state['a'].clone()
        b = state['b'].clone()
        p = state['p'].clone()

    # Extract parameters
    mix_k = params['mix_k']
    mix_v = params['mix_v']
    mix_r = params['mix_r']
    wk = params['wk']
    wv = params['wv']
    wr = params['wr']
    wo = params['wo']
    decay_raw = params['decay_raw']
    u = params['u']

    # Precompute ell (negative log decay)
    ell = -torch.exp(decay_raw)

    # Initialize output tensors
    xk_out = torch.empty(B, T, D, dtype=torch.float64, device=x.device)
    xv_out = torch.empty(B, T, D, dtype=torch.float64, device=x.device)
    xr_out = torch.empty(B, T, D, dtype=torch.float64, device=x.device)
    k_out = torch.empty(B, T, C, dtype=torch.float64, device=x.device)
    v_out = torch.empty(B, T, C, dtype=torch.float64, device=x.device)
    receptance_out = torch.empty(B, T, C, dtype=torch.float64, device=x.device)
    wkv_out = torch.empty(B, T, C, dtype=torch.float64, device=x.device)
    output_out = torch.empty(B, T, D, dtype=torch.float64, device=x.device)
    a_trace = torch.empty(B, T, C, dtype=torch.float64, device=x.device)
    b_trace = torch.empty(B, T, C, dtype=torch.float64, device=x.device)
    p_trace = torch.empty(B, T, C, dtype=torch.float64, device=x.device)

    # Time loop
    for t in range(T):
        # 1. Shift (Interpolation)
        # prev is last_x for t=0, else x[:, t-1, :]
        if t == 0:
            prev = last_x
        else:
            prev = x[:, t-1, :]

        # xk, xv, xr
        xk_t = x[:, t, :] * mix_k + prev * (1 - mix_k)
        xv_t = x[:, t, :] * mix_v + prev * (1 - mix_v)
        xr_t = x[:, t, :] * mix_r + prev * (1 - mix_r)

        xk_out[:, t, :] = xk_t
        xv_out[:, t, :] = xv_t
        xr_out[:, t, :] = xr_t

        # 2. Project
        # k, v, r
        k_t = torch.matmul(xk_t, wk)  # [B, C]
        v_t = torch.matmul(xv_t, wv)  # [B, C]
        r_t = torch.matmul(xr_t, wr)  # [B, C]

        # Receptance (stable sigmoid)
        # sigmoid(r) = 1 / (1 + exp(-r))
        # To be stable: use torch.sigmoid which is stable, or manual stable implementation
        receptance_t = torch.sigmoid(r_t)

        k_out[:, t, :] = k_t
        v_out[:, t, :] = v_t
        receptance_out[:, t, :] = receptance_t

        # 3. WKV Scan (Read then Update)
        # Current state: a, b, p (shape [B, C])
        # k_t, v_t, u, ell (shape [C] or [B, C])

        # Read phase
        # z = max(p, u + k_t)
        # Note: p is [B, C], u is [C], k_t is [B, C]
        # u + k_t broadcasts u to [B, C]
        u_k_t = u + k_t
        z = torch.max(p, u_k_t)

        # e_old = exp(p - z)
        # e_new = exp(u + k_t - z)
        # Handle -inf in p: exp(-inf - z) = 0. If z is finite, this is fine.
        # If p is -inf and u+k_t is finite, z = u+k_t. p-z = -inf. exp(-inf)=0.
        # If p is finite, z >= p, so p-z <= 0. exp is stable.
        e_old = torch.exp(p - z)
        e_new = torch.exp(u_k_t - z)

        # wkv_t = (e_old * a + e_new * v_t) / (e_old * b + e_new)
        # Denominator is guaranteed positive by spec (b>=0, e_new>0 since u+k_t finite and z=max(...))
        numerator = e_old * a + e_new * v_t
        denominator = e_old * b + e_new
        wkv_t = numerator / denominator

        wkv_out[:, t, :] = wkv_t

        # Update phase
        # z_next = max(p + ell, k_t)
        # ell is [C], p is [B, C]. p + ell broadcasts.
        p_ell = p + ell
        z_next = torch.max(p_ell, k_t)

        # f_old = exp(p + ell - z_next)
        # f_new = exp(k_t - z_next)
        f_old = torch.exp(p_ell - z_next)
        f_new = torch.exp(k_t - z_next)

        # a_next = f_old * a + f_new * v_t
        # b_next = f_old * b + f_new
        # p_next = z_next
        a_next = f_old * a + f_new * v_t
        b_next = f_old * b + f_new
        p_next = z_next

        a_trace[:, t, :] = a_next
        b_trace[:, t, :] = b_next
        p_trace[:, t, :] = p_next

        # Update state for next step
        a = a_next
        b = b_next
        p = p_next

        # 4. Output
        # output[b,t,d] = sum_c (receptance[b,t,c] * wkv[b,t,c]) * wo[c,d]
        # gated_wkv = receptance_t * wkv_t  [B, C]
        gated_wkv = receptance_t * wkv_t
        output_t = torch.matmul(gated_wkv, wo)  # [B, D]
        output_out[:, t, :] = output_t

    # Final state
    # last_x is the last input token
    final_last_x = x[:, -1, :].clone()

    final_state = {
        'last_x': final_last_x,
        'a': a,
        'b': b,
        'p': p
    }

    return {
        'xk': xk_out,
        'xv': xv_out,
        'xr': xr_out,
        'k': k_out,
        'v': v_out,
        'receptance': receptance_out,
        'wkv': wkv_out,
        'output': output_out,
        'a_trace': a_trace,
        'b_trace': b_trace,
        'p_trace': p_trace,
        'state': final_state
    }
