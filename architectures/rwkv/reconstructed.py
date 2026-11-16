import torch

def time_mix(x, params, state=None):
    B, T, D = x.shape
    C = params['wk'].shape[1]
    
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
    
    # Precompute ell
    ell = -torch.exp(decay_raw)
    
    # Allocate outputs
    xk = torch.empty(B, T, D, dtype=torch.float64, device=x.device)
    xv = torch.empty(B, T, D, dtype=torch.float64, device=x.device)
    xr = torch.empty(B, T, D, dtype=torch.float64, device=x.device)
    k = torch.empty(B, T, C, dtype=torch.float64, device=x.device)
    v = torch.empty(B, T, C, dtype=torch.float64, device=x.device)
    receptance = torch.empty(B, T, C, dtype=torch.float64, device=x.device)
    wkv = torch.empty(B, T, C, dtype=torch.float64, device=x.device)
    output = torch.empty(B, T, D, dtype=torch.float64, device=x.device)
    a_trace = torch.empty(B, T, C, dtype=torch.float64, device=x.device)
    b_trace = torch.empty(B, T, C, dtype=torch.float64, device=x.device)
    p_trace = torch.empty(B, T, C, dtype=torch.float64, device=x.device)
    
    # Time loop
    for t in range(T):
        # 1. Shift
        if t == 0:
            prev = last_x
        else:
            prev = x[:, t-1, :]
            
        xk[:, t, :] = x[:, t, :] * mix_k + prev * (1 - mix_k)
        xv[:, t, :] = x[:, t, :] * mix_v + prev * (1 - mix_v)
        xr[:, t, :] = x[:, t, :] * mix_r + prev * (1 - mix_r)
        
        # 2. Project
        k[:, t, :] = torch.matmul(xk[:, t, :], wk)
        v[:, t, :] = torch.matmul(xv[:, t, :], wv)
        r = torch.matmul(xr[:, t, :], wr)
        
        # Stable sigmoid
        receptance[:, t, :] = torch.sigmoid(r)
        
        # 3. WKV Scan
        # Current k_t and v_t are [B, C]
        k_t = k[:, t, :]
        v_t = v[:, t, :]
        
        # Read
        z = torch.max(p, u + k_t)
        e_old = torch.exp(p - z)
        e_new = torch.exp(u + k_t - z)
        
        wkv[:, t, :] = (e_old * a + e_new * v_t) / (e_old * b + e_new)
        
        # Update
        z_next = torch.max(p + ell, k_t)
        f_old = torch.exp(p + ell - z_next)
        f_new = torch.exp(k_t - z_next)
        
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
        output[:, t, :] = torch.matmul(receptance[:, t, :] * wkv[:, t, :], wo)
        
        # Update last_x for next step
        last_x = x[:, t, :]
    
    return {
        'xk': xk,
        'xv': xv,
        'xr': xr,
        'k': k,
        'v': v,
        'receptance': receptance,
        'wkv': wkv,
        'output': output,
        'a_trace': a_trace,
        'b_trace': b_trace,
        'p_trace': p_trace,
        'state': {
            'last_x': last_x,
            'a': a,
            'b': b,
            'p': p
        }
    }