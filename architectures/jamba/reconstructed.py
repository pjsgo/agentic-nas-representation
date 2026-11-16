import torch

def jamba(x, parameters, state=None):
    B, T, D = x.shape
    assert D == 4
    assert len(parameters) == 8
    
    if state is None:
        state = [None] * 8
    
    h = x
    layers_out = []
    routes_out = []
    new_states = []
    
    for l in range(8):
        p = parameters[l]
        s = state[l]
        
        # 1. Pre-mixer: RMSNorm
        # RMS(z,g) = z * g / sqrt(mean(z*z, -1, keepdim=True) + 1e-6)
        mean_sq = torch.mean(h * h, dim=-1, keepdim=True)
        u = h * p['n1'] / torch.sqrt(mean_sq + 1e-6)
        
        # 2. Mixer (Attention or Mamba)
        if l == 4:
            # Attention
            # Q = reshape(u @ q, [B, T, 2, 2])
            # Knew = u @ k
            # Vnew = u @ v
            Q = torch.matmul(u, p['q']) # [B, T, 4]
            Q = Q.view(B, T, 2, 2)     # [B, T, H, D_head]
            
            Knew = torch.matmul(u, p['k']) # [B, T, 2]
            Vnew = torch.matmul(u, p['v']) # [B, T, 2]
            
            # State handling
            if s is None:
                K_prev = torch.empty(B, 0, 2, dtype=x.dtype, device=x.device)
                V_prev = torch.empty(B, 0, 2, dtype=x.dtype, device=x.device)
            else:
                K_prev = s['k']
                V_prev = s['v']
            
            K = torch.cat([K_prev, Knew], dim=1) # [B, L+T, 2]
            V = torch.cat([V_prev, Vnew], dim=1) # [B, L+T, 2]
            
            # Causal Attention
            # scores[b, t, a, s_idx] = dot(Q[b, t, a, :], K[b, s_idx, :]) / sqrt(2)
            # Q: [B, T, 2, 2] -> [B, T, 2, 1, 2] for broadcasting with K
            # K: [B, L+T, 2] -> [B, 1, 1, L+T, 2]
            
            # We need scores for each t in 0..T-1, considering s in 0..L+t
            # Let's compute full scores matrix [B, T, 2, L+T] and mask
            
            # Q_expanded: [B, T, 2, 1, 2]
            Q_exp = Q.unsqueeze(3) # [B, T, 2, 1, 2]
            # K_expanded: [B, 1, 1, L+T, 2]
            K_exp = K.unsqueeze(1).unsqueeze(1) # [B, 1, 1, L+T, 2]
            
            # scores: [B, T, 2, L+T]
            scores = torch.einsum('btha,bsa->bths', Q, K) # [B, T, 2, L+T]
            scores = scores / (2 ** 0.5)
            
            # Mask: for token t (0-indexed), valid keys are 0..L+t
            # Create mask [T, L+T] where mask[t, s] = 1 if s <= L+t else 0
            L = K.shape[1] - T
            # Indices for s: 0..L+T-1
            s_indices = torch.arange(L + T, device=x.device).view(1, -1) # [1, L+T]
            t_indices = torch.arange(T, device=x.device).view(-1, 1)    # [T, 1]
            
            # Valid if s_indices <= L + t_indices
            mask = (s_indices <= (L + t_indices)).float() # [T, L+T]
            mask = mask.view(1, T, 1, L + T) # [1, T, 1, L+T]
            
            # Apply mask: set invalid to -inf
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
            # Softmax over last dim
            p_attn = torch.softmax(scores, dim=-1) # [B, T, 2, L+T]
            
            # c[b, t, a, :] = sum_s p[s] * V[b, s, :]
            # V: [B, L+T, 2] -> [B, 1, 1, L+T, 2]
            V_exp = V.unsqueeze(1).unsqueeze(1) # [B, 1, 1, L+T, 2]
            
            # p_attn: [B, T, 2, L+T] -> [B, T, 2, L+T, 1]
            p_attn_exp = p_attn.unsqueeze(-1)
            
            c = torch.einsum('bths,bsa->btha', p_attn, V) # [B, T, 2, 2]
            
            # mixer = reshape(c, [B, T, 4]) @ o
            c_flat = c.view(B, T, 4)
            mixer_out = torch.matmul(c_flat, p['o']) # [B, T, 4]
            
            new_state_l = {
                'k': K,
                'v': V
            }
        else:
            # Mamba
            # Parameters
            in_proj = p['in']       # [4, 16]
            conv_kernel = p['conv'] # [8, 3]
            cb = p['cb']            # [8]
            select = p['select']    # [8, 6]
            nt = p['nt']            # [2]
            nb = p['nb']            # [2]
            nc = p['nc']            # [2]
            dt = p['dt']            # [2, 8]
            db = p['db']            # [8]
            A_log = p['A_log']      # [8, 2]
            skip = p['skip']        # [8]
            out_proj = p['out']     # [8, 4]
            
            # State
            if s is None:
                conv_hist = torch.zeros(B, 8, 3, dtype=x.dtype, device=x.device)
                ssm_state = torch.zeros(B, 8, 2, dtype=x.dtype, device=x.device)
            else:
                conv_hist = s['conv'].clone() # [B, 8, 3]
                ssm_state = s['ssm'].clone()  # [B, 8, 2]
            
            # Pre-compute A
            A = -torch.exp(A_log) # [8, 2]
            
            mixer_out = torch.empty(B, T, 4, dtype=x.dtype, device=x.device)
            
            # Process tokens sequentially
            for t in range(T):
                # u[:, t, :] is [B, 4]
                u_t = u[:, t, :] # [B, 4]
                
                # [raw, z] = split(u_t @ in_proj, 8, 8)
                proj = torch.matmul(u_t, in_proj) # [B, 16]
                raw = proj[:, :8] # [B, 8]
                z = proj[:, 8:]   # [B, 8]
                
                # Update conv history
                # Shift left by one, append raw at index 2
                # conv_hist is [B, 8, 3]. Indices 0,1,2. Oldest to newest.
                # New history: [old[1], old[2], raw]
                conv_hist = torch.cat([conv_hist[:, :, 1:], raw.unsqueeze(-1)], dim=-1) # [B, 8, 3]
                
                # a = silu(sum_j conv_hist[:,:,j] * conv_kernel[:,j] + cb)
                # conv_hist: [B, 8, 3]
                # conv_kernel: [8, 3] -> [1, 8, 3]
                conv_kernel_exp = conv_kernel.unsqueeze(0) # [1, 8, 3]
                
                conv_sum = torch.sum(conv_hist * conv_kernel_exp, dim=-1) # [B, 8]
                a = torch.nn.functional.silu(conv_sum + cb) # [B, 8]
                
                # [d0, B0, C0] = split(a @ select, 2, 2, 2)
                sel_out = torch.matmul(a, select) # [B, 6]
                d0 = sel_out[:, :2] # [B, 2]
                B0 = sel_out[:, 2:4] # [B, 2]
                C0 = sel_out[:, 4:6] # [B, 2]
                
                # d = RMS(d0, nt)
                mean_d = torch.mean(d0 * d0, dim=-1, keepdim=True)
                d = d0 * nt / torch.sqrt(mean_d + 1e-6) # [B, 2]
                
                # Bv = RMS(B0, nb)
                mean_B = torch.mean(B0 * B0, dim=-1, keepdim=True)
                Bv = B0 * nb / torch.sqrt(mean_B + 1e-6) # [B, 2]
                
                # Cv = RMS(C0, nc)
                mean_C = torch.mean(C0 * C0, dim=-1, keepdim=True)
                Cv = C0 * nc / torch.sqrt(mean_C + 1e-6) # [B, 2]
                
                # delta = log(1 + exp(d @ dt + db))
                # d: [B, 2], dt: [2, 8] -> d @ dt: [B, 8]
                dt_val = torch.matmul(d, dt) + db # [B, 8]
                delta = torch.log1p(torch.exp(dt_val)) # [B, 8]
                
                # ssm update
                # ssm[b, i, n] = exp(delta[b, i] * A[i, n]) * ssm[b, i, n] + delta[b, i] * Bv[b, n] * a[b, i]
                
                # delta: [B, 8]
                # A: [8, 2] -> A_exp: [1, 8, 2]
                A_exp = A.unsqueeze(0) # [1, 8, 2]
                
                # delta_exp: [B, 8, 1]
                delta_exp = delta.unsqueeze(-1) # [B, 8, 1]
                
                # decay: [B, 8, 2]
                decay = torch.exp(delta_exp * A_exp) # [B, 8, 2]
                
                # ssm_state: [B, 8, 2]
                ssm_state = decay * ssm_state # [B, 8, 2]
                
                # input term: delta[b, i] * Bv[b, n] * a[b, i]
                # delta: [B, 8] -> [B, 8, 1]
                # Bv: [B, 2] -> [B, 1, 2]
                # a: [B, 8] -> [B, 8, 1]
                
                delta_exp2 = delta.unsqueeze(-1) # [B, 8, 1]
                Bv_exp = Bv.unsqueeze(1) # [B, 1, 2]
                a_exp = a.unsqueeze(-1) # [B, 8, 1]
                
                input_term = delta_exp2 * Bv_exp * a_exp # [B, 8, 2]
                
                ssm_state = ssm_state + input_term # [B, 8, 2]
                
                # y[b, i] = (sum_n ssm[b, i, n] * Cv[b, n] + skip[i] * a[b, i]) * silu(z[b, i])
                
                # ssm_state: [B, 8, 2]
                # Cv: [B, 2] -> [B, 1, 2]
                Cv_exp = Cv.unsqueeze(1) # [B, 1, 2]
                
                ssm_dot_cv = torch.sum(ssm_state * Cv_exp, dim=-1) # [B, 8]
                
                # skip: [8] -> [1, 8]
                skip_exp = skip.unsqueeze(0) # [1, 8]
                
                y = (ssm_dot_cv + skip_exp * a) * torch.nn.functional.silu(z) # [B, 8]
                
                # mixer[:, t, :] = y @ out
                mixer_out[:, t, :] = torch.matmul(y, out_proj) # [B, 4]
            
            new_state_l = {
                'conv': conv_hist,
                'ssm': ssm_state
            }
        
        # 3. Residual connection
        r = h + mixer_out
        
        # 4. Pre-FFN: RMSNorm
        mean_r = torch.mean(r * r, dim=-1, keepdim=True)
        v = r * p['n2'] / torch.sqrt(mean_r + 1e-6)
        
        # 5. Experts
        if l % 2 == 0:
            # Even layer: 1 dense expert
            # expert_0(v) = (silu(v @ gate[0]) * (v @ up[0])) @ down[0]
            gate_0 = p['gate'][0] # [4, 6]
            up_0 = p['up'][0]     # [4, 6]
            down_0 = p['down'][0] # [6, 4]
            
            g = torch.matmul(v, gate_0) # [B, T, 6]
            u_exp = torch.matmul(v, up_0) # [B, T, 6]
            
            f = torch.matmul(torch.nn.functional.silu(g) * u_exp, down_0) # [B, T, 4]
            routes_l = None
        else:
            # Odd layer: 4 experts, top-2 routing
            router = p['router'] # [4, 4]
            
            logits = torch.matmul(v, router) # [B, T, 4]
            
            # Softmax over all 4 experts
            routes = torch.softmax(logits, dim=-1) # [B, T, 4]
            
            # Select top 2 probabilities
            # We need indices of top 2
            top2_vals, top2_idx = torch.topk(routes, 2, dim=-1) # [B, T, 2]
            
            # Compute expert outputs for all 4 experts? Or just selected?
            # "f = sum_selected routes[e] * expert_e(v)"
            # It is more efficient to compute all and mask, or compute only selected.
            # Given small size, computing all is fine.
            
            # gate: [4, 4, 6], up: [4, 4, 6], down: [4, 6, 4]
            # v: [B, T, 4]
            
            # Compute all expert outputs
            # v_exp: [B, T, 1, 4]
            v_exp = v.unsqueeze(2) # [B, T, 1, 4]
            
            # gate: [4, 4, 6] -> [1, 1, 4, 4, 6]
            gate_exp = p['gate'].unsqueeze(0).unsqueeze(0) # [1, 1, 4, 4, 6]
            up_exp = p['up'].unsqueeze(0).unsqueeze(0)     # [1, 1, 4, 4, 6]
            down_exp = p['down'].unsqueeze(0).unsqueeze(0) # [1, 1, 4, 6, 4]
            
            # g_all: [B, T, 4, 6]
            g_all = torch.einsum('btd,edf->btef', v, p['gate']) # [B, T, 4, 6]
            u_all = torch.einsum('btd,edf->btef', v, p['up'])   # [B, T, 4, 6]
            
            # f_all: [B, T, 4, 4]
            f_all = torch.einsum('btef,efd->bted', torch.nn.functional.silu(g_all) * u_all, p['down']) # [B, T, 4, 4]
            
            # Select top 2
            # routes: [B, T, 4]
            # top2_idx: [B, T, 2]
            
            # We need to sum routes[e] * f_all[:, :, e, :] for e in top2_idx
            
            # Expand routes to [B, T, 4, 1]
            routes_exp = routes.unsqueeze(-1) # [B, T, 4, 1]
            
            # Create mask for top 2
            # mask: [B, T, 4] where 1 if index is in top2_idx
            
            # One way: gather f_all using top2_idx
            # f_selected: [B, T, 2, 4]
            # top2_idx: [B, T, 2] -> [B, T, 2, 1] -> [B, T, 2, 1, 1] for gathering from [B, T, 4, 4]
            
            idx_exp = top2_idx.unsqueeze(-1).unsqueeze(-1) # [B, T, 2, 1, 1]
            
            # f_all: [B, T, 4, 4] -> [B, T, 1, 4, 4] for broadcasting? No, gather needs matching dims.
            # Let's use torch.gather
            # f_all: [B, T, 4, 4]
            # We want to gather along dim 2 (expert dim)
            
            # Expand idx to [B, T, 2, 4] to match last dim of f_all? No, gather works on specific dim.
            # f_all: [B, T, 4, 4]
            # idx: [B, T, 2]
            
            # Let's expand f_all to [B, T, 4, 1, 4] and idx to [B, T, 1, 2, 1] ? No.
            
            # Simpler: loop over the 2 selected experts? Or use advanced indexing.
            
            # Let's use advanced indexing.
            # b_idx, t_idx, e_idx
            b_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, T, 2)
            t_idx = torch.arange(T, device=x.device).view(1, T, 1).expand(B, T, 2)
            e_idx = top2_idx # [B, T, 2]
            
            # f_selected: [B, T, 2, 4]
            f_selected = f_all[b_idx, t_idx, e_idx] # [B, T, 2, 4]
            
            # r_selected: [B, T, 2]
            r_selected = routes[b_idx, t_idx, e_idx] # [B, T, 2]
            
            # f = sum(r_selected * f_selected, dim=2)
            f = torch.sum(r_selected.unsqueeze(-1) * f_selected, dim=2) # [B, T, 4]
            
            routes_l = routes
        
        # 6. Layer Output
        h = r + f
        layers_out.append(h)
        routes_out.append(routes_l)
        new_states.append(new_state_l)
    
    return {
        'output': h,
        'layers': layers_out,
        'routes': routes_out,
        'state': new_states
    }