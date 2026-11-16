import torch

def sparse_moe(x, router, wi, wo, k):
    T, D = x.shape
    E = router.shape[1]
    H = wi.shape[2]
    O = wo.shape[2]
    
    # 1. Router: logits[t,e] = sum_d x[t,d]*router[d,e]
    logits = torch.matmul(x, router)
    
    # Agent repair: stable descending score sort preserves lower-index ties
    # without changing the ordering of distinct, close scores.
    indices = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :k]

    # Create dispatch mask
    dispatch = torch.zeros(T, E, dtype=torch.bool, device=x.device)
    dispatch.scatter_(1, indices, True)
    
    # Calculate gates
    # m[t] is the maximum selected logit
    selected_logits = torch.gather(logits, 1, indices) # [T, k]
    m = selected_logits.max(dim=1, keepdim=True).values # [T, 1]
    
    # exp(logits[t,e] - m[t]) for selected e
    # We need to compute this for all e, but only use it for selected ones.
    # Let's compute exp(logits - m) for all e, then mask.
    exp_logits = torch.exp(logits - m) # [T, E]
    
    # Sum over selected experts
    # We can use the dispatch mask to zero out unselected ones before summing.
    exp_logits_masked = exp_logits * dispatch.float()
    sum_exp = exp_logits_masked.sum(dim=1, keepdim=True) # [T, 1]
    
    # gates[t,e] = exp_logits_masked[t,e] / sum_exp[t] for selected e, else 0
    gates = exp_logits_masked / sum_exp
    
    # 3. Experts
    # expert_outputs[t,e,o] = sum_h hidden[t,e,h]*wo[e,h,o] * (H/D)^(-0.5) for selected pairs
    # hidden[t,e,h] = max(0, sum_d x[t,d]*wi[e,d,h]) for selected pairs only
    
    # We need to compute this for all t, e, but only for selected pairs.
    # We can compute the full FFN for all experts and then mask, but that's inefficient.
    # However, the problem says "Unselected expert_outputs entries are zero and their FFN need not execute."
    # This implies we should only compute for selected pairs.
    
    # Let's compute the FFN for all experts first, then mask.
    # x: [T, D]
    # wi: [E, D, H]
    # We want hidden[t, e, h] = max(0, sum_d x[t, d] * wi[e, d, h])
    
    # We can do this by broadcasting x to [T, 1, D] and wi to [1, E, D, H]? No.
    # Let's reshape x to [T, 1, D] and wi to [1, E, D, H].
    # Then x * wi will be [T, E, D, H]. Sum over D gives [T, E, H].
    
    x_expanded = x.unsqueeze(1) # [T, 1, D]
    wi_expanded = wi.unsqueeze(0) # [1, E, D, H]
    
    # Multiply and sum over D
    # This creates a [T, E, D, H] tensor, which might be large.
    # Alternative: Use einsum or bmm.
    # Let's use einsum: 'td, edh -> teh'
    hidden = torch.einsum('td,edh->teh', x, wi)
    hidden = torch.relu(hidden)
    
    # Now compute expert_outputs
    # expert_outputs[t, e, o] = sum_h hidden[t, e, h] * wo[e, h, o] * scale
    # wo: [E, H, O]
    # hidden: [T, E, H]
    # We want [T, E, O]
    # Use einsum: 'teh, eho -> teo'
    expert_outputs = torch.einsum('teh,e ho->teo', hidden, wo)
    
    # Apply scaling
    scale_factor = (H / D) ** (-0.5)
    expert_outputs = expert_outputs * scale_factor
    
    # Mask unselected pairs
    expert_outputs = expert_outputs * dispatch.unsqueeze(-1).float()
    
    # 4. Combine
    # output[t, o] = sum_e gates[t, e] * expert_outputs[t, e, o]
    # gates: [T, E]
    # expert_outputs: [T, E, O]
    # Use einsum: 'te, teo -> to'
    output = torch.einsum('te,teo->to', gates, expert_outputs)
    
    return {
        'logits': logits,
        'indices': indices,
        'gates': gates,
        'dispatch': dispatch,
        'expert_outputs': expert_outputs,
        'output': output
    }