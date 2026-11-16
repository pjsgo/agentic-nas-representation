import torch

def pre_ln(x, params, padding, heads, eps=1e-5):
    B, T, E = x.shape
    H = heads
    D = E // H
    
    # 1. Pre-attention LayerNorm
    # LN(z,g,b) = (z - mean) / sqrt(var + eps) * g + b
    mean1 = x.mean(dim=-1, keepdim=True)
    var1 = x.var(dim=-1, unbiased=False, keepdim=True)
    u = (x - mean1) / torch.sqrt(var1 + eps) * params['norm1_weight'] + params['norm1_bias']
    
    # 2. Multihead Self-Attention
    # Linear projections
    # linear(z,W,b)[...,o] = sum_i z[...,i]*W[o,i] + b[o]
    # x: [B, T, E], W: [E, E] -> out: [B, T, E]
    q0 = torch.matmul(u, params['q_weight'].t()) + params['q_bias']
    k0 = torch.matmul(u, params['k_weight'].t()) + params['k_bias']
    v0 = torch.matmul(u, params['v_weight'].t()) + params['v_bias']
    
    # Reshape to [B, H, T, D]
    # q0[b, t, h*D+d] -> q[b, h, t, d]
    q = q0.view(B, T, H, D).permute(0, 2, 1, 3)
    k = k0.view(B, T, H, D).permute(0, 2, 1, 3)
    v = v0.view(B, T, H, D).permute(0, 2, 1, 3)
    
    # Compute logits
    # logit[b, h, i, j] = sum_d q[b, h, i, d] * k[b, h, j, d] / sqrt(D)
    # q: [B, H, T, D], k: [B, H, T, D]
    # q @ k^T -> [B, H, T, T]
    scale = torch.sqrt(torch.tensor(D, dtype=torch.float64, device=x.device))
    logits = torch.matmul(q, k.transpose(-2, -1)) / scale
    
    # Masking
    # padding: [B, T], True excludes a KEY for every query and head.
    # masked[b, h, i, j] = -inf if padding[b, j] else logits[b, h, i, j]
    # padding shape [B, T] needs to be broadcast to [B, H, T, T] (query, key)
    # We mask the key dimension (last dim).
    # padding: [B, T] -> [B, 1, 1, T]
    mask = padding.unsqueeze(1).unsqueeze(1) # [B, 1, 1, T]
    # Expand to [B, H, T, T]
    # We use where to set -inf where mask is True
    # Note: padding is bool. True means exclude.
    logits = torch.where(mask, torch.tensor(float('-inf'), dtype=torch.float64, device=x.device), logits)
    
    # Softmax
    # probabilities = exp(masked - max_j(masked)) / sum_j exp(masked - max_j(masked))
    # max over key axis (dim -1)
    max_logits = logits.max(dim=-1, keepdim=True).values
    # Handle case where all keys are masked? Constraint says "each batch has at least one nonpadding key".
    # So max is finite.
    exp_logits = torch.exp(logits - max_logits)
    # Sum over key axis
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    probabilities = exp_logits / sum_exp
    
    # Weighted sum of values
    # merged[b, i, h*D+d] = sum_j probabilities[b, h, i, j] * v[b, h, j, d]
    # probabilities: [B, H, T, T], v: [B, H, T, D]
    # matmul: [B, H, T, T] @ [B, H, T, D] -> [B, H, T, D]
    attn_output = torch.matmul(probabilities, v)
    
    # Reshape back to [B, T, E]
    # attn_output: [B, H, T, D] -> [B, T, H, D] -> [B, T, E]
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, E)
    
    # Output projection
    a = torch.matmul(attn_output, params['out_weight'].t()) + params['out_bias']
    
    # 3. Attention Residual
    r = x + a
    after_attention = r
    
    # 4. Pre-FFN LayerNorm
    mean2 = r.mean(dim=-1, keepdim=True)
    var2 = r.var(dim=-1, unbiased=False, keepdim=True)
    w = (r - mean2) / torch.sqrt(var2 + eps) * params['norm2_weight'] + params['norm2_bias']
    
    # 5. FFN (Two Linear ReLU)
    # f = linear(max(0, linear(w, fc1_weight, fc1_bias)), fc2_weight, fc2_bias)
    # w: [B, T, E], fc1_weight: [F, E] -> [B, T, F]
    fc1_out = torch.matmul(w, params['fc1_weight'].t()) + params['fc1_bias']
    fc1_act = torch.relu(fc1_out)
    # fc1_act: [B, T, F], fc2_weight: [E, F] -> [B, T, E]
    f = torch.matmul(fc1_act, params['fc2_weight'].t()) + params['fc2_bias']
    
    # 6. FFN Residual
    output = r + f
    
    return {
        'output': output,
        'after_attention': after_attention,
        'probabilities': probabilities
    }