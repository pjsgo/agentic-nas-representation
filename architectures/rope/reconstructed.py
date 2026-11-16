import torch

def rope_attention(q, k, v, q_positions, k_positions):
    B, Q, H, D = q.shape
    _, K, _, V = v.shape
    
    # 1. Compute frequencies
    # omega[j] = 10000 ** (-2*j/D) for j=0..D/2-1
    j = torch.arange(D // 2, dtype=torch.float64, device=q.device)
    omega = 10000.0 ** (-2.0 * j / D)
    
    # 2. Compute angles
    # aq[t, j] = q_positions[t] * omega[j]
    # q_positions is [Q], omega is [D/2]
    # Result aq is [Q, D/2]
    aq = q_positions.to(torch.float64).unsqueeze(1) * omega.unsqueeze(0)
    
    # ak[u, j] = k_positions[u] * omega[j]
    # k_positions is [K], omega is [D/2]
    # Result ak is [K, D/2]
    ak = k_positions.to(torch.float64).unsqueeze(1) * omega.unsqueeze(0)
    
    # 3. Rotate Q and K
    # q_rot[b, t, h, 2*j] = q[b, t, h, 2*j]*cos(aq[t, j]) - q[b, t, h, 2*j+1]*sin(aq[t, j])
    # q_rot[b, t, h, 2*j+1] = q[b, t, h, 2*j]*sin(aq[t, j]) + q[b, t, h, 2*j+1]*cos(aq[t, j])
    
    # Reshape q to [B, Q, H, D/2, 2] to handle pairs
    q_pairs = q.view(B, Q, H, D // 2, 2)
    q_even = q_pairs[..., 0] # [B, Q, H, D/2]
    q_odd = q_pairs[..., 1]  # [B, Q, H, D/2]
    
    # Expand angles to match batch and head dimensions
    # aq is [Q, D/2], need [B, Q, H, D/2]
    cos_aq = torch.cos(aq).unsqueeze(0).unsqueeze(2).expand(B, Q, H, D // 2)
    sin_aq = torch.sin(aq).unsqueeze(0).unsqueeze(2).expand(B, Q, H, D // 2)
    
    q_rot_even = q_even * cos_aq - q_odd * sin_aq
    q_rot_odd = q_even * sin_aq + q_odd * cos_aq
    
    # Interleave back to [B, Q, H, D]
    q_rot = torch.stack([q_rot_even, q_rot_odd], dim=-1).view(B, Q, H, D)
    
    # Same for K
    k_pairs = k.view(B, K, H, D // 2, 2)
    k_even = k_pairs[..., 0] # [B, K, H, D/2]
    k_odd = k_pairs[..., 1]  # [B, K, H, D/2]
    
    cos_ak = torch.cos(ak).unsqueeze(0).unsqueeze(2).expand(B, K, H, D // 2)
    sin_ak = torch.sin(ak).unsqueeze(0).unsqueeze(2).expand(B, K, H, D // 2)
    
    k_rot_even = k_even * cos_ak - k_odd * sin_ak
    k_rot_odd = k_even * sin_ak + k_odd * cos_ak
    
    k_rot = torch.stack([k_rot_even, k_rot_odd], dim=-1).view(B, K, H, D)
    
    # 4. Compute scaled scores
    # scores[b, h, t, u] = sum_d(q_rot[b, t, h, d] * k_rot[b, u, h, d]) / sqrt(D)
    
    # q_rot: [B, Q, H, D] -> [B, H, Q, D]
    q_rot_perm = q_rot.permute(0, 2, 1, 3)
    # k_rot: [B, K, H, D] -> [B, H, K, D]
    k_rot_perm = k_rot.permute(0, 2, 1, 3)
    
    # Batched matrix multiplication: [B, H, Q, D] @ [B, H, D, K] -> [B, H, Q, K]
    scores = torch.matmul(q_rot_perm, k_rot_perm.transpose(-2, -1))
    scores = scores / (D ** 0.5)
    
    # 5. Stable softmax
    # Subtract max, exp, sum, divide
    max_scores = scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - max_scores)
    sum_exp_scores = exp_scores.sum(dim=-1, keepdim=True)
    probabilities = exp_scores / sum_exp_scores
    
    # 6. Weighted sum of values
    # output[b, t, h, e] = sum_u(probabilities[b, h, t, u] * v[b, u, h, e])
    
    # probabilities: [B, H, Q, K]
    # v: [B, K, H, V] -> [B, H, K, V]
    v_perm = v.permute(0, 2, 1, 3)
    
    # Batched matrix multiplication: [B, H, Q, K] @ [B, H, K, V] -> [B, H, Q, V]
    output = torch.matmul(probabilities, v_perm)
    
    # Permute back to [B, Q, H, V]
    output = output.permute(0, 2, 1, 3)
    
    return {
        'q_rot': q_rot,
        'k_rot': k_rot,
        'scores': scores,
        'probabilities': probabilities,
        'output': output
    }