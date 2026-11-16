import torch

def dense_synthesizer(x, w1, w2, wv, wo, allowed):
    # Validate inputs
    if not isinstance(x, torch.Tensor) or not isinstance(w1, torch.Tensor) or not isinstance(w2, torch.Tensor) or not isinstance(wv, torch.Tensor) or not isinstance(wo, torch.Tensor) or not isinstance(allowed, torch.Tensor):
        raise ValueError("All inputs must be torch.Tensor")
    
    # Check dtypes and devices
    float_tensors = [x, w1, w2, wv, wo]
    for t in float_tensors:
        if t.dtype != torch.float64:
            raise ValueError(f"Expected float64 tensor, got {t.dtype}")
        if t.device.type != 'cpu':
            raise ValueError(f"Expected CPU tensor, got {t.device}")
        if not torch.isfinite(t).all():
            raise ValueError("Non-finite values in floating input")
    
    if allowed.dtype != torch.bool:
        raise ValueError(f"Expected bool tensor for allowed, got {allowed.dtype}")
    if allowed.device.type != 'cpu':
        raise ValueError(f"Expected CPU tensor for allowed, got {allowed.device}")
    
    # Check shapes
    if x.dim() != 3:
        raise ValueError(f"x must be 3D, got {x.dim()}D")
    B, N, M = x.shape
    
    if w1.dim() != 3:
        raise ValueError(f"w1 must be 3D, got {w1.dim()}D")
    M1, H, D = w1.shape
    if M1 != M:
        raise ValueError(f"w1 first dim {M1} must match x last dim {M}")
    
    if w2.dim() != 2:
        raise ValueError(f"w2 must be 2D, got {w2.dim()}D")
    D2, L = w2.shape
    if D2 != D:
        raise ValueError(f"w2 first dim {D2} must match w1 last dim {D}")
    
    if wv.dim() != 3:
        raise ValueError(f"wv must be 3D, got {wv.dim()}D")
    Mv, Hv, Dv = wv.shape
    if Mv != M or Hv != H or Dv != D:
        raise ValueError(f"wv shape {wv.shape} must match [M,H,D] = [{M},{H},{D}]")
    
    if wo.dim() != 3:
        raise ValueError(f"wo must be 3D, got {wo.dim()}D")
    Ho, Do, Mo = wo.shape
    if Ho != H or Do != D or Mo != M:
        raise ValueError(f"wo shape {wo.shape} must match [H,D,M] = [{H},{D},{M}]")
    
    if allowed.dim() != 3:
        raise ValueError(f"allowed must be 3D, got {allowed.dim()}D")
    Ba, Na, Ma = allowed.shape
    if Ba != B or Na != N or Ma != N:
        raise ValueError(f"allowed shape {allowed.shape} must match [B,N,N] = [{B},{N},{N}]")
    
    # Check positive integers
    if B <= 0 or N <= 0 or M <= 0 or H <= 0 or D <= 0 or L <= 0:
        raise ValueError("All dimensions must be positive integers")
    if H < 2:
        raise ValueError("H must be >= 2")
    if N > L:
        raise ValueError(f"N={N} must be <= L={L}")
    
    # Check at least one True per row in allowed
    row_sums = allowed.sum(dim=-1)
    if (row_sums == 0).any():
        raise ValueError("Each row of allowed must have at least one True")
    
    # Compute logits: [B, H, N, N]
    # hidden[b,h,i,d] = max(0, sum_m x[b,i,m] * w1[m,h,d])
    # x: [B, N, M], w1: [M, H, D]
    # x.unsqueeze(1): [B, N, 1, M]
    # w1.unsqueeze(0): [1, M, H, D] -> permute to [1, H, M, D] then unsqueeze for N
    # Actually: x[b,i,m] * w1[m,h,d] -> sum over m
    # x: [B, N, M] -> [B, N, 1, M]
    # w1: [M, H, D] -> [1, 1, M, H, D] -> permute to [1, 1, H, M, D]
    # Better: use einsum or matmul
    # x: [B, N, M], w1: [M, H, D]
    # We want hidden[b,h,i,d] = sum_m x[b,i,m] * w1[m,h,d]
    # x: [B, N, M] -> [B, N, 1, M]
    # w1: [M, H, D] -> [1, 1, M, H, D] -> permute(0, 3, 1, 2, 4) -> [1, H, M, D] -> unsqueeze(1) -> [1, H, 1, M, D]
    # Then x.unsqueeze(1).unsqueeze(2) * w1_expanded -> [B, N, H, M, D] -> sum over M -> [B, N, H, D]
    # Then permute to [B, H, N, D]
    
    # Simpler: use einsum
    # hidden[b,h,i,d] = sum_m x[b,i,m] * w1[m,h,d]
    hidden = torch.einsum('bim,mhd->bhid', x, w1)
    hidden = torch.relu(hidden)
    
    # full_logits[b,h,i,ell] = sum_d hidden[b,h,i,d] * w2[d,ell] for 0<=ell<L
    # hidden: [B, H, N, D], w2: [D, L]
    full_logits = torch.einsum('bhid,de->bhi e', hidden, w2)
    # Actually: 'bhid,de->bhi e' has a space, let's fix
    full_logits = torch.einsum('bhid,de->bhi e', hidden, w2) if False else torch.einsum('bhid,de->bhi e', hidden, w2)
    # Let me just use matmul
    # hidden: [B, H, N, D], w2: [D, L]
    # full_logits[b,h,i,ell] = sum_d hidden[b,h,i,d] * w2[d,ell]
    full_logits = torch.matmul(hidden, w2)  # [B, H, N, L]
    
    # logits[b,h,i,j] = full_logits[b,h,i,j] for 0<=j<N
    logits = full_logits[:, :, :, :N]  # [B, H, N, N]
    
    # Compute values: [B, H, N, D]
    # values[b,h,j,d] = sum_m x[b,j,m] * wv[m,h,d]
    values = torch.einsum('bjm,mhd->bhjd', x, wv)
    
    # Normalize: mask then stable key softmax
    # allowed: [B, N, N] -> [B, 1, N, N] for broadcasting over H
    allowed_expanded = allowed.unsqueeze(1)  # [B, 1, N, N]
    
    # masked[b,h,i,j] = logits[b,h,i,j] if allowed[b,i,j] else -inf
    masked = torch.where(allowed_expanded, logits, torch.tensor(float('-inf'), dtype=torch.float64, device='cpu'))
    
    # probabilities = exp(masked - max_j(masked)) / sum_j(exp(masked - max_j(masked)))
    # max over j (last dim)
    max_masked = masked.max(dim=-1, keepdim=True).values  # [B, H, N, 1]
    
    # Handle case where all are -inf (should not happen due to validation, but be safe)
    # exp(-inf - (-inf)) = exp(nan), so we need to handle this
    # Since we validated at least one True per row, max_masked should be finite
    exp_masked = torch.exp(masked - max_masked)  # [B, H, N, N]
    sum_exp = exp_masked.sum(dim=-1, keepdim=True)  # [B, H, N, 1]
    probabilities = exp_masked / sum_exp  # [B, H, N, N]
    
    # Set forbidden entries to exactly zero
    probabilities = torch.where(allowed_expanded, probabilities, torch.tensor(0.0, dtype=torch.float64, device='cpu'))
    
    # Output: aggregate values and project heads
    # context[b,h,i,d] = sum_j probabilities[b,h,i,j] * values[b,h,j,d]
    # probabilities: [B, H, N, N], values: [B, H, N, D]
    context = torch.einsum('bhij,bhjd->bhid', probabilities, values)
    
    # output[b,i,m] = sum_h sum_d context[b,h,i,d] * wo[h,d,m]
    # context: [B, H, N, D], wo: [H, D, M]
    # output[b,i,m] = sum_h sum_d context[b,h,i,d] * wo[h,d,m]
    output = torch.einsum('bhid,hdm->bim', context, wo)
    
    return {
        'output': output,
        'logits': logits,
        'probabilities': probabilities
    }
