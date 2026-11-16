import torch

def rmsnorm(x, scale, eps=1e-8):
    """
    Compute RMSNorm on the last axis of x.
    
    Args:
        x: Tensor of shape [..., D], float64
        scale: Tensor of shape [D], float64
        eps: float, default 1e-8
    
    Returns:
        dict with keys 'output' and 'rms'
    """
    # Ensure float64
    x = x.to(torch.float64)
    scale = scale.to(torch.float64)
    
    # Compute RMS: sqrt(sum(x^2) / D)
    # sum over last axis, keepdim=True to get [..., 1]
    sum_sq = torch.sum(x * x, dim=-1, keepdim=True)
    D = x.shape[-1]
    rms = torch.sqrt(sum_sq / D)
    
    # Compute output: scale * (x / (rms + eps))
    # rms is [..., 1], x is [..., D], scale is [D]
    # Broadcasting: x / (rms + eps) -> [..., D]
    # scale * (...) -> [..., D]
    output = scale * (x / (rms + eps))
    
    return {
        'output': output,
        'rms': rms
    }