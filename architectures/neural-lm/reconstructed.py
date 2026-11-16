import torch


def neural_lm(tokens, C, H, d, U, W, b):
    # Validate inputs
    tensors = [tokens, C, H, d, U, W, b]
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise ValueError("All inputs must be torch.Tensor")

    # Check device
    for t in tensors:
        if t.device.type != 'cpu':
            raise ValueError("All tensors must be on CPU")

    # Check dtypes
    if tokens.dtype != torch.int64:
        raise ValueError("tokens must be int64")
    float_tensors = [C, H, d, U, W, b]
    for t in float_tensors:
        if t.dtype != torch.float64:
            raise ValueError("All weight tensors must be float64")

    # Check ranks
    if tokens.dim() != 2:
        raise ValueError("tokens must be 2D [B, K]")
    if C.dim() != 2:
        raise ValueError("C must be 2D [V, E]")
    if H.dim() != 2:
        raise ValueError("H must be 2D [D, Q]")
    if d.dim() != 1:
        raise ValueError("d must be 1D [D]")
    if U.dim() != 2:
        raise ValueError("U must be 2D [V, D]")
    if W.dim() != 2:
        raise ValueError("W must be 2D [V, Q]")
    if b.dim() != 1:
        raise ValueError("b must be 1D [V]")

    B, K = tokens.shape
    V, E = C.shape
    D, Q = H.shape

    # Check dimensions are positive
    if B <= 0 or K <= 0 or V <= 0 or E <= 0 or D <= 0 or Q <= 0:
        raise ValueError("All dimensions must be positive integers")

    # Check Q = K * E
    if Q != K * E:
        raise ValueError(f"Q must equal K*E, got Q={Q}, K={K}, E={E}")

    # Check shape consistency
    if d.shape[0] != D:
        raise ValueError("d shape mismatch")
    if U.shape[0] != V or U.shape[1] != D:
        raise ValueError("U shape mismatch")
    if W.shape[0] != V or W.shape[1] != Q:
        raise ValueError("W shape mismatch")
    if b.shape[0] != V:
        raise ValueError("b shape mismatch")

    # Check finite values
    for t in float_tensors:
        if not torch.isfinite(t).all():
            raise ValueError("All floating point inputs must be finite")

    # Check token range
    if tokens.min() < 0 or tokens.max() >= V:
        raise ValueError("Token IDs out of range")

    # neural_lm_lookup: embedding[a,j,e] = C[tokens[a,j],e]
    embedding = C[tokens]  # [B, K, E]

    # neural_lm_context: features[a, j*E+e] = embedding[a,j,e]
    features = embedding.reshape(B, K * E)  # [B, Q]

    # neural_lm_hidden: hidden[a,h] = tanh(d[h] + sum_q H[h,q]*features[a,q])
    # H is [D, Q], features is [B, Q]
    # We need: hidden = tanh(d + features @ H.T)
    hidden = torch.tanh(d + features @ H.T)  # [B, D]

    # neural_lm_direct: direct[a,v] = sum_q W[v,q]*features[a,q]
    # W is [V, Q], features is [B, Q]
    # direct = features @ W.T
    direct = features @ W.T  # [B, V]

    # neural_lm_readout: logits[a,v] = b[v] + direct[a,v] + sum_h U[v,h]*hidden[a,h]
    # U is [V, D], hidden is [B, D]
    # hidden_contrib = hidden @ U.T  # [B, V]
    hidden_contrib = hidden @ U.T  # [B, V]
    logits = b + direct + hidden_contrib  # [B, V]

    # neural_lm_softmax: stable softmax
    m = logits.max(dim=1, keepdim=True).values  # [B, 1]
    exp_logits = torch.exp(logits - m)  # [B, V]
    probabilities = exp_logits / exp_logits.sum(dim=1, keepdim=True)  # [B, V]

    return {
        'features': features,
        'hidden': hidden,
        'logits': logits,
        'probabilities': probabilities
    }
