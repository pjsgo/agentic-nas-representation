import torch

def ttt_linear(x, parameters, chunk_size, base_lr=1.0, state=None):
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if x.dim() != 3:
        raise ValueError("x must be 3-dimensional")
    B, T, D = x.shape
    if B <= 0 or T <= 0 or D <= 0:
        raise ValueError("x dimensions must be positive")
    if T % chunk_size != 0:
        raise ValueError("T must be divisible by chunk_size")
    if not torch.isfinite(x).all():
        raise ValueError("x must be finite")
    if x.dtype != torch.float64:
        raise ValueError("x must be float64")
    if x.device.type != 'cpu':
        raise ValueError("x must be on CPU")
    
    required_keys = ['q', 'k', 'v', 'W0', 'b0', 'gamma', 'beta', 'lr_weight', 'lr_bias', 'token_offset']
    for key in required_keys:
        if key not in parameters:
            raise ValueError(f"Missing parameter: {key}")
    
    q = parameters['q']
    k = parameters['k']
    v = parameters['v']
    W0 = parameters['W0']
    b0 = parameters['b0']
    gamma = parameters['gamma']
    beta = parameters['beta']
    lr_weight = parameters['lr_weight']
    lr_bias = parameters['lr_bias']
    token_offset = parameters['token_offset']
    
    if q.shape != (D, H) if 'H' in dir() else True:
        pass
    
    H = q.shape[1]
    if H < 2:
        raise ValueError("H must be >= 2")
    
    if q.shape != (D, H):
        raise ValueError("q shape mismatch")
    if k.shape != (D, H):
        raise ValueError("k shape mismatch")
    if v.shape != (D, H):
        raise ValueError("v shape mismatch")
    if W0.shape != (H, H):
        raise ValueError("W0 shape mismatch")
    if b0.shape != (H,):
        raise ValueError("b0 shape mismatch")
    if gamma.shape != (H,):
        raise ValueError("gamma shape mismatch")
    if beta.shape != (H,):
        raise ValueError("beta shape mismatch")
    if lr_weight.shape != (D,):
        raise ValueError("lr_weight shape mismatch")
    if lr_bias.shape != ():
        raise ValueError("lr_bias shape mismatch")
    if token_offset.shape != (chunk_size,):
        raise ValueError("token_offset shape mismatch")
    
    for name, tensor in [('q', q), ('k', k), ('v', v), ('W0', W0), ('b0', b0), ('gamma', gamma), ('beta', beta), ('lr_weight', lr_weight), ('lr_bias', lr_bias), ('token_offset', token_offset)]:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite")
        if tensor.dtype != torch.float64:
            raise ValueError(f"{name} must be float64")
        if tensor.device.type != 'cpu':
            raise ValueError(f"{name} must be on CPU")
    
    if state is not None:
        if not isinstance(state, dict):
            raise ValueError("state must be a dict or None")
        if 'W' not in state or 'b' not in state:
            raise ValueError("state must contain W and b")
        W_state = state['W']
        b_state = state['b']
        if W_state.shape != (B, H, H):
            raise ValueError("state W shape mismatch")
        if b_state.shape != (B, H):
            raise ValueError("state b shape mismatch")
        if not torch.isfinite(W_state).all() or not torch.isfinite(b_state).all():
            raise ValueError("state must be finite")
        if W_state.dtype != torch.float64 or b_state.dtype != torch.float64:
            raise ValueError("state must be float64")
        if W_state.device.type != 'cpu' or b_state.device.type != 'cpu':
            raise ValueError("state must be on CPU")
    
    if not torch.isfinite(torch.tensor(base_lr)):
        raise ValueError("base_lr must be finite")
    if base_lr < 0:
        raise ValueError("base_lr must be >= 0")
    
    Q = x @ q
    X = x @ k
    V = x @ v
    target = V - X
    
    lambda_rates = (base_lr / H) * torch.sigmoid(x @ lr_weight + lr_bias)
    
    alpha = torch.zeros(chunk_size, dtype=torch.float64)
    for i in range(chunk_size):
        alpha[i] = max(1.0 / (i + 1) + token_offset[i].item(), 0.0)
    
    output = torch.zeros(B, T, H, dtype=torch.float64)
    loss = torch.zeros(B, T, dtype=torch.float64)
    grad_W = torch.zeros(B, T, H, H, dtype=torch.float64)
    grad_b = torch.zeros(B, T, H, dtype=torch.float64)
    W_path = torch.zeros(B, T, H, H, dtype=torch.float64)
    b_path = torch.zeros(B, T, H, dtype=torch.float64)
    lr_out = torch.zeros(B, T, dtype=torch.float64)
    
    if state is None:
        W_current = W0.unsqueeze(0).expand(B, H, H).clone()
        b_current = b0.unsqueeze(0).expand(B, H).clone()
    else:
        W_current = state['W'].clone()
        b_current = state['b'].clone()
    
    num_chunks = T // chunk_size
    
    for c in range(num_chunks):
        start = c * chunk_size
        end = start + chunk_size
        
        Wc = W_current.clone()
        bc = b_current.clone()
        
        X_chunk = X[:, start:end, :]  # [B, K, H]
        target_chunk = target[:, start:end, :]  # [B, K, H]
        Q_chunk = Q[:, start:end, :]  # [B, K, H]
        lambda_chunk = lambda_rates[:, start:end]  # [B, K]
        
        z = torch.bmm(X_chunk, Wc) + bc.unsqueeze(1)
        # z: [B, K, H]
        
        mu = z.mean(dim=-1, keepdim=True)  # [B, K, 1]
        var = ((z - mu) ** 2).mean(dim=-1, keepdim=True)  # [B, K, 1]
        std = torch.sqrt(var + 1e-6)  # [B, K, 1]
        n = (z - mu) / std  # [B, K, H]
        
        ln_z = gamma * n + beta  # [B, K, H]
        
        loss_chunk = 0.5 * ((ln_z - target_chunk) ** 2).sum(dim=-1)  # [B, K]
        loss[:, start:end] = loss_chunk
        
        e = (ln_z - target_chunk) * gamma  # [B, K, H]
        
        mean_e = e.mean(dim=-1, keepdim=True)  # [B, K, 1]
        mean_en = (e * n).mean(dim=-1, keepdim=True)  # [B, K, 1]
        
        g = (e - mean_e - n * mean_en) / std  # [B, K, H]
        
        grad_W_chunk = X_chunk.unsqueeze(-1) * g.unsqueeze(2)  # [B, K, H, H]
        grad_b_chunk = g  # [B, K, H]
        
        grad_W[:, start:end] = grad_W_chunk
        grad_b[:, start:end] = grad_b_chunk
        lr_out[:, start:end] = lambda_chunk
        
        W_prefix = torch.zeros(B, chunk_size, H, H, dtype=torch.float64)
        b_prefix = torch.zeros(B, chunk_size, H, dtype=torch.float64)
        
        for i in range(chunk_size):
            sum_grad_W = torch.zeros(B, H, H, dtype=torch.float64)
            sum_grad_b = torch.zeros(B, H, dtype=torch.float64)
            for j in range(i + 1):
                sum_grad_W += lambda_chunk[:, j, None, None] * grad_W_chunk[:, j, :, :]  # [B, H, H]
                sum_grad_b += lambda_chunk[:, j, None] * grad_b_chunk[:, j, :]  # [B, H]
            
            Wi = Wc - alpha[i] * sum_grad_W  # [B, H, H]
            bi = bc - alpha[i] * sum_grad_b  # [B, H]
            
            W_prefix[:, i] = Wi
            b_prefix[:, i] = bi
            
            W_path[:, start + i] = Wi
            b_path[:, start + i] = bi
            
            pred = torch.einsum('bh,bhd->bd', Q_chunk[:, i, :], Wi) + bi  # [B, H]
            pred_mu = pred.mean(dim=-1, keepdim=True)  # [B, 1]
            pred_var = ((pred - pred_mu) ** 2).mean(dim=-1, keepdim=True)  # [B, 1]
            pred_std = torch.sqrt(pred_var + 1e-6)  # [B, 1]
            pred_n = (pred - pred_mu) / pred_std  # [B, H]
            pred_ln = gamma * pred_n + beta  # [B, H]
            
            output[:, start + i] = Q_chunk[:, i, :] + pred_ln
        
        W_current = W_prefix[:, -1, :, :]
        b_current = b_prefix[:, -1, :]
    
    return {
        'output': output,
        'loss': loss,
        'grad_W': grad_W,
        'grad_b': grad_b,
        'W_path': W_path,
        'b_path': b_path,
        'lr': lr_out,
        'state': {'W': W_current, 'b': b_current}
    }