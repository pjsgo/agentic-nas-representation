import torch

def fnet_block(x, parameters, padding_mask=None):
    B, N, D = x.shape
    
    # 1. Fourier Mixing
    # H[b,n,l] = sum_d x[b,n,d] * exp(-2*pi*i*d*l/D)
    # Z[b,k,l] = sum_n H[b,n,l] * exp(-2*pi*i*n*k/N)
    # mixed[b,k,l] = Re(Z[b,k,l])
    
    # Using torch.fft.fft2 with norm='backward' (default) on axes (1, 2)
    # Input x is real [B, N, D]. We need to transform along N (axis 1) and D (axis 2).
    # The equation defines:
    # H is DFT along D (axis 2) of x.
    # Z is DFT along N (axis 1) of H.
    # So we apply DFT along axis 2 first, then axis 1. Or just fft2 on axes (1,2).
    # torch.fft.fft2(x, dim=(1, 2)) computes:
    # Y[k, l] = sum_n sum_d x[n, d] * exp(-2*pi*i*n*k/N) * exp(-2*pi*i*d*l/D)
    # This matches Z[b,k,l] exactly.
    
    # Ensure complex128 for precision as per notation
    x_complex = x.to(torch.complex128)
    Z = torch.fft.fft2(x_complex, dim=(1, 2))
    mixed = Z.real.to(torch.float64)
    
    # 2. Residual + LayerNorm 1
    # h = LN(x + mixed, gamma1, beta1)
    # LN(z, gamma, beta): mu=mean_d(z); var=max(0, mean_d(z*z)-mu*mu); return (z-mu)*(gamma/sqrt(var+1e-12))+beta
    
    gamma1 = parameters['gamma1']
    beta1 = parameters['beta1']
    
    x_plus_mixed = x + mixed
    
    # Compute mean and variance along last axis (D)
    mu = x_plus_mixed.mean(dim=-1, keepdim=True)
    # var = max(0, mean(z^2) - mu^2)
    # Using population variance formula as specified: mean(z*z) - mu*mu
    # Note: torch.var with unbiased=False computes mean((z-mu)^2) which is mathematically equivalent to mean(z^2)-mu^2 but might have different numerical stability.
    # The spec says: var=max(0,mean_d(z*z)-mu*mu). We should follow this exactly.
    z_sq_mean = (x_plus_mixed * x_plus_mixed).mean(dim=-1, keepdim=True)
    var = z_sq_mean - mu * mu
    var = torch.clamp(var, min=0.0)
    
    h = (x_plus_mixed - mu) * (gamma1 / torch.sqrt(var + 1e-12)) + beta1
    
    # 3. FFN
    # a[b,n,f] = sum_d h[b,n,d]*w1[d,f] + b1[f]
    # g = 0.5*a*(1+tanh(sqrt(2/pi)*(a+0.044715*a^3)))
    # ff[b,n,d] = sum_f g[b,n,f]*w2[f,d] + b2[d]
    
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    
    # h is [B, N, D], w1 is [D, F]
    a = torch.matmul(h, w1) + b1
    
    # tanh_gelu activation
    # g = 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3)))
    sqrt_2_pi = (2.0 / 3.141592653589793) ** 0.5
    # Using math.pi for precision
    import math
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    
    a_cubed = a * a * a
    inner = sqrt_2_pi * (a + 0.044715 * a_cubed)
    g = 0.5 * a * (1.0 + torch.tanh(inner))
    
    # g is [B, N, F], w2 is [F, D]
    ff = torch.matmul(g, w2) + b2
    
    # 4. Residual + LayerNorm 2
    # output = LN(h + ff, gamma2, beta2)
    
    gamma2 = parameters['gamma2']
    beta2 = parameters['beta2']
    
    h_plus_ff = h + ff
    
    mu2 = h_plus_ff.mean(dim=-1, keepdim=True)
    z_sq_mean2 = (h_plus_ff * h_plus_ff).mean(dim=-1, keepdim=True)
    var2 = z_sq_mean2 - mu2 * mu2
    var2 = torch.clamp(var2, min=0.0)
    
    output = (h_plus_ff - mu2) * (gamma2 / torch.sqrt(var2 + 1e-12)) + beta2
    
    return {
        'mixed': mixed,
        'output': output
    }