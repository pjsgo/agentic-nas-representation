import torch
import torch.nn as nn
import math


class Model(nn.Module):
    def __init__(self, D=2, l_max=9, order=2, width=4):
        super().__init__()
        # Validate configuration
        if type(D) is not int or D < 1 or D > 8:
            raise ValueError("D must be an integer between 1 and 8")
        if type(l_max) is not int or l_max < 2 or l_max > 32:
            raise ValueError("l_max must be an integer between 2 and 32")
        if type(order) is not int or order not in [2, 3]:
            raise ValueError("order must be 2 or 3")
        if type(width) is not int or width < 1 or width > 16:
            raise ValueError("width must be an integer between 1 and 16")

        self.D = D
        self.l_max = l_max
        self.order = order
        self.width = width
        self.C = (order + 1) * D
        self.F = (order - 1) * D

        # Parameters
        self.W_in = nn.Parameter(torch.zeros(self.C, D, dtype=torch.float64))
        self.b_in = nn.Parameter(torch.zeros(self.C, dtype=torch.float64))
        self.W_short = nn.Parameter(torch.zeros(self.C, 1, 3, dtype=torch.float64))
        self.b_short = nn.Parameter(torch.zeros(self.C, dtype=torch.float64))
        self.W0 = nn.Parameter(torch.zeros(self.width, 5, dtype=torch.float64))
        self.b0 = nn.Parameter(torch.zeros(self.width, dtype=torch.float64))
        self.W1 = nn.Parameter(torch.zeros(self.width, self.width, dtype=torch.float64))
        self.b1 = nn.Parameter(torch.zeros(self.width, dtype=torch.float64))
        self.W2 = nn.Parameter(torch.zeros(self.width, self.width, dtype=torch.float64))
        self.b2 = nn.Parameter(torch.zeros(self.width, dtype=torch.float64))
        self.freq = nn.Parameter(torch.ones(1, self.width, dtype=torch.float64))
        self.Wf = nn.Parameter(torch.zeros(self.F, self.width, dtype=torch.float64))
        self.Dskip = nn.Parameter(torch.zeros(self.F, dtype=torch.float64))
        self.W_out = nn.Parameter(torch.zeros(D, D, dtype=torch.float64))
        self.b_out = nn.Parameter(torch.zeros(D, dtype=torch.float64))

        # Buffers
        pos = torch.zeros(l_max, 5, dtype=torch.float64)
        time = torch.zeros(l_max, dtype=torch.float64)
        for s in range(l_max):
            time[s] = s / (l_max - 1)
            omega = 2 * math.pi * s / l_max
            f0 = 0.0001
            f1 = 1.0
            pos[s, 0] = time[s]
            pos[s, 1] = math.cos(f0 * omega)
            pos[s, 2] = math.cos(f1 * omega)
            pos[s, 3] = -math.sin(f0 * omega)
            pos[s, 4] = -math.sin(f1 * omega)
        self.register_buffer('pos', pos, persistent=True)
        self.register_buffer('time', time, persistent=True)

        # Decays
        a = math.log(0.01) / 1.5
        b = math.log(0.01) / 0.3
        if self.F == 1:
            deltas = torch.tensor([a], dtype=torch.float64)
        else:
            deltas = torch.zeros(self.F, dtype=torch.float64)
            for j in range(self.F):
                deltas[j] = a + (b - a) * j / (self.F - 1)
        self.register_buffer('deltas', deltas, persistent=True)

    def forward(self, u):
        # Validate input
        if not isinstance(u, torch.Tensor):
            raise ValueError("Input u must be a torch.Tensor")
        if u.dtype != torch.float64:
            raise ValueError("Input u must be float64")
        if u.device.type != 'cpu':
            raise ValueError("Input u must be on CPU")
        if u.dim() != 3:
            raise ValueError("Input u must have shape [B, L, D]")
        B, L, D_in = u.shape
        if D_in != self.D:
            raise ValueError(f"Input u last dimension must be {self.D}, got {D_in}")
        if B < 1:
            raise ValueError("Batch size B must be >= 1")
        if L < 1 or L > self.l_max:
            raise ValueError(f"Sequence length L must be between 1 and {self.l_max}")
        if not torch.isfinite(u).all():
            raise ValueError("Input u must contain only finite values")

        # Input projection
        # p = linear(u, W_in, b_in) -> [B, L, C]
        p = u @ self.W_in.t() + self.b_in
        # Transpose to [B, C, L]
        p = p.transpose(1, 2)

        # Short filter
        # z[b,c,t] = b_short[c] + sum_{r=0}^2 W_short[c,0,r] * p[b,c,t+r-2]
        # Equivalent to groups=C Conv1d, padding=2, then first L outputs
        # p is [B, C, L], W_short is [C, 1, 3]
        # We need to compute causal convolution with kernel size 3, padding 2 (left padding)
        # p_padded: [B, C, L+2] with 2 zeros at the beginning
        p_padded = torch.cat([torch.zeros(B, self.C, 2, dtype=torch.float64, device=u.device), p], dim=2)
        # Manual convolution: z[b,c,t] = sum_{r=0}^2 W_short[c,0,r] * p_padded[b,c,t+r]
        # Wait, the formula says p[b,c,t+r-2]. Let's re-read.
        # z[b,c,t] = b_short[c] + sum_{r=0}^2 W_short[c,0,r]*p[b,c,t+r-2]
        # For t=0: r=0 -> p[-2]=0, r=1 -> p[-1]=0, r=2 -> p[0]
        # For t=1: r=0 -> p[-1]=0, r=1 -> p[0], r=2 -> p[1]
        # For t=2: r=0 -> p[0], r=1 -> p[1], r=2 -> p[2]
        # This is a causal convolution where the kernel is applied such that the last element of the kernel aligns with the current time step.
        # Let's use F.conv1d with groups=C, padding=0, and pad the input manually.
        # Actually, standard conv1d with kernel size 3 and padding 1 gives:
        # out[t] = sum_{k=0}^2 W[k] * in[t-1+k]
        # We want out[t] = sum_{r=0}^2 W[r] * in[t+r-2]
        # Let k = r-2. Then r=k+2. out[t] = sum_{k=-2}^0 W[k+2] * in[t+k]
        # This is equivalent to convolving with a reversed kernel and appropriate padding.
        # Let's just do it manually to be safe.
        # Reset z
        z = torch.zeros(B, self.C, L, dtype=torch.float64, device=u.device)
        for r in range(3):
            # p_padded[:, :, r:r+L] gives p[b,c,0..L-1] shifted by r
            # We want p[b,c,t+r-2] for t=0..L-1
            # This is p_padded[b,c,r:r+L] if we define p_padded such that p_padded[i] = p[i-2] for i>=2, else 0.
            # So p_padded has length L+2. p_padded[0]=0, p_padded[1]=0, p_padded[2]=p[0], ..., p_padded[L+1]=p[L-1]
            # Then p[b,c,t+r-2] = p_padded[b,c,t+r]
            # So for a fixed r, the vector of values for t=0..L-1 is p_padded[:, :, r:r+L]
            z += self.W_short[None, :, 0, r, None] * p_padded[:, :, r:r+L]
        z += self.b_short[None, :, None]

        # Split z along C into order+1 consecutive D-channel groups x[0]..x[order-1], v
        # z is [B, C, L] = [B, (order+1)*D, L]
        # x[0] is z[:, 0:D, :], x[1] is z[:, D:2D, :], ..., x[order-1] is z[:, (order-1)*D:order*D, :], v is z[:, order*D:(order+1)*D, :]
        x = []
        for i in range(self.order):
            x.append(z[:, i * self.D:(i + 1) * self.D, :])
        v = z[:, self.order * self.D:(self.order + 1) * self.D, :]

        # Filter MLP
        # pos[:L] is [L, 5]
        pos_L = self.pos[:L]
        # a0 = sin(freq * linear(pos_L, W0, b0))
        # linear(pos_L, W0, b0) = pos_L @ W0.t() + b0 -> [L, width]
        h0 = pos_L @ self.W0.t() + self.b0
        a0 = torch.sin(self.freq * h0) # freq is [1, width], broadcasts to [L, width]
        h1 = a0 @ self.W1.t() + self.b1
        a1 = torch.sin(self.freq * h1)
        h2 = a1 @ self.W2.t() + self.b2
        a2 = torch.sin(self.freq * h2)
        # raw = a2 @ Wf.t() -> [L, F]
        raw = a2 @ self.Wf.t()

        # Modulation
        # h[s,j] = raw[s,j] * (exp(-time[s]*abs(deltas[j])) + 0.05)
        # time is [L], deltas is [F]
        # time_exp = exp(-time[:, None] * abs(deltas)[None, :]) -> [L, F]
        time_exp = torch.exp(-self.time[:L, None] * torch.abs(self.deltas)[None, :])
        h = raw * (time_exp + 0.05)
        # kernel[o,d,s] = h[s, o*D+d]
        # h is [L, F] = [L, (order-1)*D]
        # We need kernel to be [order-1, D, L]
        # Reshape h to [L, order-1, D], then transpose to [order-1, D, L]
        kernel = h.reshape(L, self.order - 1, self.D).permute(1, 2, 0)

        # Gated Recurrence
        # Starting v from split z, for o=0..order-2 in ascending order:
        # a = v * x[order-1-o]
        # v = conv(a, kernel[o]) + Dskip[o*D:(o+1)*D][None,:,None] * a
        # long_outputs[o] = v
        # After loop q = v * x[0]
        long_outputs = []
        for o in range(self.order - 1):
            a = v * x[self.order - 1 - o]
            # conv(a, kernel[o])
            # a is [B, D, L], kernel[o] is [D, L]
            # conv(a,k)[b,d,t] = sum_{s=0}^t k[d,s]*a[b,d,t-s]
            # Implement via real FFT with n=2*L, zero padding, rfft norm=backward on both,
            # multiply transforms, irfft norm=backward, keep first L. No extra scale.
            n = 2 * L
            a_fft = torch.fft.rfft(a, n=n, norm='backward') # [B, D, L]
            k_fft = torch.fft.rfft(kernel[o].unsqueeze(0), n=n, norm='backward') # [1, D, L]
            conv_result = torch.fft.irfft(a_fft * k_fft, n=n, norm='backward')[:, :, :L]
            # Diagonal skip
            skip = self.Dskip[o * self.D:(o + 1) * self.D][None, :, None] * a
            v = conv_result + skip
            long_outputs.append(v)
        q = v * x[0]

        # Output
        # Transpose q from [B, D, L] to [B, L, D]
        q = q.transpose(1, 2)
        # y = linear(q, W_out, b_out)
        y = q @ self.W_out.t() + self.b_out

        return {
            'y': y,
            'kernel': kernel,
            'long_outputs': torch.stack(long_outputs, dim=0) # [order-1, B, D, L]
        }