import torch
import torch.nn as nn

class TransformerXLAttention(nn.Module):
    def __init__(self, spec, parameters):
        super().__init__()
        self.d_model = spec['config']['d_model']
        self.heads = spec['config']['heads']
        self.d_head = spec['config']['d_head']
        self.memory_length = spec['config']['memory_length']
        self.eps = spec['config']['eps']

        self.qkv = parameters['qkv']
        self.out = parameters['out']
        self.relative = parameters['relative']
        self.u = parameters['u']
        self.v = parameters['v']
        self.gamma = parameters['gamma']
        self.beta = parameters['beta']

    def forward(self, x, memory):
        T, B, D = x.shape
        M = memory.shape[0]
        H = self.heads
        d_head = self.d_head

        # 1. Memory Read
        if M > 0:
            mem_cat = torch.cat([memory.detach(), x], dim=0)
        else:
            mem_cat = x

        # 2. QKV Projection
        # Input: [M+T, B, D]
        # Weight: [3*H*d_head, D]
        # Output: [M+T, B, 3*H*d_head]
        qkv_out = torch.matmul(mem_cat, self.qkv.t())

        # Split into Q, K, V
        # Q: last T positions
        # K, V: all M+T positions

        # Reshape to [M+T, B, 3, H, d_head]
        qkv_out = qkv_out.view(M+T, B, 3, H, d_head)

        # Permute to [3, M+T, B, H, d_head]
        qkv_out = qkv_out.permute(2, 0, 1, 3, 4)

        Q_full, K_full, V_full = qkv_out[0], qkv_out[1], qkv_out[2]

        # Q is last T positions
        Q = Q_full[-T:]  # [T, B, H, d_head]

        # K, V are all positions
        K = K_full  # [M+T, B, H, d_head]
        V = V_full  # [M+T, B, H, d_head]

        # 3. Relative Position
        # Distance delta = M + i - j for allowed key j <= M + i
        # i in [0, T), j in [0, M+T)
        # delta ranges from M - (M+T-1) = -T+1 to M + (T-1) - 0 = M+T-1
        # We need R for all possible deltas.
        # The equation uses R[M+i-j].
        # Let's generate sinusoids for all deltas in the range.

        # Deltas: M+i-j
        # i: 0..T-1
        # j: 0..M+T-1
        # Min delta: M + 0 - (M+T-1) = -T+1
        # Max delta: M + (T-1) - 0 = M+T-1

        # We need to compute R for these deltas.
        # R is [H, d_head] per delta? No, R is [H, d_head] in the equation?
        # "relative_position.output" -> "txl.relative_score.R"
        # Equation: (Q[b,h,i]+v[h]) dot R[M+i-j,h]
        # R is indexed by delta and head h. So R is [num_deltas, H, d_head] or similar.

        # Let's compute the sinusoidal features for all deltas.
        # Deltas: -T+1 to M+T-1
        # Total deltas: (M+T-1) - (-T+1) + 1 = M + 2T - 1

        # However, the relative weight is [H*d_head, D].
        # The sinusoid is D-dimensional.
        # "sinusoid_linear" -> "relative" weight.
        # So we compute sinusoid(delta) -> [D], then matmul with relative.t() -> [H*d_head], then split heads -> [H, d_head].

        # Let's create a tensor of deltas.
        # We need R for each (i, j) pair? Or just for each unique delta?
        # The equation uses R[M+i-j].
        # We can compute R for all deltas in the range and index into it.

        # Deltas: M + i - j
        # i: 0..T-1
        # j: 0..M+T-1

        # Let's create a matrix of deltas [T, M+T]
        i_idx = torch.arange(T, dtype=torch.float64, device=x.device).view(T, 1)
        j_idx = torch.arange(M+T, dtype=torch.float64, device=x.device).view(1, M+T)
        deltas = M + i_idx - j_idx  # [T, M+T]

        # We need to compute sinusoids for these deltas.
        # But the relative weight is applied to the sinusoid.
        # So we compute sinusoid for each delta, then project.

        # To be efficient, we can compute sinusoids for all unique deltas.
        # But T and M+T are small in this context? Let's just compute for all deltas in the matrix.
        # Actually, we can compute the sinusoid for each delta in the matrix.

        # D/2 sine, D/2 cosine.
        # inv_freq: 10000**(-2*k/D), k=0..D/2-1

        D = self.d_model
        half_D = D // 2
        k = torch.arange(half_D, dtype=torch.float64, device=x.device)
        inv_freq = 10000 ** (-2 * k / D)  # [half_D]

        # deltas: [T, M+T]
        # We need to compute sin(delta * inv_freq) and cos(delta * inv_freq)
        # delta * inv_freq: [T, M+T, half_D]

        # Expand deltas to [T, M+T, 1]
        deltas_exp = deltas.unsqueeze(-1)  # [T, M+T, 1]

        # inv_freq: [1, 1, half_D]
        inv_freq_exp = inv_freq.view(1, 1, half_D)

        # Product: [T, M+T, half_D]
        prod = deltas_exp * inv_freq_exp

        # Sin and Cos
        sin_part = torch.sin(prod)  # [T, M+T, half_D]
        cos_part = torch.cos(prod)  # [T, M+T, half_D]

        # Concatenate: [T, M+T, D]
        sinusoid = torch.cat([sin_part, cos_part], dim=-1)

        # Project with relative weight: [H*d_head, D]
        # sinusoid: [T, M+T, D]
        # relative.t(): [D, H*d_head]
        # Result: [T, M+T, H*d_head]

        R_proj = torch.matmul(sinusoid, self.relative.t())  # [T, M+T, H*d_head]

        # Split heads: [T, M+T, H, d_head]
        R_proj = R_proj.view(T, M+T, H, d_head)

        # 4. Relative Score
        # score[b,h,i,j] = ((Q[b,h,i]+u[h]) dot K[b,h,j] + (Q[b,h,i]+v[h]) dot R[M+i-j,h]) / sqrt(d_head)

        # Q: [T, B, H, d_head]
        # K: [M+T, B, H, d_head]
        # R_proj: [T, M+T, H, d_head]
        # u: [H, d_head]
        # v: [H, d_head]

        # We need to compute for each b, h, i, j.

        # Let's permute to [B, H, T, M+T, d_head] or similar.

        # Q: [T, B, H, d_head] -> [B, H, T, d_head]
        Q_perm = Q.permute(1, 2, 0, 3)  # [B, H, T, d_head]

        # K: [M+T, B, H, d_head] -> [B, H, M+T, d_head]
        K_perm = K.permute(1, 2, 0, 3)  # [B, H, M+T, d_head]

        # R_proj: [T, M+T, H, d_head] -> [H, T, M+T, d_head]
        R_perm = R_proj.permute(2, 0, 1, 3)  # [H, T, M+T, d_head]

        # u: [H, d_head] -> [1, H, 1, d_head]
        u_exp = self.u.view(1, H, 1, d_head)

        # v: [H, d_head] -> [1, H, 1, d_head]
        v_exp = self.v.view(1, H, 1, d_head)

        # Q + u: [B, H, T, d_head] + [1, H, 1, d_head] -> [B, H, T, d_head]
        Q_u = Q_perm + u_exp

        # Q + v: [B, H, T, d_head] + [1, H, 1, d_head] -> [B, H, T, d_head]
        Q_v = Q_perm + v_exp

        # Term 1: (Q+u) dot K
        # Q_u: [B, H, T, d_head]
        # K_perm: [B, H, M+T, d_head]
        # We need dot product over d_head for each i, j.
        # Result: [B, H, T, M+T]

        # Use einsum or matmul.
        # Q_u: [B, H, T, d_head]
        # K_perm: [B, H, M+T, d_head]
        # We can do: Q_u @ K_perm^T
        # Q_u: [B, H, T, d_head]
        # K_perm^T: [B, H, d_head, M+T]
        # Result: [B, H, T, M+T]

        term1 = torch.matmul(Q_u, K_perm.transpose(-1, -2))  # [B, H, T, M+T]

        # Term 2: (Q+v) dot R
        # Q_v: [B, H, T, d_head]
        # R_perm: [H, T, M+T, d_head]
        # We need to align B and H.
        # R_perm: [H, T, M+T, d_head] -> [1, H, T, M+T, d_head]
        R_exp = R_perm.unsqueeze(0)  # [1, H, T, M+T, d_head]

        # Q_v: [B, H, T, d_head] -> [B, H, T, 1, d_head]
        Q_v_exp = Q_v.unsqueeze(3)  # [B, H, T, 1, d_head]

        # Dot product over d_head:
        # Q_v_exp: [B, H, T, 1, d_head]
        # R_exp: [1, H, T, M+T, d_head]
        # We need to sum over d_head.
        # Result: [B, H, T, M+T]

        # Use einsum: 'bhtid,htjd->bhtj'
        term2 = (Q_v_exp * R_exp).sum(dim=-1)

        # Total score: (term1 + term2) / sqrt(d_head)
        scores = (term1 + term2) / (d_head ** 0.5)  # [B, H, T, M+T]

        # 5. Causal Attention
        # Mask: j > M + i => -inf
        # i: 0..T-1
        # j: 0..M+T-1

        # Create mask [T, M+T]
        # j > M + i

        # i_idx: [T, 1]
        # j_idx: [1, M+T]
        # mask: [T, M+T]

        mask = j_idx > (M + i_idx)  # [T, M+T]

        # Expand to [B, H, T, M+T]
        mask_exp = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, M+T]

        # Apply mask
        scores_masked = scores.masked_fill(mask_exp, float('-inf'))

        # Softmax over keys (dim=-1)
        attn_weights = torch.softmax(scores_masked, dim=-1)  # [B, H, T, M+T]

        # V: [M+T, B, H, d_head] -> [B, H, M+T, d_head]
        V_perm = V.permute(1, 2, 0, 3)  # [B, H, M+T, d_head]

        # attn_weights: [B, H, T, M+T]
        # V_perm: [B, H, M+T, d_head]
        # Result: [B, H, T, d_head]

        attn_out = torch.matmul(attn_weights, V_perm)  # [B, H, T, d_head]

        # 6. Output
        # Merge heads: [B, H, T, d_head] -> [B, T, H*d_head]
        attn_out = attn_out.permute(0, 2, 1, 3)  # [B, T, H, d_head]
        attn_out = attn_out.reshape(B, T, H * d_head)

        # Linear: out weight [D, H*d_head]
        # attn_out: [B, T, H*d_head]
        # out.t(): [H*d_head, D]
        # Result: [B, T, D]

        out_proj = torch.matmul(attn_out, self.out.t())  # [B, T, D]

        # 7. Postnorm
        # x: [T, B, D] -> [B, T, D]
        x_perm = x.permute(1, 0, 2)  # [B, T, D]

        # Residual: x + out_proj
        residual = x_perm + out_proj  # [B, T, D]

        # LayerNorm
        mean = residual.mean(dim=-1, keepdim=True)  # [B, T, 1]
        var = residual.var(dim=-1, unbiased=False, keepdim=True)  # [B, T, 1]

        # Normalize
        normalized = (residual - mean) / torch.sqrt(var + self.eps)  # [B, T, D]

        # Scale and shift
        y = normalized * self.gamma + self.beta  # [B, T, D]

        # Permute back to [T, B, D]
        y = y.permute(1, 0, 2)  # [T, B, D]

        # 8. Memory Update
        # next_memory: last memory_length rows of mem_cat, detached
        if self.memory_length > 0:
            next_memory = mem_cat[-self.memory_length:].detach()
        else:
            next_memory = torch.empty(0, B, D, dtype=torch.float64, device=x.device)

        return y, next_memory, scores_masked

def build(spec, parameters):
    return TransformerXLAttention(spec, parameters)
