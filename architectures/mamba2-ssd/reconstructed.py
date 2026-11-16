import torch
import torch.nn as nn

class LocalSSMCore(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.heads = 2
        self.head_dim = 3
        self.state_size = 4
        self.block_len = 2

        self.A_log = nn.Parameter(values['A_log'].clone())
        self.dt_bias = nn.Parameter(values['dt_bias'].clone())
        self.D = nn.Parameter(values['D'].clone())

    def forward(self, x, dt_logits, B, C, initial_state):
        batch, time, heads, head_dim = x.shape
        Q = self.block_len
        M = time // Q

        # 1. Discretize
        # dt: [batch, time, heads]
        dt = torch.nn.functional.softplus(dt_logits + self.dt_bias)

        # A: [heads] -> [1, 1, heads, 1]
        A = -torch.exp(self.A_log)

        # a: [batch, time, heads]
        a = dt * A

        # X: [batch, time, heads, head_dim]
        X = dt.unsqueeze(-1) * x

        # 2. Chunk Partition
        # Reshape to chunks
        # Xc: [batch, M, Q, heads, head_dim]
        Xc = X.view(batch, M, Q, heads, head_dim)
        # ac: [batch, M, Q, heads]
        ac = a.view(batch, M, Q, heads)
        # Bc: [batch, M, Q, heads, state_size]
        Bc = B.view(batch, M, Q, heads, self.state_size)
        # Cc: [batch, M, Q, heads, state_size]
        Cc = C.view(batch, M, Q, heads, self.state_size)

        # prefix: [batch, M, Q, heads]
        prefix = torch.cumsum(ac, dim=2)

        # 3. Diagonal Blocks
        # L[i, j] = exp(sum_{k=j+1..i} a[k]) for i >= j else 0
        # sum_{k=j+1..i} a[k] = prefix[i] - prefix[j] (if j < i) or 0 (if j == i)
        # Note: prefix is cumulative sum of a.
        # Let's compute the exponent matrix E[i, j]
        # E[i, j] = prefix[i] - prefix[j] for i > j
        # E[i, i] = 0
        # E[i, j] = -inf for i < j

        # prefix: [batch, M, Q, heads]
        # We need to broadcast to [batch, M, Q, Q, heads]
        # p_i = prefix[..., i, :] -> [batch, M, Q, heads]
        # p_j = prefix[..., j, :] -> [batch, M, Q, heads]

        p_i = prefix.unsqueeze(3) # [batch, M, 1, Q, heads] (i index)
        p_j = prefix.unsqueeze(2) # [batch, M, Q, 1, heads] (j index)

        # diff = p_i - p_j -> [batch, M, Q, Q, heads]
        diff = p_i - p_j

        # Mask for i >= j
        # i is dim 2, j is dim 3
        i_idx = torch.arange(Q, device=x.device).view(1, 1, Q, 1)
        j_idx = torch.arange(Q, device=x.device).view(1, 1, 1, Q)
        mask = i_idx >= j_idx # [1, 1, Q, Q]

        # For i < j, we want L=0. exp(-inf) = 0.
        # For i >= j, L = exp(diff).
        # Note: if i==j, diff=0, exp(0)=1. Correct.

        # Set diff to -inf where mask is False
        diff = torch.where(mask.unsqueeze(-1), diff, torch.tensor(float('-inf'), device=x.device))

        L = torch.exp(diff) # [batch, M, Q, Q, heads]

        # Ydiag[i, p] = sum_{j, n} C[i, n] * B[j, n] * L[i, j] * X[j, p]
        # Cc: [batch, M, Q, heads, state_size] -> C[i, n]
        # Bc: [batch, M, Q, heads, state_size] -> B[j, n]
        # Xc: [batch, M, Q, heads, head_dim] -> X[j, p]

        # C_i: [batch, M, Q, heads, 1, state_size] (i index)
        C_i = Cc.unsqueeze(3) # [batch, M, Q, 1, heads, state_size]
        # B_j: [batch, M, 1, Q, heads, state_size] (j index)
        B_j = Bc.unsqueeze(2) # [batch, M, 1, Q, heads, state_size]
        # X_j: [batch, M, 1, Q, heads, head_dim] (j index)
        X_j = Xc.unsqueeze(2) # [batch, M, 1, Q, heads, head_dim]

        # L: [batch, M, Q, Q, heads] -> [batch, M, Q, Q, heads, 1]
        L_exp = L.unsqueeze(-1)

        # Term = C_i * B_j * L_exp * X_j
        # C_i: [b, M, Q, 1, h, n]
        # B_j: [b, M, 1, Q, h, n]
        # L_exp: [b, M, Q, Q, h, 1]
        # X_j: [b, M, 1, Q, h, p]

        # Product: [b, M, Q, Q, h, n, p] ? No, dimensions must align.
        # C_i * B_j -> [b, M, Q, Q, h, n]
        # * L_exp -> [b, M, Q, Q, h, n]
        # * X_j -> [b, M, Q, Q, h, n, p] (n and p are separate dims)

        # We need to sum over j (dim 3) and n (dim 5).
        # Result Ydiag: [b, M, Q, h, p]

        # Let's do it step by step to avoid huge intermediate tensors if possible, but Q=2 is small.
        # Term = C_i * B_j * L_exp
        # C_i: [b, M, Q, 1, h, n]
        # B_j: [b, M, 1, Q, h, n]
        # L_exp: [b, M, Q, Q, h, 1]

        # prod1 = C_i * B_j * L_exp -> [b, M, Q, Q, h, n]
        prod1 = C_i * B_j * L_exp

        # Now multiply by X_j: [b, M, 1, Q, h, p]
        # prod2 = prod1 * X_j -> [b, M, Q, Q, h, n, p]
        prod2 = prod1.unsqueeze(-1) * X_j.unsqueeze(-2) # [b, M, Q, Q, h, n, p]

        # Sum over j (dim 3) and n (dim 5)
        Ydiag = torch.sum(prod2, dim=(3, 5)) # [b, M, Q, h, p]

        # 4. Local State Factor
        # U[c, p, n] = sum_j exp(prefix[c, -1] - prefix[c, j]) * X[c, j, p] * B[c, j, n]
        # prefix_last = prefix[..., -1, :] -> [b, M, h]
        prefix_last = prefix[:, :, -1, :] # [b, M, h]

        # exp_term = exp(prefix_last - prefix) -> [b, M, Q, h]
        exp_term = torch.exp(prefix_last.unsqueeze(2) - prefix) # [b, M, Q, h]

        # Xc: [b, M, Q, h, p]
        # Bc: [b, M, Q, h, n]

        # U[c, p, n] = sum_j exp_term[j] * Xc[j, p] * Bc[j, n]
        # exp_term: [b, M, Q, h] -> [b, M, Q, h, 1, 1]
        # Xc: [b, M, Q, h, p] -> [b, M, Q, h, p, 1]
        # Bc: [b, M, Q, h, n] -> [b, M, Q, h, 1, n]

        local_states = torch.sum(
            exp_term.unsqueeze(-1).unsqueeze(-1) *
            Xc.unsqueeze(-1) *
            Bc.unsqueeze(-2),
            dim=2
        ) # [b, M, h, p, n]

        # 5. Boundary Propagation
        # d[0] = 0
        # d[c+1] = prefix[c, Q-1] = prefix_last[c]
        # d: [b, M+1, h]
        d = torch.cat([torch.zeros(batch, 1, heads, device=x.device), prefix_last], dim=1) # [b, M+1, h]

        # V[0] = initial_state: [b, h, p, n]
        # V[c+1] = U[c]: [b, M, h, p, n]
        # V: [b, M+1, h, p, n]
        V = torch.cat([initial_state.unsqueeze(1), local_states], dim=1) # [b, M+1, h, p, n]

        # W[z, r] = 0 if z < r else exp(sum_{k=r+1..z} d[k])
        # sum_{k=r+1..z} d[k] = D_cumsum[z] - D_cumsum[r]
        # where D_cumsum[i] = sum_{k=0..i} d[k]

        D_cumsum = torch.cumsum(d, dim=1) # [b, M+1, h]

        # W[z, r] = exp(D_cumsum[z] - D_cumsum[r]) for z >= r else 0
        # D_cumsum_z: [b, M+1, 1, h]
        # D_cumsum_r: [b, 1, M+1, h]

        D_z = D_cumsum.unsqueeze(2) # [b, M+1, 1, h]
        D_r = D_cumsum.unsqueeze(1) # [b, 1, M+1, h]

        W_diff = D_z - D_r # [b, M+1, M+1, h]

        # Mask z >= r
        z_idx = torch.arange(M+1, device=x.device).view(1, M+1, 1)
        r_idx = torch.arange(M+1, device=x.device).view(1, 1, M+1)
        W_mask = z_idx >= r_idx # [1, M+1, M+1]

        W_diff = torch.where(W_mask.unsqueeze(-1), W_diff, torch.tensor(float('-inf'), device=x.device))
        W = torch.exp(W_diff) # [b, M+1, M+1, h]

        # all_states[z] = sum_{r=0..M} W[z, r] * V[r]
        # V: [b, M+1, h, p, n] -> [b, M+1, 1, h, p, n] (r index)
        # W: [b, M+1, M+1, h] -> [b, M+1, M+1, h, 1, 1]

        V_exp = V.unsqueeze(1) # [b, M+1, 1, h, p, n]
        W_exp = W.unsqueeze(-1).unsqueeze(-1) # [b, M+1, M+1, h, 1, 1]

        all_states = torch.sum(W_exp * V_exp, dim=2) # [b, M+1, h, p, n]

        starts = all_states[:, :-1] # [b, M, h, p, n]
        final_state = all_states[:, -1] # [b, h, p, n]

        # 6. Offdiagonal Blocks
        # Yoff[c, i, p] = exp(prefix[c, i]) * sum_n C[c, i, n] * starts[c, p, n]
        # prefix: [b, M, Q, h]
        # Cc: [b, M, Q, h, n]
        # starts: [b, M, h, p, n]

        # exp_prefix = exp(prefix) -> [b, M, Q, h]
        exp_prefix = torch.exp(prefix) # [b, M, Q, h]

        # sum_n C[c, i, n] * starts[c, p, n]
        # Cc: [b, M, Q, h, n]
        # starts: [b, M, h, p, n] -> [b, M, 1, h, p, n]

        Cc_exp = Cc.unsqueeze(-2) # [b, M, Q, h, n] -> [b, M, Q, h, n] (i index)
        starts_exp = starts.unsqueeze(2) # [b, M, 1, h, p, n]

        # Product: [b, M, Q, h, p, n]
        prod_off = Cc_exp * starts_exp # [b, M, Q, h, p, n]

        # Sum over n (dim 5)
        sum_off = torch.sum(prod_off, dim=5) # [b, M, Q, h, p]

        # Multiply by exp_prefix
        Yoff = exp_prefix.unsqueeze(-1) * sum_off # [b, M, Q, h, p]

        # 7. Merge and Skip
        # y = reshape(Ydiag + Yoff) + D[head] * x
        # Ydiag: [b, M, Q, h, p]
        # Yoff: [b, M, Q, h, p]

        y_chunked = Ydiag + Yoff # [b, M, Q, h, p]
        y = y_chunked.view(batch, time, heads, head_dim)

        # D: [h] -> [1, 1, h, 1]
        D_exp = self.D.view(1, 1, heads, 1)
        y = y + D_exp * x

        return {
            'y': y,
            'final_state': final_state
        }


def build(values):
    return LocalSSMCore(values)
