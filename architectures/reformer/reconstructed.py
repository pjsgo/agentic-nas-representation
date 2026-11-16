import torch
import math

class ReformerLSHCore(torch.nn.Module):
    def __init__(self, num_buckets: int, chunk_len: int, causal: bool = False):
        super().__init__()
        self.num_buckets = num_buckets
        self.chunk_len = chunk_len
        self.causal = causal

    def forward(self, qk, v, rotations):
        T, D = qk.shape
        Dv = v.shape[1]
        _, R, half_K = rotations.shape
        K = self.num_buckets
        C = self.chunk_len
        
        # 1. reformer_hash_sort
        # z[r,t,a] = sum_d qk[t,d] * rotations[d,r,a]
        # qk: [T, D], rotations: [D, R, K/2]
        # We need z for each r, t, a. 
        # Let's compute z for all r, t, a.
        # qk: [T, D] -> [1, T, D]
        # rotations: [D, R, K/2] -> [D, R, 1, K/2] ? No, let's align dims.
        # We want z[r, t, a] where a is in 0..K-1.
        # The equation says: h[r,t] = argmax_a concat(z[r,t,:], -z[r,t,:])[a] in 0..K-1.
        # This implies z[r,t,:] has length K/2. Let's call it z_half.
        # z_half[r, t, a_half] = sum_d qk[t, d] * rotations[d, r, a_half]
        
        # qk: [T, D]
        # rotations: [D, R, K/2]
        # We can do: qk.unsqueeze(0) [1, T, D] * rotations.permute(1, 0, 2) [R, D, K/2] -> [R, T, K/2] ?
        # Let's check dims:
        # qk: [T, D]
        # rotations: [D, R, K/2]
        # We want to sum over D.
        # qk: [T, D] -> [1, T, D]
        # rotations: [D, R, K/2] -> [D, R, 1, K/2] ? No.
        # Let's permute rotations to [R, D, K/2].
        # rot_rdk = rotations.permute(1, 0, 2) # [R, D, K/2]
        # qk: [T, D] -> [1, T, D]
        # z_half = torch.einsum('td,rdk->rtk', qk, rot_rdk) # [R, T, K/2]
        
        rot_rdk = rotations.permute(1, 0, 2) # [R, D, K/2]
        z_half = torch.einsum('td,rdk->rtk', qk, rot_rdk) # [R, T, K/2]
        
        # concat(z_half, -z_half) along last dim -> [R, T, K]
        z_full = torch.cat([z_half, -z_half], dim=-1) # [R, T, K]
        
        # h[r,t] = argmax_a z_full[r,t,a]
        h = torch.argmax(z_full, dim=-1) # [R, T]
        
        # buckets[r,t] = h[r,t] + r*K
        r_idx = torch.arange(R, device=qk.device).unsqueeze(1) # [R, 1]
        buckets = h + r_idx * K # [R, T]
        
        # sort_key[u] = T * buckets[r,t] + t, with u = r*T + t
        # buckets: [R, T] -> flatten to [R*T]
        buckets_flat = buckets.flatten() # [R*T]
        t_idx = torch.arange(T, device=qk.device).unsqueeze(0).expand(R, T).flatten() # [R*T]
        sort_key = T * buckets_flat + t_idx # [R*T]
        
        # permutation = argsort(sort_key) ascending
        permutation = torch.argsort(sort_key) # [R*T]
        
        # token[s] = permutation[s] mod T
        token = permutation % T # [R*T]
        
        # Inverse[permutation[s]] = s
        # We need Inverse such that Inverse[u] = s where permutation[s] = u.
        # This is the inverse permutation.
        inverse_perm = torch.empty_like(permutation)
        inverse_perm[permutation] = torch.arange(R * T, device=qk.device)
        
        # 2. reformer_chunk_connections
        # Sorted occurrences are reshaped to [N, C]. N = R*T / C.
        N = (R * T) // C
        
        # For sorted query s, c = floor(s/C).
        # Its 2*C key occurrence rows are ((c-1) mod N)*C + l for l=0..C-1,
        # followed by c*C + l for l=0..C-1.
        # key_indices[s, a] = token[key_row(s, a)]
        
        # s ranges from 0 to R*T - 1.
        # c = s // C
        s_idx = torch.arange(R * T, device=qk.device) # [R*T]
        c_idx = s_idx // C # [R*T]
        
        # l ranges from 0 to C-1
        l_idx = torch.arange(C, device=qk.device) # [C]
        
        # prev_chunk_start = ((c-1) mod N) * C
        # curr_chunk_start = c * C
        
        # We need to construct key_rows for each s.
        # key_rows shape: [R*T, 2*C]
        # For each s, the first C keys are from prev chunk, next C from curr chunk.
        
        # prev_start = ((c_idx - 1) % N) * C
        # curr_start = c_idx * C
        
        # We can broadcast c_idx [R*T, 1] and l_idx [1, C] to get offsets.
        c_idx_exp = c_idx.unsqueeze(1) # [R*T, 1]
        l_idx_exp = l_idx.unsqueeze(0) # [1, C]
        
        prev_start = ((c_idx_exp - 1) % N) * C # [R*T, 1]
        curr_start = c_idx_exp * C # [R*T, 1]
        
        prev_rows = prev_start + l_idx_exp # [R*T, C]
        curr_rows = curr_start + l_idx_exp # [R*T, C]
        
        key_rows = torch.cat([prev_rows, curr_rows], dim=1) # [R*T, 2*C]
        
        # key_indices[s, a] = token[key_rows[s, a]]
        key_indices = token[key_rows] # [R*T, 2*C]
        
        # 3. reformer_score_mask
        # key[j,d] = qk[j,d] / sqrt(mean_e(qk[j,e]^2) + 1e-6) / sqrt(D)
        # qk: [T, D]
        # mean_sq = torch.mean(qk**2, dim=1, keepdim=True) # [T, 1]
        # norm = torch.sqrt(mean_sq + 1e-6) * torch.sqrt(torch.tensor(D, dtype=qk.dtype))
        # key = qk / norm # [T, D]
        
        mean_sq = torch.mean(qk**2, dim=1, keepdim=True) # [T, 1]
        norm = torch.sqrt(mean_sq + 1e-6) * math.sqrt(D)
        key = qk / norm # [T, D]
        
        # For sorted row s with i=token[s] and j=key_indices[s,a],
        # score[s,a] = sum_d qk[i,d] * key[j,d]
        
        # i = token[s] -> [R*T]
        # j = key_indices[s,a] -> [R*T, 2*C]
        
        # qk_i = qk[token] # [R*T, D]
        # key_j = key[key_indices] # [R*T, 2*C, D]
        
        qk_i = qk[token] # [R*T, D]
        key_j = key[key_indices] # [R*T, 2*C, D]
        
        # score[s,a] = sum_d qk_i[s,d] * key_j[s,a,d]
        # score = torch.einsum('sd,sad->sa', qk_i, key_j) # [R*T, 2*C]
        score = torch.einsum('sd,sad->sa', qk_i, key_j) # [R*T, 2*C]
        
        # If causal, subtract 1e9 when j > i
        # always subtract 1e5 when j == i
        
        # i: [R*T, 1]
        # j: [R*T, 2*C]
        i_exp = token.unsqueeze(1) # [R*T, 1]
        j_exp = key_indices # [R*T, 2*C]
        
        if self.causal:
            causal_mask = (j_exp > i_exp).to(qk.dtype) # [R*T, 2*C]
            score = score - 1e9 * causal_mask
        
        self_mask = (j_exp == i_exp).to(qk.dtype) # [R*T, 2*C]
        score = score - 1e5 * self_mask
        
        # 4. reformer_chunk_attention
        # L[s] = max_a(score[s,a]) + log(sum_a exp(score[s,a] - max_a(score[s,a])))
        # P[s,a] = exp(score[s,a] - L[s])
        # O[s,b] = sum_a P[s,a] * v[key_indices[s,a], b]
        
        max_score = torch.max(score, dim=1, keepdim=True).values # [R*T, 1]
        exp_score = torch.exp(score - max_score) # [R*T, 2*C]
        sum_exp = torch.sum(exp_score, dim=1, keepdim=True) # [R*T, 1]
        L = max_score + torch.log(sum_exp) # [R*T, 1]
        
        P = exp_score / sum_exp # [R*T, 2*C]
        
        # v[key_indices[s,a], b] -> [R*T, 2*C, Dv]
        v_j = v[key_indices] # [R*T, 2*C, Dv]
        
        # O[s,b] = sum_a P[s,a] * v_j[s,a,b]
        O = torch.einsum('sa,sab->sb', P, v_j) # [R*T, Dv]
        
        # 5. reformer_restore_merge
        # round_logsumexp[r,t] = L[Inverse[r*T+t]]
        # L is [R*T, 1]. We need to reshape to [R, T] using inverse_perm.
        # L_flat = L.flatten() # [R*T]
        # round_logsumexp_flat = L_flat[inverse_perm] # [R*T]
        # round_logsumexp = round_logsumexp_flat.reshape(R, T) # [R, T]
        
        L_flat = L.flatten() # [R*T]
        round_logsumexp_flat = L_flat[inverse_perm] # [R*T]
        round_logsumexp = round_logsumexp_flat.reshape(R, T) # [R, T]
        
        # round_output[r,t,b] = O[Inverse[r*T+t], b]
        # O is [R*T, Dv]
        # round_output_flat = O[inverse_perm] # [R*T, Dv]
        # round_output = round_output_flat.reshape(R, T, Dv) # [R, T, Dv]
        
        round_output_flat = O[inverse_perm] # [R*T, Dv]
        round_output = round_output_flat.reshape(R, T, Dv) # [R, T, Dv]
        
        # A[t] = max_r round_logsumexp[r,t]
        A = torch.max(round_logsumexp, dim=0, keepdim=True).values # [1, T]
        
        # G[t] = A[t] + log(sum_r exp(round_logsumexp[r,t] - A[t]))
        exp_round = torch.exp(round_logsumexp - A) # [R, T]
        sum_exp_round = torch.sum(exp_round, dim=0, keepdim=True) # [1, T]
        G = A + torch.log(sum_exp_round) # [1, T]
        
        # round_weights[r,t] = exp(round_logsumexp[r,t] - G[t])
        round_weights = torch.exp(round_logsumexp - G) # [R, T]
        
        # output[t,b] = sum_r round_weights[r,t] * round_output[r,t,b]
        # round_weights: [R, T] -> [R, T, 1]
        # round_output: [R, T, Dv]
        output = torch.sum(round_weights.unsqueeze(2) * round_output, dim=0) # [T, Dv]
        
        return {
            'buckets': buckets,
            'permutation': permutation,
            'key_indices': key_indices,
            'round_logsumexp': round_logsumexp,
            'round_weights': round_weights,
            'output': output
        }