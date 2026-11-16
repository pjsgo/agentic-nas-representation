import torch
import torch.nn as nn

def build(spec, parameters):
    class CompressiveTransformerMemoryAttention(nn.Module):
        def __init__(self, spec, parameters):
            super().__init__()
            self.spec = spec
            self.parameters = parameters
            self.d_model = spec['config']['d_model']
            self.heads = spec['config']['heads']
            self.d_head = spec['config']['d_head']
            self.episodic_capacity = spec['config']['episodic_capacity']
            self.compressed_capacity = spec['config']['compressed_capacity']
            self.compression_rate = spec['config']['compression_rate']

            # Register parameters as buffers to avoid them being treated as trainable
            # but we need to keep them as the exact objects provided
            self.register_parameter('query', parameters['query'])
            self.register_parameter('key', parameters['key'])
            self.register_parameter('value', parameters['value'])
            self.register_parameter('out', parameters['out'])
            self.register_parameter('out_bias', parameters['out_bias'])
            self.register_parameter('relative_compressed', parameters['relative_compressed'])
            self.register_parameter('relative_episodic', parameters['relative_episodic'])
            self.register_parameter('u', parameters['u'])
            self.register_parameter('v_compressed', parameters['v_compressed'])
            self.register_parameter('v_episodic', parameters['v_episodic'])

        def forward(self, x, episodic, compressed, index):
            B, T, D = x.shape
            M = self.episodic_capacity
            C = self.compressed_capacity
            H = self.heads
            d_head = self.d_head
            rate = self.compression_rate

            # 1. Memory Read: cat [compressed, episodic, x] along axis 1
            # Order: oldest_bank_to_current -> compressed, episodic, x
            memory = torch.cat([compressed, episodic, x], dim=1)
            # memory shape: [B, C+M+T, D]

            # 2. Project
            # Q from x
            # K, V from memory
            # Projections: row-vector input times weight (no transpose)
            # x: [B, T, D] -> [B, T, H*d_head]
            # memory: [B, C+M+T, D] -> [B, C+M+T, H*d_head]

            Q = torch.matmul(x, self.query) # [B, T, H*d_head]
            K = torch.matmul(memory, self.key) # [B, C+M+T, H*d_head]
            V = torch.matmul(memory, self.value) # [B, C+M+T, H*d_head]

            # Scale Q by 1/sqrt(d_head) BEFORE adding u or v
            Q_scaled = Q / (d_head ** 0.5)

            # Split into heads: [B, T, H, d_head]
            Q_h = Q_scaled.view(B, T, H, d_head).permute(0, 2, 1, 3) # [B, H, T, d_head]
            K_h = K.view(B, C+M+T, H, d_head).permute(0, 2, 1, 3) # [B, H, C+M+T, d_head]
            V_h = V.view(B, C+M+T, H, d_head).permute(0, 2, 1, 3) # [B, H, C+M+T, d_head]

            # 3. Relative Scores
            # Sinusoid generation
            # k: 0..D/2-1
            # distance: C+i-j for compressed, M+i-j for episodic/current
            # Note: The relative score is computed as (Q_scaled + v) dot R
            # R is the relative weight matrix applied to sinusoids?
            # "multiply by the respective relative weight, then split heads"
            # relative_compressed shape: [D, H*d_head]
            # This implies the sinusoid vector (size D) is multiplied by relative_compressed (D x H*d_head) to get H*d_head features.

            # Let's compute the sinusoid matrix for all possible distances.
            # Max distance needed:
            # Compressed: i in [0, T-1], j in [0, C-1]. dist = C+i-j. Max: C+T-1. Min: 1.
            # Episodic/Current: i in [0, T-1], j in [0, M+T-1]. dist = M+i-j. Max: M+T-1. Min: 1-M (if j > M+i).
            # Wait, the distance definition is specific.
            # "For compressed-bank slot j, distance is C+i-j"
            # "For episodic/current local slot j, distance is M+i-j"
            # j is the index in the concatenated memory [C, M, T].
            # Compressed slots: j=0..C-1.
            # Episodic slots: j=C..C+M-1. Let j_e = j-C. dist = M+i-j_e.
            # Current slots: j=C+M..C+M+T-1. Let j_c = j-(C+M). dist = M+i-j_c.

            # We need to compute the relative score contribution for each query i and key j.
            # Score_ij = (Q_scaled_i + u) dot K_j + (Q_scaled_i + v) dot R(dist_ij)
            # The equation says: softmax_key((Q_scaled+u) dot K + concat((Q_scaled+v_compressed) dot R_compressed, (Q_scaled+v_episodic) dot R_episodic) + causal_mask) @ V

            # Let's compute the relative part separately.
            # R_compressed for slot j (0..C-1): dist = C+i-j
            # R_episodic for slot j (C..C+M+T-1): dist = M+i-(j-C)

            # We can compute the sinusoid vectors for all i, j pairs.
            # i: 0..T-1
            # j: 0..C+M+T-1

            # Create distance matrix [T, C+M+T]
            i_idx = torch.arange(T).view(T, 1) # [T, 1]
            j_idx = torch.arange(C+M+T).view(1, C+M+T) # [1, C+M+T]

            # Distances for compressed part (j < C)
            dist_comp = C + i_idx - j_idx # [T, C+M+T] (only first C cols valid)
            # Distances for episodic/current part (j >= C)
            # j_local = j - C
            j_local = j_idx - C
            dist_epi = M + i_idx - j_local # [T, C+M+T] (only last M+T cols valid)

            # We need sinusoids for these distances.
            # Sinusoid: concat(sin(dist * 10000**(-2*k/D)), cos(dist * 10000**(-2*k/D)))
            # k: 0..D/2-1

            # Let's compute the sinusoid matrix for all distances in dist_comp and dist_epi.
            # To avoid huge memory, we can compute on the fly or vectorize.
            # D=8, T<=M, M=4, C=2. Small sizes. Vectorization is fine.

            # Flatten distances
            dist_comp_flat = dist_comp.view(-1) # [T*(C+M+T)]
            dist_epi_flat = dist_epi.view(-1)

            # Compute sinusoids
            # k: 0..D/2-1
            k = torch.arange(D // 2, dtype=torch.float64)
            # 10000 ** (-2*k/D)
            angles = 10000 ** (-2 * k / D) # [D/2]

            # sin(dist * angles)
            # dist: [N], angles: [D/2] -> [N, D/2]
            sin_comp = torch.sin(dist_comp_flat.view(-1, 1) * angles.view(1, -1)) # [N, D/2]
            cos_comp = torch.cos(dist_comp_flat.view(-1, 1) * angles.view(1, -1)) # [N, D/2]
            sinusoid_comp = torch.cat([sin_comp, cos_comp], dim=1) # [N, D]

            sin_epi = torch.sin(dist_epi_flat.view(-1, 1) * angles.view(1, -1)) # [N, D/2]
            cos_epi = torch.cos(dist_epi_flat.view(-1, 1) * angles.view(1, -1)) # [N, D/2]
            sinusoid_epi = torch.cat([sin_epi, cos_epi], dim=1) # [N, D]

            # Multiply by relative weights
            # relative_compressed: [D, H*d_head]
            # sinusoid_comp: [N, D]
            # Result: [N, H*d_head]
            R_comp = torch.matmul(sinusoid_comp, self.relative_compressed) # [N, H*d_head]
            R_epi = torch.matmul(sinusoid_epi, self.relative_episodic) # [N, H*d_head]

            # Reshape back to [T, C+M+T, H*d_head]
            R_comp = R_comp.view(T, C+M+T, H*d_head)
            R_epi = R_epi.view(T, C+M+T, H*d_head)

            # We need to combine them based on which part of j is being used.
            # For j < C, use R_comp. For j >= C, use R_epi.
            # Create a mask
            mask_comp = torch.zeros(T, C+M+T, dtype=torch.bool)
            mask_comp[:, :C] = True
            mask_epi = ~mask_comp

            R = torch.where(mask_comp.unsqueeze(-1), R_comp, R_epi) # [T, C+M+T, H*d_head]

            # Now compute the relative score contribution: (Q_scaled + v) dot R
            # Q_scaled: [B, T, H*d_head]
            # v_compressed: [H, d_head] -> needs to be broadcast to [H*d_head]?
            # "v_compressed: [H,d_head]". "shared_content_bias" for u.
            # The equation says: (Q_scaled + v_compressed) dot R_compressed
            # This implies v is added to Q_scaled before dotting with R.
            # But v is [H, d_head]. Q_scaled is [B, T, H, d_head].
            # So we add v to each head of Q_scaled.

            # Let's compute the scores in head space.
            # Q_h: [B, H, T, d_head]
            # K_h: [B, H, C+M+T, d_head]
            # R: [T, C+M+T, H*d_head] -> [T, C+M+T, H, d_head] -> [H, T, C+M+T, d_head]
            R_h = R.view(T, C+M+T, H, d_head).permute(2, 0, 1, 3) # [H, T, C+M+T, d_head]

            # u: [H, d_head]
            # v_compressed: [H, d_head]
            # v_episodic: [H, d_head]

            # We need to add u to Q_scaled for the K dot product part.
            # And add v to Q_scaled for the R dot product part.
            # But v depends on whether the key is compressed or episodic.

            # Let's compute the base score: (Q_scaled + u) dot K
            # Q_scaled_h: [B, H, T, d_head]
            # u: [H, d_head] -> [1, H, 1, d_head]
            Q_plus_u = Q_h + self.u.view(1, H, 1, d_head)
            base_score = torch.matmul(Q_plus_u, K_h.transpose(-2, -1)) # [B, H, T, C+M+T]

            # Now the relative part: (Q_scaled + v) dot R
            # For compressed keys (j < C), use v_compressed.
            # For episodic keys (j >= C), use v_episodic.

            # We can compute two separate relative scores and combine them.
            # Rel_comp: (Q_scaled + v_comp) dot R_comp
            # Rel_epi: (Q_scaled + v_epi) dot R_epi

            Q_plus_v_comp = Q_h + self.v_compressed.view(1, H, 1, d_head)
            # R_comp_h: [H, T, C, d_head] (only first C cols of R)
            R_comp_h = R_h[:, :, :C, :] # [H, T, C, d_head]
            # We need to dot Q_plus_v_comp [B, H, T, d_head] with R_comp_h [H, T, C, d_head]
            # This is not a standard matmul. It's a sum over d_head for each i, j.
            # Score[i, j] = sum_d (Q[i, d] + v[d]) * R[i, j, d]
            # = sum_d Q[i, d] R[i, j, d] + sum_d v[d] R[i, j, d]

            # Let's compute this using einsum or manual broadcasting.
            # Q_plus_v_comp: [B, H, T, d_head]
            # R_comp_h: [H, T, C, d_head]
            # We want [B, H, T, C]

            # term1: sum_d Q[i, d] R[i, j, d]
            # Q: [B, H, T, 1, d_head]
            # R: [1, H, T, C, d_head]
            rel_comp_term1 = torch.sum(Q_plus_v_comp.unsqueeze(3) * R_comp_h.unsqueeze(0), dim=-1) # [B, H, T, C]

            # Wait, R_comp_h is [H, T, C, d_head]. Q is [B, H, T, d_head].
            # We need to align T. Q's T is query index i. R's T is also query index i.
            # So for each i, we dot Q[i] with R[i, j] for all j.

            # Let's use einsum for clarity.
            # Q_plus_v_comp: [B, H, T, d_head]
            # R_comp_h: [H, T, C, d_head]
            # Result: [B, H, T, C]
            rel_comp = torch.einsum('bhtd,htjd->bhtj', Q_plus_v_comp, R_comp_h)

            # Similarly for episodic
            Q_plus_v_epi = Q_h + self.v_episodic.view(1, H, 1, d_head)
            R_epi_h = R_h[:, :, C:, :] # [H, T, M+T, d_head]
            rel_epi = torch.einsum('bhtd,htjd->bhtj', Q_plus_v_epi, R_epi_h) # [B, H, T, M+T]

            # Combine relative scores
            # rel_comp: [B, H, T, C]
            # rel_epi: [B, H, T, M+T]
            rel_score = torch.cat([rel_comp, rel_epi], dim=-1) # [B, H, T, C+M+T]

            # Total score
            scores = base_score + rel_score # [B, H, T, C+M+T]

            # Apply causal mask
            # "mask: all memory allowed; current j>i blocked"
            # Memory slots (j < C+M) are always allowed.
            # Current slots (j >= C+M) are blocked if j > i.
            # j is index in [0, C+M+T-1].
            # i is index in [0, T-1].
            # Current slot index j_c = j - (C+M). j_c in [0, T-1].
            # Block if j_c > i.

            mask = torch.zeros(T, C+M+T, dtype=torch.bool)
            # For j >= C+M:
            j_current = torch.arange(C+M, C+M+T).view(1, T) # [1, T]
            i_idx = torch.arange(T).view(T, 1) # [T, 1]
            # Block if j_current > i_idx
            mask[:, C+M:] = ((j_current - (C + M)) > i_idx) # [T, T]

            # Expand mask to [B, H, T, C+M+T]
            mask = mask.unsqueeze(0).unsqueeze(0).expand(B, H, T, C+M+T)

            # Apply mask
            scores = scores.masked_fill(mask, float('-inf'))

            # Softmax
            # "normalization: single_softmax_across_both_banks_and_current"
            # This means softmax over the last dimension (C+M+T) for each i.
            attn_weights = torch.softmax(scores, dim=-1) # [B, H, T, C+M+T]

            # Weighted sum of V
            # V_h: [B, H, C+M+T, d_head]
            attn_output = torch.matmul(attn_weights, V_h) # [B, H, T, d_head]

            # Merge heads
            # [B, H, T, d_head] -> [B, T, H*d_head]
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, H*d_head)

            # Output projection
            # mix @ out + out_bias
            y = torch.matmul(attn_output, self.out) + self.out_bias # [B, T, D]

            # 4. Evict and Compress
            # "Before updating, take the oldest T rows of OLD episodic memory."
            # episodic: [B, M, D]
            # Oldest T rows: episodic[:, :T, :]
            oldest_episodic = episodic[:, :T, :] # [B, T, D]

            # Mean-pool consecutive non-overlapping rate-sized time blocks
            # kernel=stride=rate, valid
            # Input: [B, T, D]
            # We need to reshape to [B, D, T] for avg_pool1d? Or use manual pooling.
            # avg_pool1d expects [B, C, L]. Here C=D, L=T.
            # Let's use manual pooling to be safe with shapes.
            # T % rate == 0
            T_pooled = T // rate
            # Reshape [B, T, D] -> [B, T_pooled, rate, D] -> mean over rate
            oldest_episodic_reshaped = oldest_episodic.view(B, T_pooled, rate, D)
            pooled = oldest_episodic_reshaped.mean(dim=2) # [B, T_pooled, D]

            # 5. Update States
            # Episodic Update: concat_tail(episodic, x, keep=M)
            # episodic: [B, M, D]
            # x: [B, T, D]
            # Concat: [B, M+T, D]
            # Keep last M: [B, M, D]
            new_episodic = torch.cat([episodic, x], dim=1)[:, -M:, :] # [B, M, D]

            # Compressed Update: concat_tail(compressed, pooled, keep=C)
            # compressed: [B, C, D]
            # pooled: [B, T_pooled, D]
            # Concat: [B, C+T_pooled, D]
            # Keep last C: [B, C, D]
            new_compressed = torch.cat([compressed, pooled], dim=1)[:, -C:, :] # [B, C, D]

            # Index Update
            new_index = index + 1

            # Auxiliary Loss
            auxiliary_loss = torch.tensor(0.0, dtype=torch.float64)

            return {
                'y': y,
                'episodic': new_episodic,
                'compressed': new_compressed,
                'index': new_index,
                'pooled': pooled,
                'scores': scores,
                'auxiliary_loss': auxiliary_loss
            }

    return CompressiveTransformerMemoryAttention(spec, parameters)
