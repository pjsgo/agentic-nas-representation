import torch
import torch.nn as nn
import math

class LongformerCore(nn.Module):
    def __init__(self, embed_dim, num_heads, radius):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.radius = radius
        self.head_dim = embed_dim // num_heads

        # Register exactly the six named affine modules
        self.query = nn.Linear(embed_dim, embed_dim, bias=True)
        self.key = nn.Linear(embed_dim, embed_dim, bias=True)
        self.value = nn.Linear(embed_dim, embed_dim, bias=True)
        self.query_global = nn.Linear(embed_dim, embed_dim, bias=True)
        self.key_global = nn.Linear(embed_dim, embed_dim, bias=True)
        self.value_global = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x, valid, global_mask):
        B, N, E = x.shape
        H = self.num_heads
        D = self.head_dim
        r = self.radius

        # 1. Projections
        # Q, K, V for local attention
        Q = self.query(x) # [B, N, E]
        K = self.key(x)   # [B, N, E]
        V = self.value(x) # [B, N, E]

        # Qg, Kg, Vg for global attention
        Qg = self.query_global(x) # [B, N, E]
        Kg = self.key_global(x)   # [B, N, E]
        Vg = self.value_global(x) # [B, N, E]

        # Reshape to [B, N, H, D]
        Q = Q.view(B, N, H, D)
        K = K.view(B, N, H, D)
        V = V.view(B, N, H, D)

        Qg = Qg.view(B, N, H, D)
        Kg = Kg.view(B, N, H, D)
        Vg = Vg.view(B, N, H, D)

        # Initialize output
        y = torch.zeros_like(x) # [B, N, E]

        # Precompute masks for efficiency if needed, but N is small enough for loops or vectorization.
        # Given constraints (E<=8, arbitrary N), we can iterate over batch and heads or use vectorized ops.
        # Let's try to vectorize where possible, but the window logic is per-token.

        # We will compute O (ordinary) and Og (global) for all positions, then select.
        # O[b, i, h, d] is defined for valid non-global i.
        # Og[b, i, h, d] is defined for global i.

        # Let's compute attention scores and weights.

        # --- Ordinary Attention ---
        # For each valid non-global i:
        # S = L[b,i] union G[b]
        # L[b,i] = {j: valid[b,j] and not global_mask[b,j] and abs(i-j)<=r}
        # G[b] = {j: global_mask[b,j]}
        # Note: L and G are disjoint because L excludes global_mask.

        # We can compute the dot products Q[b,i] * K[b,j] for all j.
        # Q: [B, N, H, D]
        # K: [B, N, H, D]

        # Compute scores: [B, N, H, N] where scores[b, i, h, j] = sum_d Q[b,i,h,d]*K[b,j,h,d] / sqrt(D)
        # Using einsum: 'b i h d, b j h d -> b i h j'
        scores_local = torch.einsum('bihd,bjhd->bihj', Q, K) / math.sqrt(D)

        # Mask for local window L[b,i]
        # valid: [B, N]
        # global_mask: [B, N]
        # We need a mask [B, N, N] where mask[b, i, j] is True if j is in L[b,i]
        # Condition: valid[b,j] AND NOT global_mask[b,j] AND abs(i-j)<=r

        # Let's construct the mask.
        # valid_expanded: [B, 1, N] (for j dimension)
        # global_mask_expanded: [B, 1, N]
        # i_indices: [1, N, 1]
        # j_indices: [1, 1, N]

        i_idx = torch.arange(N, device=x.device).view(1, N, 1)
        j_idx = torch.arange(N, device=x.device).view(1, 1, N)

        # Distance mask: abs(i-j) <= r
        dist_mask = (i_idx - j_idx).abs() <= r # [1, N, N]

        # Valid and non-global mask for keys j
        # valid[b, j] is [B, N]
        # global_mask[b, j] is [B, N]
        # We want keys j that are valid and NOT global.
        # key_mask_local[b, j] = valid[b, j] AND NOT global_mask[b, j]
        key_mask_local = valid & (~global_mask) # [B, N]

        # Expand to [B, N, N] where the first N is i (query) and second N is j (key)
        # The condition for j depends on b and j, not i. So we broadcast key_mask_local to [B, 1, N] and then to [B, N, N]
        # Actually, the mask for the score matrix scores[b, i, h, j] needs to be applied to the j dimension.
        # So we need a mask [B, N, N] where mask[b, i, j] = dist_mask[i, j] AND key_mask_local[b, j]

        # dist_mask is [1, N, N]
        # key_mask_local is [B, N] -> [B, 1, N]

        local_key_mask = key_mask_local.unsqueeze(1) # [B, 1, N]
        # Combine with distance mask
        # dist_mask is [1, N, N]. We need to broadcast it to [B, N, N].
        # local_window_mask[b, i, j] = dist_mask[i, j] AND local_key_mask[b, j]
        local_window_mask = dist_mask & local_key_mask # [B, N, N] (broadcasting works: [1,N,N] & [B,1,N] -> [B,N,N])

        # Now, for ordinary attention, the set of keys S is L union G.
        # G[b] = {j: global_mask[b, j]}
        # global_key_mask[b, j] = global_mask[b, j]
        global_key_mask = global_mask # [B, N]
        global_key_mask_expanded = global_key_mask.unsqueeze(1) # [B, 1, N]

        # The union mask for ordinary attention keys:
        # ordinary_key_mask[b, i, j] = local_window_mask[b, i, j] OR global_key_mask_expanded[b, j]
        # Note: L and G are disjoint, so OR is equivalent to union.
        ordinary_key_mask = local_window_mask | global_key_mask_expanded # [B, N, N]

        # Apply mask to scores_local
        # scores_local is [B, N, H, N]
        # ordinary_key_mask is [B, N, N] -> [B, N, 1, N]
        scores_masked = scores_local.masked_fill(~ordinary_key_mask.unsqueeze(2), float('-inf'))

        # Softmax over j (dim -1)
        # Handle case where all keys are masked (e.g., no valid keys in window and no global keys)
        # If a row is all -inf, softmax will produce NaN. We should handle this.
        # The spec says: "entirely padded batch rows are allowed and return zero".
        # Also, if a valid non-global token has no keys in S, what happens?
        # S = L union G. If L is empty and G is empty, S is empty.
        # Softmax over empty set is undefined. However, the problem states "global_mask is a subset of valid".
        # If a token is valid and non-global, it must have at least itself in L?
        # L[b,i] includes j=i if valid[b,i] and not global_mask[b,i] and abs(i-i)<=r (0<=r).
        # Since i is valid and non-global, j=i is in L. So S is never empty for valid non-global i.
        # So we are safe for valid non-global tokens.

        # For global tokens, we compute Og separately.

        # Compute weights for ordinary attention
        # p[b, i, h, j] = softmax_j(scores_masked[b, i, h, j])
        # Empty key rows must not pass all -inf logits into softmax.
        ordinary_allowed = ordinary_key_mask.unsqueeze(2)
        ordinary_has_key = ordinary_allowed.any(dim=-1, keepdim=True)
        safe_local_scores = torch.where(ordinary_has_key, scores_masked, torch.zeros_like(scores_masked))
        p_ordinary = torch.softmax(safe_local_scores, dim=-1).masked_fill(~ordinary_allowed, 0.0)

        # Compute O[b, i, h, d] = sum_j p[b, i, h, j] * V[b, j, h, d]
        # V is [B, N, H, D]
        # p_ordinary is [B, N, H, N]
        # O = einsum('bihj,bjhd->bihd', p_ordinary, V)
        O = torch.einsum('bihj,bjhd->bihd', p_ordinary, V)

        # --- Global Attention ---
        # For global i:
        # A = {j: valid[b, j]}
        # pg[j] = softmax_j(sum_d Qg[b,i,h,d]*Kg[b,j,h,d]/sqrt(D))
        # Og[b,i,h,d] = sum_j pg[j]*Vg[b,j,h,d]

        # Compute scores_global: [B, N, H, N]
        scores_global = torch.einsum('bihd,bjhd->bihj', Qg, Kg) / math.sqrt(D)

        # Mask for A: valid[b, j]
        valid_key_mask = valid # [B, N]
        valid_key_mask_expanded = valid_key_mask.unsqueeze(1) # [B, 1, N]

        # Apply mask
        scores_global_masked = scores_global.masked_fill(~valid_key_mask_expanded.unsqueeze(2), float('-inf'))

        # Softmax
        # For global tokens, A is the set of all valid keys.
        # If there are no valid keys in the batch row, A is empty.
        # The spec says "entirely padded batch rows are allowed and return zero".
        # If a row has no valid keys, then for any global token in that row (which must be valid, so if row is entirely padded, there are no global tokens), it's fine.
        # If a row has some valid keys, then for a global token, A is non-empty.
        # So we are safe.

        global_allowed = valid_key_mask_expanded.unsqueeze(2)
        global_has_key = global_allowed.any(dim=-1, keepdim=True)
        safe_global_scores = torch.where(global_has_key, scores_global_masked, torch.zeros_like(scores_global_masked))
        p_global = torch.softmax(safe_global_scores, dim=-1).masked_fill(~global_allowed, 0.0)

        # Compute Og
        Og = torch.einsum('bihj,bjhd->bihd', p_global, Vg)

        # --- Output Selection ---
        # y[b,i] = flatten_heads(Og[b,i]) if global_mask[b,i]
        #          else flatten_heads(O[b,i]) if valid[b,i]
        #          else zero[E]

        # O and Og are [B, N, H, D]
        # Flatten heads: [B, N, E]
        O_flat = O.reshape(B, N, E)
        Og_flat = Og.reshape(B, N, E)

        # Select
        # global_mask: [B, N]
        # valid: [B, N]

        # We can use torch.where
        # First, select between O_flat and Og_flat based on global_mask
        # If global_mask is True, use Og_flat, else use O_flat
        selected = torch.where(global_mask.unsqueeze(-1), Og_flat, O_flat)

        # Then, if not valid, set to zero
        # valid: [B, N] -> [B, N, 1]
        valid_mask = valid.unsqueeze(-1)
        y = torch.where(valid_mask, selected, torch.zeros_like(selected))

        return y
