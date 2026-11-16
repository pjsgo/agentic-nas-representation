import math
import torch

def attention(q, k, v, query_tile, key_tile, causal=False):
    B, H, N, D = q.shape
    
    # Initialize state for each query row
    # m: running max, shape [B, H, N]
    # l: running sum of exp(s - m), shape [B, H, N]
    # a: running weighted sum, shape [B, H, N, D]
    m = torch.full((B, H, N), float('-inf'), dtype=torch.float64, device=q.device)
    l = torch.zeros((B, H, N), dtype=torch.float64, device=q.device)
    a = torch.zeros((B, H, N, D), dtype=torch.float64, device=q.device)
    
    # Iterate over query tiles
    for i_start in range(0, N, query_tile):
        i_end = min(i_start + query_tile, N)
        
        # Extract query slice
        q_slice = q[:, :, i_start:i_end, :] # [B, H, Tq, D]
        Tq = i_end - i_start
        
        # Initialize local accumulators for this query tile
        # We need to update the global m, l, a for rows i_start to i_end
        # Let's work with slices of the global state
        m_slice = m[:, :, i_start:i_end] # [B, H, Tq]
        l_slice = l[:, :, i_start:i_end] # [B, H, Tq]
        a_slice = a[:, :, i_start:i_end, :] # [B, H, Tq, D]
        
        # Iterate over key tiles
        for j_start in range(0, N, key_tile):
            j_end = min(j_start + key_tile, N)
            
            # Extract key/value slices
            k_slice = k[:, :, j_start:j_end, :] # [B, H, Tk, D]
            v_slice = v[:, :, j_start:j_end, :] # [B, H, Tk, D]
            Tk = j_end - j_start
            
            # Compute scores s[i, j] = sum_d q[i,d] * k[j,d] / sqrt(D)
            # q_slice: [B, H, Tq, D]
            # k_slice: [B, H, Tk, D]
            # We need s: [B, H, Tq, Tk]
            # s[b, h, i, j] = dot(q[b,h,i,:], k[b,h,j,:]) / sqrt(D)
            
            # Using einsum or matmul
            # q_slice @ k_slice.transpose(-2, -1) -> [B, H, Tq, Tk]
            scores = torch.matmul(q_slice, k_slice.transpose(-2, -1)) / math.sqrt(D)
            
            # Apply causal mask if needed
            if causal:
                # i is absolute index: i_start + i_local
                # j is absolute index: j_start + j_local
                # Condition: j > i is disallowed, so we keep j <= i
                # i_local ranges 0..Tq-1, j_local ranges 0..Tk-1
                # i_abs = i_start + i_local
                # j_abs = j_start + j_local
                # Mask where j_abs > i_abs
                
                # Create index grids
                i_idx = torch.arange(i_start, i_end, device=q.device).view(1, 1, Tq, 1)
                j_idx = torch.arange(j_start, j_end, device=q.device).view(1, 1, 1, Tk)
                
                # Mask is True where j_idx > i_idx
                mask = j_idx > i_idx
                
                # Set scores to -inf where masked
                scores = scores.masked_fill(mask, float('-inf'))
            
            # Determine allowed keys for each row in the current query tile
            # A key is allowed if score is not -inf
            # We need to find max_allowed_j s[i,j] for each row i
            
            # If a row has no allowed keys (all -inf), we skip update for that row
            # Check if any score in the row is not -inf
            has_allowed = torch.any(scores != float('-inf'), dim=-1) # [B, H, Tq]
            
            # Compute max of allowed scores
            # For rows with no allowed keys, max will be -inf, but we handle that separately
            # We can use masked max or just max since -inf is the minimum
            # However, if all are -inf, max is -inf.
            
            # m_new = max(m_old, max_allowed_j s[i,j])
            # m_old is m_slice
            
            # Compute max of scores for each row
            # If a row is all -inf, max is -inf
            row_max_scores = torch.max(scores, dim=-1).values # [B, H, Tq]
            
            # m_new = max(m_slice, row_max_scores)
            m_new = torch.max(m_slice, row_max_scores)
            
            # alpha = exp(m_old - m_new)
            # If m_old is -inf and m_new is -inf, alpha is exp(nan) -> nan? 
            # But if m_old is -inf, it means no previous keys were seen.
            # If m_new is -inf, it means no keys in current tile are allowed either.
            # In this case, we should skip the update for that row.
            
            # Let's identify rows to update: has_allowed is True OR m_old is not -inf?
            # Actually, if has_allowed is False, m_new will be m_old (since max(m_old, -inf) = m_old).
            # If m_old is -inf and has_allowed is False, m_new is -inf.
            # alpha = exp(-inf - -inf) = exp(nan). This is problematic.
            
            # The rule says: "For an all-masked tile row, preserve m,l,a unchanged and skip the following update for that row."
            # So we only update rows where has_allowed is True.
            # Wait, what if has_allowed is False but m_old is not -inf? 
            # Then m_new = m_old. alpha = exp(0) = 1.
            # l_new = 1*l + 0 = l. a_new = 1*a + 0 = a.
            # So the update is a no-op. We can just skip the update for rows where has_allowed is False.
            
            # Let's create a mask for rows to update
            update_mask = has_allowed # [B, H, Tq]
            
            # Compute alpha for all rows, but we will only use it for update_mask
            # To avoid nan, we can compute alpha safely.
            # If m_old is -inf, we can set alpha to 0? No, if m_old is -inf and m_new is finite, alpha should be 0.
            # exp(-inf - finite) = 0. This is correct.
            # If m_old is finite and m_new is finite, alpha is finite.
            # If m_old is -inf and m_new is -inf, this case is handled by update_mask=False.
            
            # So we can compute alpha = exp(m_slice - m_new)
            # For rows where m_slice is -inf and m_new is finite, alpha = 0.
            # For rows where m_slice is finite and m_new is finite, alpha > 0.
            # For rows where m_slice is -inf and m_new is -inf, alpha is nan, but we don't use it.
            
            # To be safe, we can replace -inf in m_slice with 0 before subtraction? No.
            # Let's just compute it and mask later.
            
            # However, torch.exp(-inf - -inf) is nan. 
            # We can use torch.where to handle the -inf case.
            # If m_slice is -inf, alpha should be 0 (if m_new is finite) or 0 (if we skip).
            # Actually, if m_slice is -inf, it means l=0 and a=0. So alpha * l = 0 and alpha * a = 0.
            # So we can set alpha to 0 when m_slice is -inf.
            
            # Let's compute alpha safely
            # alpha = exp(m_slice - m_new)
            # If m_slice is -inf, set alpha to 0.
            
            # We can do: alpha = torch.exp(m_slice - m_new)
            # Then alpha = torch.where(m_slice == float('-inf'), torch.zeros_like(alpha), alpha)
            
            # But wait, if m_slice is -inf and m_new is -inf, we skip anyway.
            # If m_slice is -inf and m_new is finite, alpha should be 0.
            
            # Let's compute alpha
            # To avoid nan, we can clamp m_slice to a large negative number? No.
            # Let's use the where trick.
            
            # Compute diff = m_slice - m_new
            # If m_slice is -inf, diff is -inf (if m_new finite) or nan (if m_new -inf).
            # exp(-inf) = 0. exp(nan) = nan.
            
            # So we need to handle the nan case.
            
            # Let's compute alpha only for rows where update_mask is True.
            # For those rows, m_new is finite (because has_allowed is True, so row_max_scores is finite, so m_new is finite).
            # m_slice might be -inf or finite.
            # If m_slice is -inf, alpha = 0.
            # If m_slice is finite, alpha = exp(m_slice - m_new).
            
            # So we can compute alpha for all rows, but replace nan with 0.
            
            # Let's do:
            # alpha = torch.exp(m_slice - m_new)
            # alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            # This should work. If m_slice is -inf and m_new is finite, exp(-inf) = 0. No nan.
            # If m_slice is -inf and m_new is -inf, exp(nan) = nan. nan_to_num makes it 0.
            # If m_slice is finite and m_new is finite, exp(finite) is finite.
            
            # So:
            alpha = torch.exp(m_slice - m_new)
            alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Compute p[i,j] = exp(s[i,j] - m_new[i]) for allowed keys, else 0
            # s[i,j] is scores
            # m_new[i] is m_new[:, :, i, None]
            
            # p = exp(scores - m_new[:, :, :, None])
            # But for masked keys (scores = -inf), p should be 0.
            # exp(-inf - finite) = 0. So this works naturally.
            
            # However, if m_new is -inf (which shouldn't happen for update_mask=True rows), exp(finite - -inf) = inf.
            # But for update_mask=True rows, m_new is finite.
            
            # So we can compute p for all rows, but we will only use it for update_mask=True rows.
            
            # p = torch.exp(scores - m_new[:, :, :, None])
            # p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            
            # But wait, if scores is -inf and m_new is finite, p is 0. Correct.
            # If scores is finite and m_new is finite, p is finite. Correct.
            
            # Let's compute p
            p = torch.exp(scores - m_new[:, :, :, None])
            p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Now, we need to update l and a for rows where update_mask is True.
            
            # l_new = alpha * l_old + sum_j p
            # a_new = alpha * a_old + sum_j p[i,j] * v[j,:]
            
            # sum_j p is sum over the last dimension (Tk)
            # l_sum = torch.sum(p, dim=-1) # [B, H, Tq]
            
            # a_sum = torch.matmul(p, v_slice) # [B, H, Tq, D]
            
            # l_new = alpha * l_slice + l_sum
            # a_new = alpha * a_slice + a_sum
            
            # But we only update rows where update_mask is True.
            # For other rows, l and a remain unchanged.
            
            # We can use torch.where to select between new and old values.
            
            # l_new_all = alpha * l_slice + l_sum
            # a_new_all = alpha * a_slice + a_sum
            
            # l_slice = torch.where(update_mask, l_new_all, l_slice)
            # a_slice = torch.where(update_mask, a_new_all, a_slice)
            
            # m_slice = torch.where(update_mask, m_new, m_slice)
            
            # Let's compute l_sum and a_sum
            l_sum = torch.sum(p, dim=-1) # [B, H, Tq]
            a_sum = torch.matmul(p, v_slice) # [B, H, Tq, D]
            
            # Compute new l and a
            l_new_all = alpha * l_slice + l_sum
            a_new_all = alpha[..., None] * a_slice + a_sum
            
            # Update m, l, a for rows where update_mask is True
            # Note: m_new is already computed as max(m_slice, row_max_scores)
            # For rows where update_mask is False, m_new might be m_slice (if m_slice > -inf) or -inf (if m_slice is -inf).
            # But we don't want to update m for rows where update_mask is False.
            # So we use torch.where.
            
            m_slice = torch.where(update_mask, m_new, m_slice)
            l_slice = torch.where(update_mask, l_new_all, l_slice)
            a_slice = torch.where(update_mask[..., None], a_new_all, a_slice)
            
            # Update the global state
            m[:, :, i_start:i_end] = m_slice
            l[:, :, i_start:i_end] = l_slice
            a[:, :, i_start:i_end, :] = a_slice
    
    # Finalize
    # y = a / l
    # lse = m + log(l)
    
    # l is [B, H, N], a is [B, H, N, D]
    # y = a / l[:, :, :, None]
    
    # lse = m + torch.log(l)
    
    # Since l > 0 for all rows (as per problem statement), log(l) is safe.
    
    y = a / l[:, :, :, None]
    lse = m + torch.log(l)
    
    return y, lse