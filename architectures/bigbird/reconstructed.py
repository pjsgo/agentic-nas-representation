import torch

def attention(q, k, v, random_blocks, block_size):
    B, H, N, D = q.shape
    S = block_size
    M = N // S

    # 1. Define block indices for each query block i
    # i ranges from 0 to M-1
    # For each i, we need the set of key blocks A_h(i)

    # Local blocks L(i) = {j in [0, M) : abs(i - j) <= 1}
    # Global blocks G(i) = [0, M) if i in {0, M-1} else {0, M-1}
    # Random blocks R_h(i) = random_blocks[h, i-1, :] if i not in {0, M-1} else empty

    # We will compute the attention for each query token t.
    # t belongs to block i = t // S.
    # The set of valid key tokens T_h(t) is the union of tokens in blocks A_h(i).

    # Since the set of valid key blocks depends only on the query block i and head h,
    # we can precompute the mask for each (h, i) pair.

    # Mask shape: [H, M, N] where mask[h, i, u] is 1 if key token u is in T_h(t) for any t in block i.
    # Actually, T_h(t) is the same for all t in block i. So we can define a boolean mask for each (h, i).

    # Let's create a tensor `valid_key_blocks` of shape [H, M, M] where valid_key_blocks[h, i, j] is True if block j is in A_h(i).

    valid_key_blocks = torch.zeros((H, M, M), dtype=torch.bool, device=q.device)

    for i in range(M):
        # Local blocks
        for j in range(max(0, i - 1), min(M, i + 2)):
            valid_key_blocks[:, i, j] = True

        # Global blocks
        if i == 0 or i == M - 1:
            valid_key_blocks[:, i, :] = True
        else:
            valid_key_blocks[:, i, 0] = True
            valid_key_blocks[:, i, M - 1] = True

        # Random blocks
        if i != 0 and i != M - 1:
            # random_blocks has shape [H, M-2, R]
            # The row index for query block i is i - 1
            rand_blocks = random_blocks[:, i - 1, :] # Shape [H, R]
            # Set valid_key_blocks[h, i, rand_blocks[h, r]] = True for all h, r
            # We can use advanced indexing
            # rand_blocks is [H, R]. We want to set valid_key_blocks[h, i, rand_blocks[h, r]] = True.
            # Let's expand rand_blocks to [H, R, 1] and use it to index the last dimension.
            # But we need to do this for each i.
            # Since i is fixed in the loop, we can do:
            # indices = rand_blocks # [H, R]
            # valid_key_blocks[:, i, indices] = True # This might not work directly with 2D indices on 3D tensor slice.

            # Let's use scatter or manual loop if R is small, but vectorized is better.
            # valid_key_blocks[:, i, :] is a tensor of shape [H, M].
            # We want to set specific columns to True.
            # We can create a mask for the columns.
            col_mask = torch.zeros((H, M), dtype=torch.bool, device=q.device)
            # rand_blocks is [H, R]. We want to set col_mask[h, rand_blocks[h, r]] = True.
            # We can use torch.scatter_ or just index.
            # Let's use advanced indexing:
            # h_indices = torch.arange(H).unsqueeze(1).expand(H, R)
            # r_indices = rand_blocks
            # col_mask[h_indices, r_indices] = True
            h_indices = torch.arange(H, device=q.device).unsqueeze(1).expand(H, -1) # [H, R]
            col_mask[h_indices, rand_blocks] = True
            valid_key_blocks[:, i, :] |= col_mask

    # Now we have valid_key_blocks [H, M, M].
    # We need to expand this to token level.
    # For a query token t in block i, the valid key tokens are those in blocks j where valid_key_blocks[h, i, j] is True.
    # This means for key token u in block j = u // S, it is valid if valid_key_blocks[h, i, j] is True.

    # Let's create a mask for each query token t and key token u.
    # mask[t, u] = valid_key_blocks[h, t // S, u // S]

    # We can create a tensor `query_block_indices` of shape [N] where query_block_indices[t] = t // S.
    # And `key_block_indices` of shape [N] where key_block_indices[u] = u // S.

    query_block_indices = torch.arange(N, device=q.device) // S # [N]
    key_block_indices = torch.arange(N, device=q.device) // S   # [N]

    # We want a mask of shape [H, N, N] where mask[h, t, u] = valid_key_blocks[h, query_block_indices[t], key_block_indices[u]]

    # valid_key_blocks is [H, M, M].
    # We can index it with query_block_indices and key_block_indices.
    # query_block_indices is [N]. We need to expand it to [1, N, 1] to index the second dimension.
    # key_block_indices is [N]. We need to expand it to [1, 1, N] to index the third dimension.

    # Let's use advanced indexing.
    # h_indices = torch.arange(H).view(H, 1, 1)
    # q_indices = query_block_indices.view(1, N, 1)
    # k_indices = key_block_indices.view(1, 1, N)
    # mask = valid_key_blocks[h_indices, q_indices, k_indices] # This results in [H, N, N]

    h_indices = torch.arange(H, device=q.device).view(H, 1, 1)
    q_indices = query_block_indices.view(1, N, 1)
    k_indices = key_block_indices.view(1, 1, N)

    # valid_key_blocks is [H, M, M]
    # h_indices is [H, 1, 1]
    # q_indices is [1, N, 1]
    # k_indices is [1, 1, N]
    # The result of valid_key_blocks[h_indices, q_indices, k_indices] will be [H, N, N]

    mask = valid_key_blocks[h_indices, q_indices, k_indices] # [H, N, N]

    # Now we compute the attention scores.
    # q: [B, H, N, D]
    # k: [B, H, N, D]
    # v: [B, H, N, D]

    # scores = q @ k^T / sqrt(D)
    # q is [B, H, N, D], k is [B, H, N, D].
    # We need to transpose k to [B, H, D, N].

    k_transposed = k.transpose(-2, -1) # [B, H, D, N]
    scores = torch.matmul(q, k_transposed) # [B, H, N, N]

    # Scale by sqrt(D)
    scores = scores / (D ** 0.5)

    # Apply mask.
    # mask is [H, N, N]. We need to expand it to [B, H, N, N].
    # mask_expanded = mask.unsqueeze(0).expand(B, H, N, N)

    # Set scores to -inf where mask is False.
    # We can use torch.where or add a large negative number.
    # Using -inf is standard for softmax.

    # mask is boolean. We want to keep scores where mask is True, and set to -inf where False.
    # scores = torch.where(mask_expanded, scores, torch.tensor(float('-inf'), device=q.device))

    # However, we must be careful with the dtype. q, k, v are float64.
    # Let's use a large negative number to avoid potential issues with -inf in some backends, though -inf is usually fine.
    # But the YAML says "softmax_over_connected_keys". Standard softmax handles -inf by setting probability to 0.

    # Let's expand mask to [1, H, N, N] to broadcast with [B, H, N, N].
    mask_expanded = mask.unsqueeze(0) # [1, H, N, N]

    # Create a tensor of -inf with the same shape as scores.
    neg_inf = torch.tensor(float('-inf'), device=q.device, dtype=q.dtype)

    # Apply mask
    scores = torch.where(mask_expanded, scores, neg_inf)

    # Compute softmax over the last dimension (key tokens).
    # The YAML says "One joint normalization across all three paths".
    # This is exactly what softmax over the connected keys does.

    # We need to be careful with rows that might be all -inf (if no keys are connected, which shouldn't happen given the constraints).
    # The constraints say M>=8, and local/global blocks ensure at least some keys are connected.

    # torch.softmax handles -inf correctly by setting the corresponding probability to 0.
    # However, if a row is all -inf, softmax will produce NaNs. But this case is excluded by the problem constraints.

    probabilities = torch.softmax(scores, dim=-1) # [B, H, N, N]

    # Compute output.
    # y = probabilities @ v
    # probabilities: [B, H, N, N]
    # v: [B, H, N, D]

    output = torch.matmul(probabilities, v) # [B, H, N, D]

    return output
