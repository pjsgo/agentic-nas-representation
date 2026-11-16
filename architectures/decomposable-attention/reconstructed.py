import torch

def decompose(a, b, mask_a, mask_b, params):
    """
    Decomposable attention forward pass.

    Args:
        a: float64 [B, M, D]
        b: float64 [B, N, D]
        mask_a: bool [B, M]
        mask_b: bool [B, N]
        params: dict with fw1, fb1, fw2, fb2, gw1, gb1, gw2, gb2

    Returns:
        dict with keys: scores, attention_ab, attention_ba, aligned_b, aligned_a,
                        compare_a, compare_b, aggregate_a, aggregate_b, output
    """
    B, M, D = a.shape
    N = b.shape[1]

    # Validate masks: at least one valid token per sequence per batch item
    for i in range(B):
        if not mask_a[i].any():
            raise ValueError("mask_a has no valid tokens for batch item " + str(i))
        if not mask_b[i].any():
            raise ValueError("mask_b has no valid tokens for batch item " + str(i))

    # da_embed_mask: zero out invalid tokens
    a0 = torch.where(mask_a.unsqueeze(-1), a, torch.zeros_like(a))
    b0 = torch.where(mask_b.unsqueeze(-1), b, torch.zeros_like(b))

    # da_shared_F: F(x) = relu(relu(x @ fw1.T + fb1) @ fw2.T + fb2)
    fw1 = params['fw1']
    fb1 = params['fb1']
    fw2 = params['fw2']
    fb2 = params['fb2']

    def F(x):
        h = torch.relu(x @ fw1.t() + fb1)
        return torch.relu(h @ fw2.t() + fb2)

    fa = F(a0)  # [B, M, F]
    fb = F(b0)  # [B, N, F]

    # da_scores: e[q,i,j] = sum_f fa[q,i,f] * fb[q,j,f]
    # e: [B, M, N]
    e = torch.bmm(fa, fb.transpose(1, 2))  # [B, M, N]

    # scores: zero at invalid token pairs
    valid_pair = mask_a.unsqueeze(2) & mask_b.unsqueeze(1)  # [B, M, N]
    scores = torch.where(valid_pair, e, torch.zeros_like(e))

    # da_align: bidirectional masked softmax
    # attention_ab: for each valid a query i, softmax over valid b keys j
    # attention_ba: for each valid b query j, softmax over valid a keys i

    # For attention_ab: softmax over N dimension (b keys) for each a query
    # Mask out invalid b keys with -inf
    neg_inf = torch.tensor(float('-inf'), dtype=torch.float64)
    mask_b_expanded = mask_b.unsqueeze(1)  # [B, 1, N]
    e_masked_ab = torch.where(mask_b_expanded, e, neg_inf)  # [B, M, N]

    # For valid a queries, compute softmax over valid b keys
    # For invalid a queries, we want zero rows
    # First compute max over valid b keys for numerical stability
    # max over N dim, but only valid b keys
    max_ab = e_masked_ab.max(dim=2, keepdim=True).values  # [B, M, 1]
    # For invalid a queries, max_ab might be -inf; handle that
    # exp(e - max)
    exp_ab = torch.exp(e_masked_ab - max_ab)  # [B, M, N]
    # For invalid b keys, exp_ab is 0 (exp(-inf - something) = 0)
    # Sum over valid b keys
    sum_ab = exp_ab.sum(dim=2, keepdim=True)  # [B, M, 1]
    # attention_ab = exp_ab / sum_ab, but zero out invalid a rows and invalid b columns
    attention_ab = exp_ab / sum_ab  # [B, M, N]
    # Zero out invalid a rows
    attention_ab = torch.where(mask_a.unsqueeze(2), attention_ab, torch.zeros_like(attention_ab))
    # Zero out invalid b columns (should already be 0 from exp, but ensure)
    attention_ab = torch.where(mask_b.unsqueeze(1), attention_ab, torch.zeros_like(attention_ab))

    # For attention_ba: softmax over M dimension (a keys) for each b query
    mask_a_expanded = mask_a.unsqueeze(1)  # [B, 1, M]
    # e is [B, M, N], we need to transpose for b queries: e_ba[q,j,i] = e[q,i,j]
    e_t = e.transpose(1, 2)  # [B, N, M]
    e_masked_ba = torch.where(mask_a_expanded, e_t, neg_inf)  # [B, N, M]
    max_ba = e_masked_ba.max(dim=2, keepdim=True).values  # [B, N, 1]
    exp_ba = torch.exp(e_masked_ba - max_ba)  # [B, N, M]
    sum_ba = exp_ba.sum(dim=2, keepdim=True)  # [B, N, 1]
    attention_ba = exp_ba / sum_ba  # [B, N, M]
    # Zero out invalid b rows
    attention_ba = torch.where(mask_b.unsqueeze(2), attention_ba, torch.zeros_like(attention_ba))
    # Zero out invalid a columns
    attention_ba = torch.where(mask_a.unsqueeze(1), attention_ba, torch.zeros_like(attention_ba))

    # aligned_b = attention_ab @ b0: [B, M, N] @ [B, N, D] -> [B, M, D]
    aligned_b = torch.bmm(attention_ab, b0)
    # aligned_a = attention_ba @ a0: [B, N, M] @ [B, M, D] -> [B, N, D]
    aligned_a = torch.bmm(attention_ba, a0)

    # da_shared_G: G(x) = relu(relu(x @ gw1.T + gb1) @ gw2.T + gb2)
    gw1 = params['gw1']
    gb1 = params['gb1']
    gw2 = params['gw2']
    gb2 = params['gb2']

    def G(x):
        h = torch.relu(x @ gw1.t() + gb1)
        return torch.relu(h @ gw2.t() + gb2)

    # compare_a = where(mask_a, G(concat(a0, aligned_b, dim=2)), 0)
    concat_a = torch.cat([a0, aligned_b], dim=2)  # [B, M, 2D]
    compare_a_raw = G(concat_a)  # [B, M, G]
    compare_a = torch.where(mask_a.unsqueeze(-1), compare_a_raw, torch.zeros_like(compare_a_raw))

    # compare_b = where(mask_b, G(concat(b0, aligned_a, dim=2)), 0)
    concat_b = torch.cat([b0, aligned_a], dim=2)  # [B, N, 2D]
    compare_b_raw = G(concat_b)  # [B, N, G]
    compare_b = torch.where(mask_b.unsqueeze(-1), compare_b_raw, torch.zeros_like(compare_b_raw))

    # da_aggregate: sum over token dimension
    aggregate_a = compare_a.sum(dim=1)  # [B, G]
    aggregate_b = compare_b.sum(dim=1)  # [B, G]
    output = torch.cat([aggregate_a, aggregate_b], dim=1)  # [B, 2G]

    return {
        'scores': scores,
        'attention_ab': attention_ab,
        'attention_ba': attention_ba,
        'aligned_b': aligned_b,
        'aligned_a': aligned_a,
        'compare_a': compare_a,
        'compare_b': compare_b,
        'aggregate_a': aggregate_a,
        'aggregate_b': aggregate_b,
        'output': output,
    }
