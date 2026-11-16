import torch

def switch_ffn(x, router, wi, wo, nonpadding, capacity_factor, min_capacity):
    T, D = x.shape
    E = router.shape[1]
    H = wi.shape[2]
    O = wo.shape[2]

    # 1. Router
    logits = x @ router
    max_logits = torch.max(logits, dim=1, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)
    sum_exp = torch.sum(exp_logits, dim=1, keepdim=True)
    probabilities = exp_logits / sum_exp
    
    # Argmax with tie-breaking to lower index
    indices = torch.argmax(logits, dim=1)
    selected_gate = probabilities[torch.arange(T), indices]

    # 2. Capacity and Dispatch
    # C = max(min_capacity, min(T, floor(T * capacity_factor / E)))
    C = max(min_capacity, min(T, int(T * capacity_factor / E)))
    
    positions = torch.full((T,), -1, dtype=torch.int64, device=x.device)
    accepted = torch.zeros(T, dtype=torch.bool, device=x.device)
    
    # Count valid tokens per expert to determine positions
    # We need to assign positions in token order (t=0 to T-1)
    # For each expert, we maintain a counter of how many valid tokens have been seen so far.
    
    # Efficient vectorized approach for positions:
    # Create a tensor of indices for valid tokens
    valid_mask = nonpadding
    
    # We can compute positions by iterating or using cumulative sums on masked indices.
    # Since T is likely not huge in this context, but let's try to be reasonably efficient.
    # However, pure vectorization of "rank within group by order" is tricky without loops or advanced indexing.
    # Let's use a loop for clarity and correctness, as T is the sequence length.
    
    expert_counts = torch.zeros(E, dtype=torch.int64, device=x.device)
    
    for t in range(T):
        if valid_mask[t]:
            e = indices[t].item()
            pos = expert_counts[e].item()
            positions[t] = pos
            if pos < C:
                accepted[t] = True
            expert_counts[e] += 1
        else:
            positions[t] = -1
            accepted[t] = False

    # Build dispatch tensor [T, E, C]
    dispatch = torch.zeros(T, E, C, dtype=torch.float64, device=x.device)
    combine = torch.zeros(T, E, C, dtype=torch.float64, device=x.device)
    
    for t in range(T):
        if accepted[t]:
            e = indices[t].item()
            c = positions[t].item()
            dispatch[t, e, c] = 1.0
            combine[t, e, c] = selected_gate[t].item()

    # 3. Experts
    # expert_inputs[e, c, d] = sum_t dispatch[t, e, c] * x[t, d]
    # This is equivalent to: for each expert e and slot c, gather the x[t] where dispatch[t,e,c]==1
    # Since dispatch is one-hot in t for each (e,c) (at most one t per e,c), we can just index.
    
    expert_inputs = torch.zeros(E, C, D, dtype=torch.float64, device=x.device)
    
    # We can vectorize this by using the dispatch tensor.
    # dispatch is [T, E, C]. We want to sum over T.
    # expert_inputs[e, c, d] = sum_t dispatch[t, e, c] * x[t, d]
    # This is a batched matrix multiplication or einsum.
    # dispatch: [T, E, C] -> permute to [E, C, T]
    # x: [T, D]
    # result: [E, C, D]
    
    dispatch_perm = dispatch.permute(1, 2, 0) # [E, C, T]
    expert_inputs = torch.bmm(dispatch_perm, x.unsqueeze(0).expand(E, T, D))
    # Wait, x is [T, D]. We need to broadcast x to [E, T, D].
    # bmm requires [B, N, M] and [B, M, K].
    # dispatch_perm is [E, C, T]. x is [T, D].
    # We can do: torch.einsum('tec,td->ecd', dispatch, x)
    
    expert_inputs = torch.einsum('tec,td->ecd', dispatch, x)

    # hidden[e, c, h] = max(0, sum_d expert_inputs[e, c, d] * wi[e, d, h])
    # wi is [E, D, H]
    # expert_inputs is [E, C, D]
    # We need to multiply expert_inputs[e, c, d] with wi[e, d, h] and sum over d.
    # This is a batched matmul where the batch is E, and the matrices are [C, D] and [D, H].
    
    # Reshape expert_inputs to [E, C, D]
    # Reshape wi to [E, D, H]
    # bmm(expert_inputs, wi) -> [E, C, H]
    
    hidden = torch.bmm(expert_inputs, wi)
    hidden = torch.relu(hidden)

    # expert_outputs[e, c, o] = sum_h hidden[e, c, h] * wo[e, h, o]
    # wo is [E, H, O]
    # bmm(hidden, wo) -> [E, C, O]
    
    expert_outputs = torch.bmm(hidden, wo)

    # 4. Combine
    # output[t, o] = sum_{e, c} combine[t, e, c] * expert_outputs[e, c, o]
    # combine is [T, E, C]
    # expert_outputs is [E, C, O]
    # We need to sum over e and c.
    # This is an einsum: 'tec,eco->to'
    
    output = torch.einsum('tec,eco->to', combine, expert_outputs)

    return {
        "capacity": C,
        "logits": logits,
        "probabilities": probabilities,
        "indices": indices,
        "selected_gate": selected_gate,
        "positions": positions,
        "accepted": accepted,
        "dispatch": dispatch,
        "combine": combine,
        "expert_inputs": expert_inputs,
        "expert_outputs": expert_outputs,
        "output": output
    }