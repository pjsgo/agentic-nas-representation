import torch

def switch_ffn(x, router, wi, wo, nonpadding, capacity_factor, min_capacity):
    T, D = x.shape
    E = router.shape[1]
    H = wi.shape[2]
    O = wo.shape[2]

    # 1. Router
    # Ensure gradients flow through router
    logits = x @ router
    max_logits = torch.max(logits, dim=1, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)
    sum_exp = torch.sum(exp_logits, dim=1, keepdim=True)
    probabilities = exp_logits / sum_exp

    # Argmax with tie-breaking to lower index
    # Note: torch.argmax is not differentiable, but indices are used for discrete routing.
    # The gradient flows through 'selected_gate' which is derived from 'probabilities' (differentiable).
    indices = torch.argmax(logits, dim=1)

    # selected_gate[t] = probabilities[t, indices[t]]
    # This operation is differentiable with respect to probabilities (and thus router).
    # We use gather to ensure the graph is connected.
    selected_gate = probabilities[torch.arange(T), indices]

    # 2. Capacity and Dispatch
    # C = max(min_capacity, min(T, floor(T * capacity_factor / E)))
    C = max(min_capacity, min(T, int(T * capacity_factor / E)))

    positions = torch.full((T,), -1, dtype=torch.int64, device=x.device)
    accepted = torch.zeros(T, dtype=torch.bool, device=x.device)

    # Count valid tokens per expert to determine positions
    expert_counts = torch.zeros(E, dtype=torch.int64, device=x.device)

    for t in range(T):
        if nonpadding[t]:
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
    # dispatch is a constant mask (0 or 1) determined by routing decisions.
    # It does not require gradients itself, but it multiplies differentiable tensors.
    dispatch = torch.zeros(T, E, C, dtype=torch.float64, device=x.device)
    combine = torch.zeros(T, E, C, dtype=torch.float64, device=x.device)

    for t in range(T):
        if accepted[t]:
            e = indices[t].item()
            c = positions[t].item()
            dispatch[t, e, c] = 1.0
            # combine[t, e, c] = dispatch[t, e, c] * selected_gate[t]
            # selected_gate is a tensor leaf in the graph (or intermediate), so this preserves the graph.
            combine[t, e, c] = selected_gate[t]

    # 3. Experts
    # expert_inputs[e, c, d] = sum_t dispatch[t, e, c] * x[t, d]
    # dispatch is constant, x is input. If x requires grad, expert_inputs will too.
    # If x does not require grad, expert_inputs is constant.
    # The spec says x is input. Usually in training, x might be constant (from embeddings) or variable.
    # However, the failure was about router, wi, wo.

    # Using einsum to compute expert_inputs
    # dispatch: [T, E, C], x: [T, D] -> expert_inputs: [E, C, D]
    expert_inputs = torch.einsum('tec,td->ecd', dispatch, x)

    # hidden[e, c, h] = max(0, sum_d expert_inputs[e, c, d] * wi[e, d, h])
    # wi is [E, D, H]
    # bmm(expert_inputs, wi) -> [E, C, H]
    # This operation is differentiable with respect to wi and expert_inputs.
    hidden = torch.bmm(expert_inputs, wi)
    hidden = torch.relu(hidden)

    # expert_outputs[e, c, o] = sum_h hidden[e, c, h] * wo[e, h, o]
    # wo is [E, H, O]
    # bmm(hidden, wo) -> [E, C, O]
    # This operation is differentiable with respect to wo and hidden.
    expert_outputs = torch.bmm(hidden, wo)

    # 4. Combine
    # output[t, o] = sum_{e, c} combine[t, e, c] * expert_outputs[e, c, o]
    # combine contains selected_gate (differentiable w.r.t router).
    # expert_outputs is differentiable w.r.t wi, wo.
    # output is differentiable w.r.t router, wi, wo.

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
