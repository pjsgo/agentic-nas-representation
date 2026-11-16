import torch


def _validate_inputs(tokens, h0, A, R, O):
    # Validate that all inputs are tensors
    for name, t in [('tokens', tokens), ('h0', h0), ('A', A), ('R', R), ('O', O)]:
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")

    # Validate tokens
    if tokens.dtype != torch.int64:
        raise ValueError("tokens must be int64")
    if tokens.device.type != 'cpu':
        raise ValueError("tokens must be on CPU")
    if tokens.dim() != 2:
        raise ValueError("tokens must be 2D [T, B]")
    T, B = tokens.shape
    if T == 0 or B == 0:
        raise ValueError("tokens dimensions must be positive")

    # Validate h0
    if h0.dtype != torch.float64:
        raise ValueError("h0 must be float64")
    if h0.device.type != 'cpu':
        raise ValueError("h0 must be on CPU")
    if h0.dim() != 2:
        raise ValueError("h0 must be 2D [B, D]")
    if h0.shape[0] != B:
        raise ValueError("h0 batch dimension must match tokens batch dimension")
    D = h0.shape[1]
    if D == 0:
        raise ValueError("h0 hidden dimension must be positive")
    if not torch.isfinite(h0).all():
        raise ValueError("h0 must contain only finite values")

    # Validate A
    if A.dtype != torch.float64:
        raise ValueError("A must be float64")
    if A.device.type != 'cpu':
        raise ValueError("A must be on CPU")
    if A.dim() != 2:
        raise ValueError("A must be 2D [V, D]")
    V, D_A = A.shape
    if D_A != D:
        raise ValueError("A hidden dimension must match h0 hidden dimension")
    if V == 0 or D_A == 0:
        raise ValueError("A dimensions must be positive")
    if not torch.isfinite(A).all():
        raise ValueError("A must contain only finite values")

    # Validate R
    if R.dtype != torch.float64:
        raise ValueError("R must be float64")
    if R.device.type != 'cpu':
        raise ValueError("R must be on CPU")
    if R.dim() != 2:
        raise ValueError("R must be 2D [D, D]")
    D_R1, D_R2 = R.shape
    if D_R1 != D or D_R2 != D:
        raise ValueError("R dimensions must match hidden dimension D")
    if D_R1 == 0 or D_R2 == 0:
        raise ValueError("R dimensions must be positive")
    if not torch.isfinite(R).all():
        raise ValueError("R must contain only finite values")

    # Validate O
    if O.dtype != torch.float64:
        raise ValueError("O must be float64")
    if O.device.type != 'cpu':
        raise ValueError("O must be on CPU")
    if O.dim() != 2:
        raise ValueError("O must be 2D [D, V]")
    D_O, V_O = O.shape
    if D_O != D or V_O != V:
        raise ValueError("O dimensions must match hidden dimension D and vocabulary size V")
    if D_O == 0 or V_O == 0:
        raise ValueError("O dimensions must be positive")
    if not torch.isfinite(O).all():
        raise ValueError("O must contain only finite values")

    # Validate token values
    if tokens.min() < 0 or tokens.max() >= V:
        raise ValueError("token IDs must be in [0, V)")

    return T, B, V, D


def rnn_lm(tokens, h0, A, R, O):
    T, B, V, D = _validate_inputs(tokens, h0, A, R, O)

    # Pre-allocate outputs
    hidden = torch.empty((T, B, D), dtype=torch.float64, device='cpu')
    logits = torch.empty((T, B, V), dtype=torch.float64, device='cpu')
    probabilities = torch.empty((T, B, V), dtype=torch.float64, device='cpu')

    # Initial state
    s = h0.clone()  # [B, D]

    for t in range(T):
        # Token map: input_term[t, a, h] = A[tokens[t, a], h]
        # tokens[t] is [B], A is [V, D]
        # Gather rows of A based on tokens[t]
        input_term = A[tokens[t]]  # [B, D]

        # Recurrent: pre[t, a, h] = input_term[t, a, h] + sum_j s[t-1, a, j] * R[j, h]
        # s is [B, D], R is [D, D]
        # s @ R gives [B, D] where (s @ R)[a, h] = sum_j s[a, j] * R[j, h]
        recurrent_term = s @ R  # [B, D]
        pre = input_term + recurrent_term  # [B, D]

        # Sigmoid: s[t, a, h] = 1 / (1 + exp(-pre[t, a, h]))
        # Numerically stable sigmoid
        s = torch.sigmoid(pre)  # [B, D]

        # Store hidden state
        hidden[t] = s

        # Readout: logits[t, a, v] = sum_h s[t, a, h] * O[h, v]
        # s is [B, D], O is [D, V]
        # s @ O gives [B, V] where (s @ O)[a, v] = sum_h s[a, h] * O[h, v]
        logits[t] = s @ O  # [B, V]

        # Softmax: stable softmax over vocabulary dimension
        # m[t, a] = max_v logits[t, a, v]
        # probabilities[t, a, v] = exp(logits[t, a, v] - m[t, a]) / sum_u exp(logits[t, a, u] - m[t, a])
        logits_t = logits[t]  # [B, V]
        max_logits = torch.max(logits_t, dim=1, keepdim=True).values  # [B, 1]
        exp_logits = torch.exp(logits_t - max_logits)  # [B, V]
        sum_exp = torch.sum(exp_logits, dim=1, keepdim=True)  # [B, 1]
        probabilities[t] = exp_logits / sum_exp  # [B, V]

    # Final state: independent copy of s[T-1]
    final_state = s.clone()  # [B, D]

    return {
        'hidden': hidden,
        'logits': logits,
        'probabilities': probabilities,
        'final_state': final_state
    }
