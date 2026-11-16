import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, N=4):
        super().__init__()
        if not isinstance(N, int) or isinstance(N, bool):
            raise ValueError("N must be an integer")
        if N < 1 or N > 16:
            raise ValueError("N must be between 1 and 16")
        self.N = N
        A = torch.zeros(N, N, dtype=torch.float64)
        B = torch.zeros(N, dtype=torch.float64)
        for n in range(N):
            A[n, n] = -(n + 1)
            for m in range(n):
                A[n, m] = -torch.sqrt(torch.tensor((2 * n + 1) * (2 * m + 1), dtype=torch.float64))
            B[n] = torch.sqrt(torch.tensor(2 * n + 1, dtype=torch.float64))
        self.register_buffer('A', A)
        self.register_buffer('B', B)

    def forward(self, u, initial_state=None, offset=0):
        if not isinstance(u, torch.Tensor):
            raise ValueError("u must be a torch.Tensor")
        if u.dtype != torch.float64:
            raise ValueError("u must be float64")
        if u.device.type != 'cpu':
            raise ValueError("u must be on cpu")
        if u.dim() != 3:
            raise ValueError("u must have shape [B, L, H]")
        B, L, H = u.shape
        if B <= 0 or L <= 0 or H <= 0:
            raise ValueError("u dimensions must be positive")
        if not torch.isfinite(u).all():
            raise ValueError("u must be finite")
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise ValueError("offset must be a nonnegative Python integer")
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        if initial_state is not None:
            if not isinstance(initial_state, torch.Tensor):
                raise ValueError("initial_state must be a torch.Tensor or None")
            if initial_state.dtype != torch.float64:
                raise ValueError("initial_state must be float64")
            if initial_state.device.type != 'cpu':
                raise ValueError("initial_state must be on cpu")
            if initial_state.shape != (B, H, self.N):
                raise ValueError("initial_state must have shape [B, H, N]")
            if not torch.isfinite(initial_state).all():
                raise ValueError("initial_state must be finite")

        N = self.N
        A = self.A
        Bv = self.B

        Ad = torch.zeros(L, N, N, dtype=torch.float64)
        Bd = torch.zeros(L, N, dtype=torch.float64)
        I = torch.eye(N, dtype=torch.float64)
        for j in range(L):
            k = offset + j + 1
            Q = I - A / (2.0 * k)
            Ad[j] = torch.linalg.solve(Q, I + A / (2.0 * k))
            Bd[j] = torch.linalg.solve(Q, Bv / k)

        if initial_state is None:
            c = torch.zeros(B, H, N, dtype=torch.float64)
        else:
            c = initial_state.clone()

        states = torch.zeros(B, L, H, N, dtype=torch.float64)
        for j in range(L):
            c_old = c
            c = torch.einsum('nm,bhm->bhn', Ad[j], c_old) + Bd[j].unsqueeze(0).unsqueeze(0) * u[:, j, :].unsqueeze(-1)
            states[:, j] = c

        final_state = c
        return {
            'states': states,
            'final_state': final_state,
            'Ad': Ad,
            'Bd': Bd
        }