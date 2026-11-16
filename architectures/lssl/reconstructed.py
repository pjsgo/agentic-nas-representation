import torch
import torch.nn as nn
import math

def _hippo_legs(state_size, dtype):
    A = torch.zeros(state_size, state_size, dtype=dtype)
    B = torch.zeros(state_size, dtype=dtype)
    for i in range(state_size):
        A[i, i] = -(i + 1)
        B[i] = math.sqrt(2 * i + 1)
        for j in range(i):
            A[i, j] = -math.sqrt((2 * i + 1) * (2 * j + 1))
    return A, B

class LSSLCore(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.heads = 2
        self.state_size = 4

        self.log_dt = nn.Parameter(values['log_dt'].clone())
        self.C = nn.Parameter(values['C'].clone())
        self.D = nn.Parameter(values['D'].clone())

        A, B = _hippo_legs(self.state_size, torch.float64)
        self.register_buffer('A', A)
        self.register_buffer('B', B)

    def forward(self, u, initial_state):
        dt = torch.exp(self.log_dt)

        I = torch.eye(self.state_size, dtype=torch.float64)

        Ad_list = []
        Bd_list = []
        for h in range(self.heads):
            dt_h = dt[h]
            M = I - dt_h * self.A / 2
            R_ad = I + dt_h * self.A / 2
            R_bd = dt_h * self.B
            Ad_h = torch.linalg.solve(M, R_ad)
            Bd_h = torch.linalg.solve(M, R_bd)
            Ad_list.append(Ad_h)
            Bd_list.append(Bd_h)

        Ad = torch.stack(Ad_list, dim=0)
        Bd = torch.stack(Bd_list, dim=0)

        batch, time, heads = u.shape
        states = torch.empty(batch, time, heads, self.state_size, dtype=torch.float64)

        x_prev = initial_state
        for t in range(time):
            u_t = u[:, t, :]
            x_t = torch.empty(batch, heads, self.state_size, dtype=torch.float64)
            for h in range(heads):
                x_t[:, h, :] = x_prev[:, h, :] @ Ad[h].transpose(0, 1) + u_t[:, h, None] * Bd[h]
            states[:, t, :, :] = x_t
            x_prev = x_t

        final_state = x_prev

        y = torch.empty(batch, time, heads, dtype=torch.float64)
        for t in range(time):
            for h in range(heads):
                y[:, t, h] = torch.sum(states[:, t, h, :] * self.C[h], dim=1) + self.D[h] * u[:, t, h]

        kernel = torch.empty(heads, time, dtype=torch.float64)
        for h in range(heads):
            Ad_h = Ad[h]
            Bd_h = Bd[h]
            Ad_power = torch.eye(self.state_size, dtype=torch.float64)
            for k in range(time):
                kernel[h, k] = self.C[h] @ (Ad_power @ Bd_h)
                Ad_power = Ad_power @ Ad_h

        return {
            'y': y,
            'final_state': final_state,
            'kernel': kernel
        }

def build(values):
    return LSSLCore(values)
