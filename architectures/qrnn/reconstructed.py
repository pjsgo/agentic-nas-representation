import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, I=3, H=4, window=2):
        super().__init__()
        self.I = I
        self.H = H
        self.window = window
        
        # Parameters as specified in YAML
        # W: [3*H, window*I]
        # b: [3*H]
        self.W = nn.Parameter(torch.zeros(3 * H, window * I, dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(3 * H, dtype=torch.float64))

    def forward(self, x, c0=None, prev_x=None):
        B, T, I = x.shape
        H = self.H
        window = self.window
        
        # Ensure dtype is float64 as per interface specification
        x = x.to(torch.float64)
        if c0 is not None:
            c0 = c0.to(torch.float64)
        if prev_x is not None:
            prev_x = prev_x.to(torch.float64)

        # Initialize c0 if None
        if c0 is None:
            c0 = torch.zeros(B, H, dtype=torch.float64, device=x.device)
        
        # Initialize prev_x if None
        if prev_x is None:
            prev_x = torch.zeros(B, I, dtype=torch.float64, device=x.device)

        # Pre-allocate outputs
        z = torch.empty(B, T, H, dtype=torch.float64, device=x.device)
        update_gate = torch.empty(B, T, H, dtype=torch.float64, device=x.device)
        output_gate = torch.empty(B, T, H, dtype=torch.float64, device=x.device)
        cells = torch.empty(B, T, H, dtype=torch.float64, device=x.device)
        output = torch.empty(B, T, H, dtype=torch.float64, device=x.device)

        old = c0.clone() # old state for recurrence

        for t in range(T):
            # 1. window_affine
            if window == 1:
                v = x[:, t, :] # [B, I]
            else: # window == 2
                if t == 0:
                    prev = prev_x # [B, I]
                else:
                    prev = x[:, t-1, :] # [B, I]
                v = torch.cat([x[:, t, :], prev], dim=1) # [B, 2*I]
            
            # a[b,t,j] = sum_i W[j,i]*v[b,i] + b[j]
            # v is [B, window*I], W is [3*H, window*I]
            # We need a [B, 3*H]
            # a = v @ W.T + b
            a = torch.matmul(v, self.W.t()) + self.b # [B, 3*H]

            # 2. gates
            # z = tanh(a[:, :H])
            # update_gate = sigmoid(a[:, H:2*H])
            # output_gate = sigmoid(a[:, 2*H:3*H])
            
            z_t = torch.tanh(a[:, :H])
            update_gate_t = torch.sigmoid(a[:, H:2*H])
            output_gate_t = torch.sigmoid(a[:, 2*H:3*H])
            
            z[:, t, :] = z_t
            update_gate[:, t, :] = update_gate_t
            output_gate[:, t, :] = output_gate_t

            # 3. pooling
            # new = update_gate[:,t]*z[:,t] + (1-update_gate[:,t])*old
            # cells[:,t] = new
            # output[:,t] = output_gate[:,t]*new
            # old = new
            
            new = update_gate_t * z_t + (1 - update_gate_t) * old
            cells[:, t, :] = new
            output[:, t, :] = output_gate_t * new
            old = new

        # 4. returns
        final_cell = cells[:, -1, :]
        final_output = output[:, -1, :]
        final_x = x[:, -1, :]

        return {
            'z': z,
            'update_gate': update_gate,
            'output_gate': output_gate,
            'cells': cells,
            'output': output,
            'final_cell': final_cell,
            'final_output': final_output,
            'final_x': final_x
        }