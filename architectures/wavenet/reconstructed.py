import torch
import torch.nn as nn


class WaveNetStack(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, skip_channels: int, dilations: tuple[int, ...]):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.skip_channels = skip_channels
        self.dilations = tuple(dilations)
        L = len(self.dilations)
        H = hidden_channels
        C = channels
        S = skip_channels

        self.Wf = nn.Parameter(torch.zeros(L, H, C, 2, dtype=torch.float64))
        self.Wg = nn.Parameter(torch.zeros(L, H, C, 2, dtype=torch.float64))
        self.Wr = nn.Parameter(torch.zeros(L, C, H, 1, dtype=torch.float64))
        self.Ws = nn.Parameter(torch.zeros(L, S, H, 1, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(dtype=torch.float64)
        B, C, T = x.shape
        L = len(self.dilations)
        H = self.hidden_channels
        S = self.skip_channels

        states = torch.empty(L + 1, B, C, T, dtype=torch.float64, device=x.device)
        filter_pre = torch.empty(L, B, H, T, dtype=torch.float64, device=x.device)
        gate_pre = torch.empty(L, B, H, T, dtype=torch.float64, device=x.device)
        gated = torch.empty(L, B, H, T, dtype=torch.float64, device=x.device)
        skips = torch.empty(L, B, S, T, dtype=torch.float64, device=x.device)

        h = x
        states[0] = h

        for l in range(L):
            d = self.dilations[l]
            Wf_l = self.Wf[l]  # [H, C, 2]
            Wg_l = self.Wg[l]  # [H, C, 2]
            Wr_l = self.Wr[l]  # [C, H, 1]
            Ws_l = self.Ws[l]  # [S, H, 1]

            # Causal dilated cross-correlation for filter and gate
            # f[l,b,o,t] = sum_i (Wf[l,o,i,0]*h[l,b,i,t-d] + Wf[l,o,i,1]*h[l,b,i,t])
            # g[l,b,o,t] = sum_i (Wg[l,o,i,0]*h[l,b,i,t-d] + Wg[l,o,i,1]*h[l,b,i,t])

            # h is [B, C, T]
            # We need to compute for each output channel o, input channel i, time t
            # This is a convolution with kernel size 2, dilation d, stride 1
            # Using manual computation for clarity and correctness

            # For filter:
            # f[b, o, t] = sum_i Wf_l[o, i, 0] * h[b, i, t-d] + Wf_l[o, i, 1] * h[b, i, t]
            # where h[b, i, t-d] is 0 if t-d < 0

            # Create shifted versions of h
            # h_shifted[b, i, t] = h[b, i, t-d] if t-d >= 0 else 0
            h_shifted = torch.zeros_like(h)
            if d < T:
                h_shifted[:, :, d:] = h[:, :, :T - d]
            # h_current = h[:, :, :] (t index)

            # Now compute f and g using einsum or matmul
            # Wf_l is [H, C, 2], h_shifted is [B, C, T], h is [B, C, T]
            # f[b, o, t] = sum_i (Wf_l[o, i, 0] * h_shifted[b, i, t] + Wf_l[o, i, 1] * h[b, i, t])
            # = sum_i Wf_l[o, i, 0] * h_shifted[b, i, t] + sum_i Wf_l[o, i, 1] * h[b, i, t]

            # Each selected tap is [out, in]; preserve batch and time.
            f = torch.einsum('oc,bct->bot', Wf_l[:, :, 0], h_shifted) + torch.einsum('oc,bct->bot', Wf_l[:, :, 1], h)
            g = torch.einsum('oc,bct->bot', Wg_l[:, :, 0], h_shifted) + torch.einsum('oc,bct->bot', Wg_l[:, :, 1], h)

            filter_pre[l] = f
            gate_pre[l] = g

            # Gate: z = tanh(f) * sigmoid(g)
            z = torch.tanh(f) * torch.sigmoid(g)
            gated[l] = z

            # Residual: h_next[b, c, t] = h[b, c, t] + sum_o Wr_l[c, o, 0] * z[b, o, t]
            # Wr_l is [C, H, 1], z is [B, H, T]
            # residual[b, c, t] = sum_o Wr_l[c, o, 0] * z[b, o, t]
            residual = torch.einsum('co,bot->bct', Wr_l[:, :, 0], z)
            h_next = h + residual

            # Skip: s[b, j, t] = sum_o Ws_l[j, o, 0] * z[b, o, t]
            # Ws_l is [S, H, 1], z is [B, H, T]
            s = torch.einsum('jo,bot->bjt', Ws_l[:, :, 0], z)
            skips[l] = s

            states[l + 1] = h_next
            h = h_next

        skip_sum = torch.sum(skips, dim=0)  # [B, S, T]
        output = states[L]  # [B, C, T]

        return {
            'states': states,
            'filter_pre': filter_pre,
            'gate_pre': gate_pre,
            'gated': gated,
            'skips': skips,
            'skip_sum': skip_sum,
            'output': output
        }
