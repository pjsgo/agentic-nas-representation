import torch
import torch.nn as nn


class ByteNetStack(nn.Module):
    def __init__(self, channels: int, bottleneck: int, dilations: tuple, causal: bool):
        super().__init__()
        L = len(dilations)
        D = bottleneck
        C = channels
        self.L = L
        self.D = D
        self.C = C
        self.dilations = tuple(dilations)
        self.causal = bool(causal)

        self.W_in = nn.Parameter(torch.zeros(L, D, C, 1, dtype=torch.float64))
        self.W_dil = nn.Parameter(torch.zeros(L, D, D, 3, dtype=torch.float64))
        self.W_out = nn.Parameter(torch.zeros(L, C, D, 1, dtype=torch.float64))
        self.gamma1 = nn.Parameter(torch.ones(L, C, dtype=torch.float64))
        self.beta1 = nn.Parameter(torch.zeros(L, C, dtype=torch.float64))
        self.gamma2 = nn.Parameter(torch.ones(L, D, dtype=torch.float64))
        self.beta2 = nn.Parameter(torch.zeros(L, D, dtype=torch.float64))
        self.gamma3 = nn.Parameter(torch.ones(L, D, dtype=torch.float64))
        self.beta3 = nn.Parameter(torch.zeros(L, D, dtype=torch.float64))

    def _ln(self, v, gamma, beta, eps=1e-5):
        # v: [B, Q, T]
        mu = v.mean(dim=1, keepdim=True)
        var = ((v - mu) ** 2).mean(dim=1, keepdim=True)
        return gamma[None, :, None] * (v - mu) / torch.sqrt(var + eps) + beta[None, :, None]

    def forward(self, x):
        x = x.to(dtype=torch.float64)
        B, C, T = x.shape
        L = self.L
        D = self.D

        states = [x]
        n1_list = []
        a_list = []
        n2_list = []
        v_list = []
        n3_list = []
        r_list = []

        h = x
        for l in range(L):
            d = self.dilations[l]

            # norm1
            n1 = self._ln(h, self.gamma1[l], self.beta1[l])
            n1_list.append(n1)

            # reduce: relu(n1) then pointwise conv W_in
            a = torch.einsum('bct,oc->bot', torch.relu(n1), self.W_in[l, :, :, 0])
            a_list.append(a)

            # norm2
            n2 = self._ln(a, self.gamma2[l], self.beta2[l])
            n2_list.append(n2)

            # dilated conv: relu(n2) then dilated cross-correlation W_dil
            n2_relu = torch.relu(n2)
            v = torch.zeros(B, D, T, dtype=torch.float64, device=x.device)
            for k in range(3):
                if self.causal:
                    q = torch.arange(T, device=x.device) + (k - 2) * d
                else:
                    q = torch.arange(T, device=x.device) + (k - 1) * d
                mask = (q >= 0) & (q < T)
                q_clamped = torch.clamp(q, 0, T - 1)
                gathered = n2_relu[:, :, q_clamped]  # [B, D, T]
                gathered = gathered * mask.unsqueeze(0).unsqueeze(0).to(torch.float64)
                v = v + torch.einsum('bdt,od->bot', gathered, self.W_dil[l, :, :, k])
            v_list.append(v)

            # norm3
            n3 = self._ln(v, self.gamma3[l], self.beta3[l])
            n3_list.append(n3)

            # expand: relu(n3) then pointwise conv W_out
            r = torch.einsum('bdt,cd->bct', torch.relu(n3), self.W_out[l, :, :, 0])
            r_list.append(r)

            # residual
            h = h + r
            states.append(h)

        states_tensor = torch.stack(states, dim=0)  # [L+1, B, C, T]
        n1_tensor = torch.stack(n1_list, dim=0)  # [L, B, C, T]
        a_tensor = torch.stack(a_list, dim=0)  # [L, B, D, T]
        n2_tensor = torch.stack(n2_list, dim=0)  # [L, B, D, T]
        v_tensor = torch.stack(v_list, dim=0)  # [L, B, D, T]
        n3_tensor = torch.stack(n3_list, dim=0)  # [L, B, D, T]
        r_tensor = torch.stack(r_list, dim=0)  # [L, B, C, T]

        return {
            'states': states_tensor,
            'normalized1': n1_tensor,
            'reduced': a_tensor,
            'normalized2': n2_tensor,
            'dilated': v_tensor,
            'normalized3': n3_tensor,
            'expanded': r_tensor,
            'output': states_tensor[L],
        }
