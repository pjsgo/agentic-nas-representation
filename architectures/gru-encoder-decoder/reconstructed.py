import torch
from typing import Dict, Tuple

class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.ParameterDict({
            'eWz': torch.nn.Parameter(torch.zeros(5, 3, dtype=torch.float64), requires_grad=True),
            'eUz': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'eWr': torch.nn.Parameter(torch.zeros(5, 3, dtype=torch.float64), requires_grad=True),
            'eUr': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'eWn': torch.nn.Parameter(torch.zeros(5, 3, dtype=torch.float64), requires_grad=True),
            'eUn': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'dWz': torch.nn.Parameter(torch.zeros(5, 3, dtype=torch.float64), requires_grad=True),
            'dUz': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'dCz': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'dWr': torch.nn.Parameter(torch.zeros(5, 3, dtype=torch.float64), requires_grad=True),
            'dUr': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'dCr': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'dWn': torch.nn.Parameter(torch.zeros(5, 3, dtype=torch.float64), requires_grad=True),
            'dUn': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'dCn': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'V': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'Vp': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
        })

    def _linear(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        out = x @ weight.T
        if bias is not None:
            out = out + bias
        return out

    def _encoder_cell(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.sigmoid(self._linear(x, self.p['eWz']) + self._linear(h, self.p['eUz']))
        r = torch.sigmoid(self._linear(x, self.p['eWr']) + self._linear(h, self.p['eUr']))
        proposal = torch.tanh(self._linear(x, self.p['eWn']) + self._linear(r * h, self.p['eUn']))
        hidden = z * h + (1 - z) * proposal
        return hidden, hidden

    def _decoder_cell(self, x: torch.Tensor, h: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.sigmoid(self._linear(x, self.p['dWz']) + self._linear(h, self.p['dUz']) + self._linear(context, self.p['dCz']))
        r = torch.sigmoid(self._linear(x, self.p['dWr']) + self._linear(h, self.p['dUr']) + self._linear(context, self.p['dCr']))
        proposal = torch.tanh(self._linear(x, self.p['dWn']) + r * (self._linear(h, self.p['dUn']) + self._linear(context, self.p['dCn'])))
        hidden = z * h + (1 - z) * proposal
        return hidden, hidden

    def _scan(self, step, x: torch.Tensor, carry: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError('scan input must be 3D [B,S,F]')
        B, S, F = x.shape
        if S == 0:
            raise ValueError('scan input must have positive sequence length')
        emissions = []
        c = carry
        for t in range(S):
            emission, c = step(x[:, t, :], c)
            emissions.append(emission)
        out = torch.stack(emissions, dim=1)
        return out, c

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoder_out, encoder_final = self._scan(self._encoder_cell, source, torch.zeros(source.shape[0], 5, dtype=source.dtype, device=source.device))
        fixed_context = torch.tanh(self._linear(encoder_final, self.p['V']))
        decoder_initial = torch.tanh(self._linear(fixed_context, self.p['Vp']))

        def decoder_step(x_t: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return self._decoder_cell(x_t, h_t, fixed_context)

        decoder_out, final_h = self._scan(decoder_step, target, decoder_initial)

        return {
            'output': decoder_out,
            'context': fixed_context,
            'decoder_initial': decoder_initial,
            'encoder_final': encoder_final,
            'final_h': final_h,
        }


def build() -> torch.nn.Module:
    return Module()
