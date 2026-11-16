import torch
from typing import Dict, Tuple

class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.ParameterDict({
            'Wi': torch.nn.Parameter(torch.zeros(5, 3, dtype=torch.float64), requires_grad=True),
            'Wh': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'bi': torch.nn.Parameter(torch.zeros(5, dtype=torch.float64), requires_grad=True),
            'bh': torch.nn.Parameter(torch.zeros(5, dtype=torch.float64), requires_grad=True),
        })

    def _linear(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x @ weight.T + bias

    def _cell(self, x_t: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_affine = self._linear(x_t, self.p['Wi'], self.p['bi'])
        recurrent_affine = self._linear(h, self.p['Wh'], self.p['bh'])
        hidden = torch.tanh(input_affine + recurrent_affine)
        return hidden, hidden

    def _scan(self, x: torch.Tensor, h0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        if T == 0:
            raise ValueError("Empty input")
        emissions = []
        carry = h0
        for t in range(T):
            emission, carry = self._cell(x[:, t, :], carry)
            emissions.append(emission)
        output = torch.stack(emissions, dim=1)
        return output, carry

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> Dict[str, torch.Tensor]:
        output, final_h = self._scan(x, h0)
        return {'output': output, 'final_h': final_h}


def build() -> torch.nn.Module:
    return Module()
