import torch
from typing import Dict, Tuple

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.ParameterDict({
            'aW': torch.nn.Parameter(torch.zeros(4, 5, dtype=torch.float64), requires_grad=True),
            'aU': torch.nn.Parameter(torch.zeros(4, 10, dtype=torch.float64), requires_grad=True),
            'av': torch.nn.Parameter(torch.zeros(1, 4, dtype=torch.float64), requires_grad=True),
            'dWz': torch.nn.Parameter(torch.zeros(5, 3, dtype=torch.float64), requires_grad=True),
            'dUz': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'dCz': torch.nn.Parameter(torch.zeros(5, 10, dtype=torch.float64), requires_grad=True),
            'dWr': torch.nn.Parameter(torch.zeros(5, 3, dtype=torch.float64), requires_grad=True),
            'dUr': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'dCr': torch.nn.Parameter(torch.zeros(5, 10, dtype=torch.float64), requires_grad=True),
            'dWn': torch.nn.Parameter(torch.zeros(5, 3, dtype=torch.float64), requires_grad=True),
            'dUn': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'dCn': torch.nn.Parameter(torch.zeros(5, 10, dtype=torch.float64), requires_grad=True),
        })

    def _linear(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        out = x @ weight.T
        if bias is not None:
            out = out + bias
        return out

    def _masked_softmax(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg_inf = float("-inf")
        masked = torch.where(mask, x, torch.full_like(x, neg_inf))
        if not mask.any(dim=1).all():
            raise ValueError("Every batch row must contain a valid source")
        return torch.softmax(masked, dim=-1)

    def additive_attention(self, query: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy = self._linear(torch.tanh(self._linear(query, self.p['aW'])[:, None, :] + self._linear(memory, self.p['aU'])), self.p['av']).squeeze(-1)
        weights = self._masked_softmax(energy, mask)
        context = (weights[:, :, None] * memory).sum(dim=1)
        return context, weights, energy

    def conditional_gru(self, x: torch.Tensor, h: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.sigmoid(self._linear(x, self.p['dWz']) + self._linear(h, self.p['dUz']) + self._linear(context, self.p['dCz']))
        r = torch.sigmoid(self._linear(x, self.p['dWr']) + self._linear(h, self.p['dUr']) + self._linear(context, self.p['dCr']))
        proposal = torch.tanh(self._linear(x, self.p['dWn']) + self._linear(r * h, self.p['dUn']) + self._linear(context, self.p['dCn']))
        hidden = (1 - z) * h + z * proposal
        return hidden, hidden

    def decoder_step(self, x: torch.Tensor, h: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        alignment = self.additive_attention(h, memory, mask)
        update = self.conditional_gru(x, h, alignment[0])
        return (update[0], alignment[0], alignment[1]), update[1]

    def forward(self, memory: torch.Tensor, mask: torch.Tensor, target: torch.Tensor, h0: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, _ = target.shape
        if T == 0:
            raise ValueError("Empty input")
        emissions = []
        carry = h0
        for t in range(T):
            emission, carry = self.decoder_step(target[:, t, :], carry, memory, mask)
            emissions.append(emission)
        output = torch.stack([e[0] for e in emissions], dim=1)
        context = torch.stack([e[1] for e in emissions], dim=1)
        weights = torch.stack([e[2] for e in emissions], dim=1)
        return {
            'output': output,
            'context': context,
            'weights': weights,
            'final_h': carry,
        }

def build() -> torch.nn.Module:
    return Model()
