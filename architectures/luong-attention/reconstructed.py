import torch
from typing import Dict, Tuple

class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.ParameterDict({
            'dwi': torch.nn.Parameter(torch.zeros(20, 8, dtype=torch.float64), requires_grad=True),
            'dwh': torch.nn.Parameter(torch.zeros(20, 5, dtype=torch.float64), requires_grad=True),
            'dbi': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'dbh': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'Wa': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'Wc': torch.nn.Parameter(torch.zeros(5, 10, dtype=torch.float64), requires_grad=True),
        })

    def _linear(self, x, weight, bias=None):
        if bias is None:
            return x @ weight.T
        return x @ weight.T + bias

    def _lstm(self, x, h, c, wi, wh, bi, bh):
        gates = self._linear(x, wi, bi) + self._linear(h, wh, bh)
        input_gate = gates[:, 0:5]
        forget_gate = gates[:, 5:10]
        proposal = gates[:, 10:15]
        output_gate = gates[:, 15:20]
        c_next = torch.sigmoid(forget_gate) * c + torch.sigmoid(input_gate) * torch.tanh(proposal)
        h_next = torch.sigmoid(output_gate) * torch.tanh(c_next)
        return h_next, c_next

    def _masked_softmax(self, scores, mask):
        masked = scores.masked_fill(~mask, float('-inf'))
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Every batch row must contain a valid source")
        return torch.softmax(masked, dim=-1)

    def _decoder_step(self, x, state, memory, mask):
        h, c, feed = state
        input_feeding = torch.cat((x, feed), dim=-1)
        h_next, c_next = self._lstm(input_feeding, h, c, self.p['dwi'], self.p['dwh'], self.p['dbi'], self.p['dbh'])
        general_scores = (self._linear(memory, self.p['Wa']) * h_next[:, None, :]).sum(dim=-1)
        weights = self._masked_softmax(general_scores, mask)
        context = (weights[:, :, None] * memory).sum(dim=1)
        attentional_vector = torch.tanh(self._linear(torch.cat((context, h_next), dim=-1), self.p['Wc']))
        emission = (attentional_vector, context, weights, h_next)
        next_state = (h_next, c_next, attentional_vector)
        return emission, next_state

    def _scan(self, step, x, carry):
        B, T, _ = x.shape
        if T == 0:
            raise ValueError("Empty input")
        emissions = []
        for t in range(T):
            emission, carry = step(x[:, t, :], carry)
            emissions.append(emission)
        stacked = tuple(torch.stack([e[t] for e in emissions], dim=1) for t in range(len(emissions[0])))
        return stacked, carry

    def forward(self, memory, mask, target, h0, c0, feed0):
        emissions, final_state = self._scan(
            lambda x, s: self._decoder_step(x, s, memory, mask),
            target,
            (h0, c0, feed0)
        )
        output, context, weights, hidden = emissions
        final_h, final_c, final_feed = final_state
        return {
            'output': output,
            'context': context,
            'weights': weights,
            'hidden': hidden,
            'final_h': final_h,
            'final_c': final_c,
            'final_feed': final_feed,
        }

def build():
    return Module()
