import torch
from typing import Tuple, Dict, Any


def _linear(x, weight, bias=None):
    if bias is None:
        return x @ weight.T
    return x @ weight.T + bias


def _lstm(x, state, wi, wh, bi, bh):
    h, c = state
    gates = _linear(x, wi, bi) + _linear(h, wh, bh)
    input_g, forget_g, proposal_g, output_g = gates[:, 0:5], gates[:, 5:10], gates[:, 10:15], gates[:, 15:20]
    c_next = torch.sigmoid(forget_g) * c + torch.sigmoid(input_g) * torch.tanh(proposal_g)
    h_next = torch.sigmoid(output_g) * torch.tanh(c_next)
    return h_next, c_next


def _masked_softmax(x, mask):
    x = x.masked_fill(~mask, float('-inf'))
    if not mask.any(dim=1).all():
        raise ValueError('Every batch row must contain a valid source')
    return torch.softmax(x, dim=-1)


def _scan(step, x, carry, reverse=False):
    B, T, F = x.shape
    if T == 0:
        raise ValueError('Empty input')
    emissions = []
    if reverse:
        for t in range(T - 1, -1, -1):
            emission, carry = step(x[:, t, :], carry)
            emissions.append(emission)
        emissions.reverse()
    else:
        for t in range(T):
            emission, carry = step(x[:, t, :], carry)
            emissions.append(emission)
    if isinstance(emissions[0], tuple):
        return tuple(torch.stack([e[i] for e in emissions], dim=1) for i in range(len(emissions[0]))), carry
    return torch.stack(emissions, dim=1), carry


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.ParameterDict({
            'efwi': torch.nn.Parameter(torch.zeros(20, 3, dtype=torch.float64), requires_grad=True),
            'efwh': torch.nn.Parameter(torch.zeros(20, 5, dtype=torch.float64), requires_grad=True),
            'efbi': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'efbh': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'ebwi': torch.nn.Parameter(torch.zeros(20, 3, dtype=torch.float64), requires_grad=True),
            'ebwh': torch.nn.Parameter(torch.zeros(20, 5, dtype=torch.float64), requires_grad=True),
            'ebbi': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'ebbh': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'e2wi': torch.nn.Parameter(torch.zeros(20, 10, dtype=torch.float64), requires_grad=True),
            'e2wh': torch.nn.Parameter(torch.zeros(20, 5, dtype=torch.float64), requires_grad=True),
            'e2bi': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'e2bh': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'e3wi': torch.nn.Parameter(torch.zeros(20, 5, dtype=torch.float64), requires_grad=True),
            'e3wh': torch.nn.Parameter(torch.zeros(20, 5, dtype=torch.float64), requires_grad=True),
            'e3bi': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'e3bh': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'd1wi': torch.nn.Parameter(torch.zeros(20, 8, dtype=torch.float64), requires_grad=True),
            'd1wh': torch.nn.Parameter(torch.zeros(20, 5, dtype=torch.float64), requires_grad=True),
            'd1bi': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'd1bh': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'd2wi': torch.nn.Parameter(torch.zeros(20, 10, dtype=torch.float64), requires_grad=True),
            'd2wh': torch.nn.Parameter(torch.zeros(20, 5, dtype=torch.float64), requires_grad=True),
            'd2bi': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'd2bh': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'd3wi': torch.nn.Parameter(torch.zeros(20, 10, dtype=torch.float64), requires_grad=True),
            'd3wh': torch.nn.Parameter(torch.zeros(20, 5, dtype=torch.float64), requires_grad=True),
            'd3bi': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'd3bh': torch.nn.Parameter(torch.zeros(20, dtype=torch.float64), requires_grad=True),
            'aW': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'aU': torch.nn.Parameter(torch.zeros(5, 5, dtype=torch.float64), requires_grad=True),
            'av': torch.nn.Parameter(torch.zeros(1, 5, dtype=torch.float64), requires_grad=True),
        })

    def _ef_cell(self, x, state):
        lstm_state = _lstm(x, state, self.p['efwi'], self.p['efwh'], self.p['efbi'], self.p['efbh'])
        return lstm_state[0], lstm_state

    def _eb_cell(self, x, state):
        lstm_state = _lstm(x, state, self.p['ebwi'], self.p['ebwh'], self.p['ebbi'], self.p['ebbh'])
        return lstm_state[0], lstm_state

    def _e2_cell(self, x, state):
        lstm_state = _lstm(x, state, self.p['e2wi'], self.p['e2wh'], self.p['e2bi'], self.p['e2bh'])
        return lstm_state[0], lstm_state

    def _e3_cell(self, x, state):
        lstm_state = _lstm(x, state, self.p['e3wi'], self.p['e3wh'], self.p['e3bi'], self.p['e3bh'])
        return lstm_state[0], lstm_state

    def _d1_cell(self, x, state):
        lstm_state = _lstm(x, state, self.p['d1wi'], self.p['d1wh'], self.p['d1bi'], self.p['d1bh'])
        return lstm_state[0], lstm_state

    def _d2_cell(self, x, state):
        lstm_state = _lstm(x, state, self.p['d2wi'], self.p['d2wh'], self.p['d2bi'], self.p['d2bh'])
        return lstm_state[0], lstm_state

    def _d3_cell(self, x, state):
        lstm_state = _lstm(x, state, self.p['d3wi'], self.p['d3wh'], self.p['d3bi'], self.p['d3bh'])
        return lstm_state[0], lstm_state

    def _additive_attention(self, query, memory, mask):
        energy = _linear(torch.tanh(_linear(query, self.p['aW'])[:, None, :] + _linear(memory, self.p['aU'])), self.p['av']).squeeze(-1)
        weights = _masked_softmax(energy, mask)
        context = (weights[:, :, None] * memory).sum(dim=1)
        return context, weights, energy

    def _decoder_step(self, x, state, memory, mask):
        bottom = self._d1_cell(torch.cat((x, state[3]), dim=-1), state[0])
        alignment = self._additive_attention(bottom[0], memory, mask)
        middle = self._d2_cell(torch.cat((bottom[0], state[3]), dim=-1), state[1])
        top = self._d3_cell(torch.cat((middle[0], state[3]), dim=-1), state[2])
        residual_output = top[0] + middle[0]
        return (residual_output, alignment[0], alignment[1], bottom[0]), (bottom[1], middle[1], top[1], alignment[0])

    def forward(self, source, target):
        zero = torch.zeros(source.shape[0], 5, dtype=source.dtype, device=source.device)
        forward_encoder = _scan(self._ef_cell, source, (zero, zero))
        backward_encoder = _scan(self._eb_cell, source, (zero, zero), reverse=True)
        bidirectional_concat = torch.cat((forward_encoder[0], backward_encoder[0]), dim=-1)
        middle_encoder = _scan(self._e2_cell, bidirectional_concat, (zero, zero))
        top_encoder = _scan(self._e3_cell, middle_encoder[0], (zero, zero))
        encoder_residual = top_encoder[0] + middle_encoder[0]
        encoder_state_bridge = (backward_encoder[1], middle_encoder[1], top_encoder[1], zero)
        mask = source[:, :, 0] == source[:, :, 0]
        decoder_scan = _scan(lambda x, s: self._decoder_step(x, s, encoder_residual, mask), target, encoder_state_bridge)
        return {
            'output': decoder_scan[0][0],
            'context': decoder_scan[0][1],
            'weights': decoder_scan[0][2],
            'bottom_hidden': decoder_scan[0][3],
            'memory': encoder_residual,
            'decoder_initial': encoder_state_bridge,
            'final_state': decoder_scan[1]
        }


def build():
    return Model()
