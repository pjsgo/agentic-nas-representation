import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentUnitA(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameters are declared as fixture_supplied in the representation.
        # We initialize them with zeros here to satisfy the constructor requirement,
        # but they are expected to be overwritten by the evaluator/fixture.
        self.a_wi = nn.Parameter(torch.zeros(16, 3, dtype=torch.float32))
        self.a_bi = nn.Parameter(torch.zeros(16, dtype=torch.float32))
        self.a_wh = nn.Parameter(torch.zeros(16, 4, dtype=torch.float32))
        self.a_bh = nn.Parameter(torch.zeros(16, dtype=torch.float32))
        
        self.b_wi = nn.Parameter(torch.zeros(16, 5, dtype=torch.float32))
        self.b_bi = nn.Parameter(torch.zeros(16, dtype=torch.float32))
        self.b_wh = nn.Parameter(torch.zeros(16, 4, dtype=torch.float32))
        self.b_bh = nn.Parameter(torch.zeros(16, dtype=torch.float32))
        
        self.out_w = nn.Parameter(torch.zeros(7, 4, dtype=torch.float32))
        self.out_b = nn.Parameter(torch.zeros(7, dtype=torch.float32))

    def _step_a(self, x, h, c):
        # x: [B, 3], h: [B, 4], c: [B, 4]
        # Linear projections
        # F.linear(input, weight, bias) computes input @ weight.T + bias
        # weight shape [16, 3] -> output [B, 16]
        lin_i = F.linear(x, self.a_wi, self.a_bi)
        lin_h = F.linear(h, self.a_wh, self.a_bh)
        
        # Sum
        total = lin_i + lin_h
        
        # Split into 4 parts of size 4 along dim 1
        # total shape [B, 16] -> 4 tensors of [B, 4]
        i, f, g, o = torch.split(total, [4, 4, 4, 4], dim=1)
        
        # Activations
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        # Cell state update
        # fc = f * c
        fc = f * c
        # ig = i * g
        ig = i * g
        # next_c = fc + ig
        next_c = fc + ig
        
        # Hidden state update
        # ct = tanh(next_c)
        ct = torch.tanh(next_c)
        # next_h = o * ct
        next_h = o * ct
        
        return next_h, next_c

    def _cell_b(self, x, h, c):
        # x: [B, 5], h: [B, 4], c: [B, 4]
        lin_i = F.linear(x, self.b_wi, self.b_bi)
        lin_h = F.linear(h, self.b_wh, self.b_bh)
        
        total = lin_i + lin_h
        
        i, f, g, o = torch.split(total, [4, 4, 4, 4], dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        fc = f * c
        ig = i * g
        next_c = fc + ig
        
        ct = torch.tanh(next_c)
        next_h = o * ct
        
        return next_h, next_c

    def _step_b(self, x, h, c):
        # x: [B, 5], h: [B, 4], c: [B, 4]
        hn, cn = self._cell_b(x, h, c)
        
        # Readout
        # input: hn [B, 4], weight: [7, 4], bias: [7]
        # output: [B, 7]
        logits = F.linear(hn, self.out_w, self.out_b)
        
        return hn, cn, logits

    def forward(self, source_embeddings, target_prev_embeddings):
        # source_embeddings: [S, 2, 3]
        # target_prev_embeddings: [T, 2, 5]
        
        # Initial states are zero constants as per representation
        # zero_h and zero_c are [2, 4] tensors of zeros
        h0 = torch.zeros(2, 4, dtype=torch.float32, device=source_embeddings.device)
        c0 = torch.zeros(2, 4, dtype=torch.float32, device=source_embeddings.device)
        
        # Read A (Encoder)
        # Scan over source_embeddings along axis 0 (S)
        S = source_embeddings.shape[0]
        encoder_sequence_list = []
        h = h0
        c = c0
        
        for t in range(S):
            x_t = source_embeddings[t] # [2, 3]
            h, c = self._step_a(x_t, h, c)
            encoder_sequence_list.append(h)
        
        # Stack sequence: [S, 2, 4]
        encoder_sequence = torch.stack(encoder_sequence_list, dim=0)
        encoder_h = h
        encoder_c = c
        
        # Handoff
        # h and c are passed directly
        bridge_h = encoder_h
        bridge_c = encoder_c
        
        # Read B (Decoder)
        # Scan over target_prev_embeddings along axis 0 (T)
        T = target_prev_embeddings.shape[0]
        decoder_sequence_list = []
        scores_list = []
        h = bridge_h
        c = bridge_c
        
        for t in range(T):
            x_t = target_prev_embeddings[t] # [2, 5]
            h, c, logits_t = self._step_b(x_t, h, c)
            decoder_sequence_list.append(h)
            scores_list.append(logits_t)
        
        # Stack sequences
        decoder_sequence = torch.stack(decoder_sequence_list, dim=0) # [T, 2, 4]
        logits = torch.stack(scores_list, dim=0) # [T, 2, 7]
        decoder_h = h
        decoder_c = c
        
        return {
            "logits": logits,
            "encoder_sequence": encoder_sequence,
            "encoder_h": encoder_h,
            "encoder_c": encoder_c,
            "bridge_h": bridge_h,
            "bridge_c": bridge_c,
            "decoder_sequence": decoder_sequence,
            "decoder_h": decoder_h,
            "decoder_c": decoder_c
        }

def create_recurrent_unit_a():
    return RecurrentUnitA()