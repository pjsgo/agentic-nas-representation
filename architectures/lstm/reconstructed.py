import torch
import torch.nn.functional as F

class RecurrentTensorRegion:
    def __init__(self, p0, p1, p2, p3):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def forward(self, x, h0, c0):
        T = x.size(1)
        outputs = []
        a = h0
        b = c0
        for t in range(T):
            x_t = x[:, t, :]
            out_a = F.linear(x_t, self.p0, self.p2)
            out_b = F.linear(a, self.p1, self.p3)
            combined = out_a + out_b
            parts = torch.split(combined, [5, 5, 5, 5], dim=1)
            g_a, g_b, g_c, g_d = parts
            g_a = torch.sigmoid(g_a)
            g_b = torch.sigmoid(g_b)
            g_c = torch.tanh(g_c)
            g_d = torch.sigmoid(g_d)
            retained = g_b * b
            written = g_a * g_c
            b = retained + written
            readout = torch.tanh(b)
            a = g_d * readout
            outputs.append(a)
        y = torch.stack(outputs, dim=1)
        return y, a, b

def create_recurrent_tensor_region(p0, p1, p2, p3):
    return RecurrentTensorRegion(p0, p1, p2, p3)