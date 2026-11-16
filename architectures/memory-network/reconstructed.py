import torch
import torch.nn as nn

class MemoryNetwork(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.UO = nn.Parameter(torch.zeros(embedding_dim, 3 * vocab_size, dtype=torch.float64))
        self.UR = nn.Parameter(torch.zeros(embedding_dim, 3 * vocab_size, dtype=torch.float64))

    def forward(self, memory, facts, query):
        memory = memory.to(dtype=torch.float64)
        facts = facts.to(dtype=torch.float64)
        query = query.to(dtype=torch.float64)

        V = query.shape[0]
        D = self.embedding_dim

        M = torch.cat([memory, facts], dim=0)
        N = M.shape[0]

        zero_V = torch.zeros(V, dtype=torch.float64)

        x1 = torch.cat([query, zero_V, zero_V], dim=0)
        uo_x1 = self.UO @ x1

        y_all = torch.cat([torch.zeros(N, V, dtype=torch.float64), torch.zeros(N, V, dtype=torch.float64), M], dim=1)
        uo_y_all = y_all @ self.UO.t()

        s1 = (uo_x1 * uo_y_all).sum(dim=1)
        i1 = torch.argmax(s1).item()

        x2 = torch.cat([query, M[i1], zero_V], dim=0)
        uo_x2 = self.UO @ x2
        s2 = (uo_x2 * uo_y_all).sum(dim=1)
        i2 = torch.argmax(s2).item()

        r = torch.cat([query, M[i1] + M[i2], zero_V], dim=0)
        ur_r = self.UR @ r

        y_w_all = torch.cat([torch.zeros(V, V, dtype=torch.float64), torch.zeros(V, V, dtype=torch.float64), torch.eye(V, dtype=torch.float64)], dim=1)
        ur_y_w_all = y_w_all @ self.UR.t()

        answer_scores = (ur_r * ur_y_w_all).sum(dim=1)
        answer = torch.argmax(answer_scores).item()

        return {
            'memory': M,
            'support_scores': torch.stack([s1, s2], dim=0),
            'support_indices': torch.tensor([i1, i2], dtype=torch.int64),
            'response_features': r,
            'answer_scores': answer_scores,
            'answer': torch.tensor(answer, dtype=torch.int64)
        }