import torch
import torch.nn as nn

class E2EMemoryNetwork(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hops: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hops = hops
        # E: [K+1, V, D]
        self.E = nn.Parameter(torch.zeros(hops + 1, vocab_size, embedding_dim, dtype=torch.float64))

    def forward(self, facts, query):
        # facts: [N, V] float64
        # query: [V] float64
        K = self.hops
        N = facts.shape[0]
        V = self.vocab_size
        D = self.embedding_dim
        E = self.E

        # Precompute embeddings for all hops
        # m[k, j, d] = sum_v facts[j, v] * E[k, v, d]
        # c[k, j, d] = sum_v facts[j, v] * E[k+1, v, d]
        # u[0, d] = sum_v query[v] * E[0, v, d]

        # facts: [N, V]
        # E[0:K, :, :]: [K, V, D]
        # m = facts @ E[0:K].permute(1, 0, 2) -> [N, K, D] -> permute to [K, N, D]
        # Actually: m[k, j, d] = sum_v facts[j, v] * E[k, v, d]
        # This is: for each k, m[k] = facts @ E[k] -> [N, D]
        # So m = torch.stack([facts @ E[k] for k in range(K)]) -> [K, N, D]
        
        # c[k, j, d] = sum_v facts[j, v] * E[k+1, v, d]
        # c = torch.stack([facts @ E[k+1] for k in range(K)]) -> [K, N, D]
        
        # u0 = query @ E[0] -> [D]
        
        m = torch.stack([facts @ E[k] for k in range(K)], dim=0)  # [K, N, D]
        c = torch.stack([facts @ E[k+1] for k in range(K)], dim=0)  # [K, N, D]
        u0 = query @ E[0]  # [D]

        # Initialize outputs
        scores = torch.zeros(K, N, dtype=torch.float64)
        probabilities = torch.zeros(K, N, dtype=torch.float64)
        reads = torch.zeros(K, D, dtype=torch.float64)
        queries = torch.zeros(K + 1, D, dtype=torch.float64)
        queries[0] = u0

        u = u0.clone()
        
        for k in range(K):
            # scores[k, j] = sum_d m[k, j, d] * u[d]
            # m[k]: [N, D], u: [D] -> scores[k] = m[k] @ u -> [N]
            scores[k] = m[k] @ u
            
            # p[k, :] = softmax(scores[k, :]) over N
            # Manual softmax for float64 precision
            z = scores[k]
            z_max = z.max()
            exp_z = torch.exp(z - z_max)
            sum_exp = exp_z.sum()
            probabilities[k] = exp_z / sum_exp
            
            # o[k, d] = sum_j p[k, j] * c[k, j, d]
            # c[k]: [N, D], p[k]: [N] -> o[k] = p[k] @ c[k] -> [D]
            reads[k] = probabilities[k] @ c[k]
            
            # u[k+1, d] = u[k, d] + o[k, d]
            u = u + reads[k]
            queries[k + 1] = u

        # Answer projection
        # logits[v] = sum_d E[K, v, d] * u[K, d]
        # E[K]: [V, D], u: [D] -> logits = E[K] @ u -> [V]
        logits = E[K] @ u
        
        # answer_probabilities = softmax(logits) over V
        z = logits
        z_max = z.max()
        exp_z = torch.exp(z - z_max)
        sum_exp = exp_z.sum()
        answer_probabilities = exp_z / sum_exp

        return {
            'input_memory': m,
            'output_memory': c,
            'scores': scores,
            'probabilities': probabilities,
            'reads': reads,
            'queries': queries,
            'logits': logits,
            'answer_probabilities': answer_probabilities
        }