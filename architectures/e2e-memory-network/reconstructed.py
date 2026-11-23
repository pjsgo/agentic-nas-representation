import torch

class E2EMemoryNetwork(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hops: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hops = hops
        # E has shape [K+1, V, D]
        # Slice 0 is the initial query embedding (B)
        # Slices 1..K are the memory embeddings (A for hop k-1, C for hop k-1)
        # Slice K is the final classifier (W)
        # Note: The spec says E is [K+1, V, D].
        # Hop k (0-indexed) uses A=E[k] and C=E[k+1].
        # So for K hops, we need E[0]..E[K].
        # E[0] is used for initial query u[0] and as A for hop 0.
        # E[1] is used as C for hop 0 and A for hop 1.
        # ...
        # E[K] is used as C for hop K-1 and as the final classifier W.
        self.E = torch.nn.Parameter(torch.zeros(hops + 1, vocab_size, embedding_dim, dtype=torch.float64))

    def forward(self, facts: torch.Tensor, query: torch.Tensor) -> dict:
        # facts: [N, V] float64
        # query: [V] float64
        K = self.hops
        N = facts.shape[0]
        V = self.vocab_size
        D = self.embedding_dim
        E = self.E

        # 1. Embeddings
        # u[0] = sum_v query[v] * E[0, v, d]
        # query is [V], E[0] is [V, D]
        u = torch.matmul(query, E[0])  # [D]

        # m[k, j, d] = sum_v facts[j, v] * E[k, v, d]
        # c[k, j, d] = sum_v facts[j, v] * E[k+1, v, d]
        # facts is [N, V]
        # We need to compute m and c for all hops k=0..K-1
        # m[k] = facts @ E[k] -> [N, D]
        # c[k] = facts @ E[k+1] -> [N, D]

        input_memory = [] # m
        output_memory = [] # c
        scores_list = []
        probabilities_list = []
        reads_list = []
        queries_list = [u] # initial query

        for k in range(K):
            # Compute m[k] and c[k]
            # facts: [N, V], E[k]: [V, D] -> m[k]: [N, D]
            m_k = torch.matmul(facts, E[k])
            c_k = torch.matmul(facts, E[k+1])

            input_memory.append(m_k)
            output_memory.append(c_k)

            # 2. Addressing
            # scores[k, j] = sum_d m[k, j, d] * u[k, d]
            # m_k: [N, D], u: [D] -> scores_k: [N]
            scores_k = torch.matmul(m_k, u) # [N]

            # p[k, :] = softmax(scores_k) over N
            # Softmax: subtract max, exp, divide by sum
            max_score = torch.max(scores_k)
            exp_scores = torch.exp(scores_k - max_score)
            sum_exp = torch.sum(exp_scores)
            p_k = exp_scores / sum_exp

            scores_list.append(scores_k)
            probabilities_list.append(p_k)

            # 3. Read
            # o[k, d] = sum_j p[k, j] * c[k, j, d]
            # p_k: [N], c_k: [N, D] -> o_k: [D]
            o_k = torch.matmul(p_k, c_k) # [D]

            reads_list.append(o_k)

            # 4. Update
            # u[k+1, d] = u[k, d] + o[k, d]
            u = u + o_k
            queries_list.append(u)

        # Stack results
        input_memory = torch.stack(input_memory, dim=0) # [K, N, D]
        output_memory = torch.stack(output_memory, dim=0) # [K, N, D]
        scores = torch.stack(scores_list, dim=0) # [K, N]
        probabilities = torch.stack(probabilities_list, dim=0) # [K, N]
        reads = torch.stack(reads_list, dim=0) # [K, D]
        queries = torch.stack(queries_list, dim=0) # [K+1, D]

        # 5. Answer
        # logits[v] = sum_d E[K, v, d] * u[K, d]
        # E[K] is [V, D], u is [D] (which is u[K])
        logits = torch.matmul(E[K], u) # [V]

        # answer_probabilities = softmax(logits) over V
        max_logit = torch.max(logits)
        exp_logits = torch.exp(logits - max_logit)
        sum_exp_logits = torch.sum(exp_logits)
        answer_probabilities = exp_logits / sum_exp_logits

        return {
            'input_memory': input_memory,
            'output_memory': output_memory,
            'scores': scores,
            'probabilities': probabilities,
            'reads': reads,
            'queries': queries,
            'logits': logits,
            'answer_probabilities': answer_probabilities
        }
