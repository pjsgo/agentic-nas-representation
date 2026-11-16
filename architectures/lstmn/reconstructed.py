import torch
import torch.nn as nn

class LSTMN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Parameters initialized to zeros as per specification
        self.Wh = nn.Parameter(torch.zeros(attention_dim, hidden_dim, dtype=torch.float64))
        self.Wx = nn.Parameter(torch.zeros(attention_dim, input_dim, dtype=torch.float64))
        self.Wr = nn.Parameter(torch.zeros(attention_dim, hidden_dim, dtype=torch.float64))
        self.v = nn.Parameter(torch.zeros(attention_dim, dtype=torch.float64))
        self.W = nn.Parameter(torch.zeros(4 * hidden_dim, hidden_dim + input_dim, dtype=torch.float64))

    def forward(self, x, hidden_memory, cell_memory, previous_read):
        # Ensure float64
        x = x.to(torch.float64)
        hidden_memory = hidden_memory.to(torch.float64)
        cell_memory = cell_memory.to(torch.float64)
        previous_read = previous_read.to(torch.float64)

        T, I = x.shape
        P, H = hidden_memory.shape
        A = self.attention_dim

        # Initialize outputs
        scores = torch.zeros(T, P + T, dtype=torch.float64)
        attention = torch.zeros(T, P + T, dtype=torch.float64)
        hidden_reads = torch.zeros(T, H, dtype=torch.float64)
        cell_reads = torch.zeros(T, H, dtype=torch.float64)
        gates = torch.zeros(T, 4, H, dtype=torch.float64)
        output = torch.zeros(T, H, dtype=torch.float64)

        # State variables for the scan
        Hmem = hidden_memory.clone() # [P, H]
        Cmem = cell_memory.clone()   # [P, H]
        rprev = previous_read.clone() # [H]

        new_hidden_states = []
        new_cell_states = []

        for t in range(T):
            n = P + t
            x_t = x[t] # [I]

            # 1. lstmn_address: causal_additive_attention_with_previous_read
            if n > 0:
                # Hmem is currently [n, H] because we append at the end of the previous iteration
                # Wait, Hmem starts as [P, H]. At t=0, n=P. Hmem has P rows.
                # At t=1, n=P+1. Hmem should have P+1 rows.
                # So Hmem is the tape up to current step t (exclusive of current new state, inclusive of prefix and previous new states).
                
                # Compute scores s[j] for j in 0..n-1
                # s[j] = sum_a v[a] * tanh( sum_h Wh[a,h]*Hmem[j,h] + sum_i Wx[a,i]*x[t,i] + sum_h Wr[a,h]*rprev[h] )
                
                # Hmem: [n, H]
                # Wh: [A, H] -> Wh.T: [H, A]
                # Hmem @ Wh.T -> [n, A]
                term_h = Hmem @ self.Wh.T # [n, A]
                
                # Wx: [A, I] -> Wx.T: [I, A]
                # x_t: [I] -> x_t @ Wx.T -> [A]
                term_x = x_t @ self.Wx.T # [A]
                
                # Wr: [A, H] -> Wr.T: [H, A]
                # rprev: [H] -> rprev @ Wr.T -> [A]
                term_r = rprev @ self.Wr.T # [A]
                
                # Sum terms: [n, A] + [A] + [A] -> [n, A]
                pre_activation = term_h + term_x + term_r
                
                # tanh: [n, A]
                activated = torch.tanh(pre_activation)
                
                # v: [A]
                # s = activated @ v -> [n]
                s = activated @ self.v # [n]
                
                # Softmax over j < n
                # Subtract max for stability
                s_max = torch.max(s)
                exp_s = torch.exp(s - s_max)
                p = exp_s / torch.sum(exp_s)
                
                # Write to scores and attention
                scores[t, :n] = s
                attention[t, :n] = p
            else:
                # n == 0, scores and attention remain zero
                p = torch.zeros(0, dtype=torch.float64)

            # 2. lstmn_read: paired_tape_weighted_read
            if n > 0:
                # rh[h] = sum_{j<n} p[j] * Hmem[j, h]
                # p: [n], Hmem: [n, H] -> p @ Hmem -> [H]
                rh = p @ Hmem # [H]
                rc = p @ Cmem # [H]
            else:
                rh = torch.zeros(H, dtype=torch.float64)
                rc = torch.zeros(H, dtype=torch.float64)
                
            hidden_reads[t] = rh
            cell_reads[t] = rc

            # 3. lstmn_gates: read_conditioned_gates
            # z = W @ concat(rh, x[t])
            # W: [4H, H+I]
            # concat(rh, x_t): [H+I]
            concat_in = torch.cat([rh, x_t], dim=0) # [H+I]
            z = self.W @ concat_in # [4H]
            
            i_gate = torch.sigmoid(z[0:H])
            f_gate = torch.sigmoid(z[H:2*H])
            o_gate = torch.sigmoid(z[2*H:3*H])
            g_gate = torch.tanh(z[3*H:4*H])
            
            # gates[t] = stack(i, f, o, g) -> [4,H]
            gates[t] = torch.stack([i_gate, f_gate, o_gate, g_gate], dim=0)

            # 4. lstmn_update: read_cell_update_and_append
            # c = f * rc + i * g
            c_new = f_gate * rc + i_gate * g_gate # [H]
            # h = o * tanh(c)
            h_new = o_gate * torch.tanh(c_new) # [H]
            
            output[t] = h_new
            
            # Append to tapes
            new_hidden_states.append(h_new)
            new_cell_states.append(c_new)
            
            # Update rprev for next step
            # "Set rprev=rh for the next step, not h."
            rprev = rh
            
            # Update Hmem and Cmem for next step
            # Hmem should now include the new state h_new
            # Hmem was [n, H]. New Hmem is [n+1, H]
            Hmem = torch.cat([Hmem, h_new.unsqueeze(0)], dim=0)
            Cmem = torch.cat([Cmem, c_new.unsqueeze(0)], dim=0)

        # Construct final outputs
        # hidden_memory: [P+T, H], old prefix then new hidden states
        final_hidden_memory = torch.cat([hidden_memory] + [h.unsqueeze(0) for h in new_hidden_states], dim=0) if T > 0 else hidden_memory
        # More efficient: just cat the list of new states
        if T > 0:
            new_h_stack = torch.stack(new_hidden_states, dim=0) # [T, H]
            new_c_stack = torch.stack(new_cell_states, dim=0) # [T, H]
            final_hidden_memory = torch.cat([hidden_memory, new_h_stack], dim=0)
            final_cell_memory = torch.cat([cell_memory, new_c_stack], dim=0)
        else:
            final_hidden_memory = hidden_memory
            final_cell_memory = cell_memory

        final_read = hidden_reads[T-1] if T > 0 else torch.zeros(H, dtype=torch.float64)

        return {
            'hidden_memory': final_hidden_memory,
            'cell_memory': final_cell_memory,
            'scores': scores,
            'attention': attention,
            'hidden_reads': hidden_reads,
            'cell_reads': cell_reads,
            'gates': gates,
            'output': output,
            'final_read': final_read
        }