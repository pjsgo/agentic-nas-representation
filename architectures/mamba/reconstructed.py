import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveStateSubunit(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameters are declared as fixture_supplied in the representation.
        # We initialize them with zeros here to satisfy PyTorch's requirement for
        # a valid tensor object. The evaluator is expected to load the correct
        # fixture values into these parameters before execution.
        self.p_A_log = nn.Parameter(torch.zeros(3, 2, dtype=torch.float32))
        self.p_D = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(self, u, gate, delta_logits, B, C, initial_state):
        # 1. delta_positive: softplus
        delta = F.softplus(delta_logits, beta=1.0, threshold=20.0)
        
        # 2. delta_layout: transpose (dim0=1, dim1=2)
        # Input: [Batch, Time, 3] -> Output: [Batch, 3, Time]
        delta_layout = delta.transpose(1, 2)
        
        # 3. rates_exp: exp
        rates_exp = torch.exp(self.p_A_log)
        
        # 4. rates_negative: neg
        rates_negative = -rates_exp
        
        # 5. rates_batch: unsqueeze (dim=0)
        rates_batch = rates_negative.unsqueeze(0) # [1, 3, 2]
        
        # 6. rates_time: unsqueeze (dim=2)
        rates_time = rates_batch.unsqueeze(2) # [1, 3, 1, 2]
        
        # 7. delta_state: unsqueeze (dim=3)
        delta_state = delta_layout.unsqueeze(3) # [Batch, 3, Time, 1]
        
        # 8. decay_argument: mul
        # rates_time: [1, 3, 1, 2], delta_state: [Batch, 3, Time, 1]
        # Broadcast to: [Batch, 3, Time, 2]
        decay_argument = rates_time * delta_state
        
        # 9. decay: exp
        decay = torch.exp(decay_argument)
        
        # 10. B_channel: unsqueeze (dim=1)
        # B: [Batch, Time, 2] -> [Batch, 1, Time, 2]
        B_channel = B.unsqueeze(1)
        
        # 11. discrete_B: mul
        # delta_state: [Batch, 3, Time, 1], B_channel: [Batch, 1, Time, 2]
        # Broadcast to: [Batch, 3, Time, 2]
        discrete_B = delta_state * B_channel
        
        # 12. u_state: unsqueeze (dim=3)
        # u: [Batch, 3, Time] -> [Batch, 3, Time, 1]
        u_state = u.unsqueeze(3)
        
        # 13. drive: mul
        # discrete_B: [Batch, 3, Time, 2], u_state: [Batch, 3, Time, 1]
        # Broadcast to: [Batch, 3, Time, 2]
        drive = discrete_B * u_state
        
        # 14. recurrence: scan
        # We implement the scan loop manually as per the frozen semantics.
        # Sequence axis is 2 (Time).
        # Inputs to step:
        #   decay_t: [Batch, 3, 2] (sliced from decay [Batch, 3, Time, 2])
        #   drive_t: [Batch, 3, 2] (sliced from drive [Batch, 3, Time, 2])
        #   C_t: [Batch, 2] (sliced from C [Batch, Time, 2])
        #   previous: [Batch, 3, 2] (carry)
        # Outputs from step:
        #   next: [Batch, 3, 2] (new carry)
        #   readout: [Batch, 3] (emitted)
        
        Time = u.shape[2]
        readouts = torch.empty((u.shape[0], u.shape[1], Time), dtype=torch.float32, device=u.device)
        
        state = initial_state
        
        for t in range(Time):
            # Slice inputs for current time step
            decay_t = decay[:, :, t, :] # [Batch, 3, 2]
            drive_t = drive[:, :, t, :] # [Batch, 3, 2]
            C_t = C[:, t, :] # [Batch, 2]
            
            # Step body
            # 15. retained: mul
            retained = decay_t * state
            
            # 16. updated: add
            updated = retained + drive_t
            
            # 17. C_column: unsqueeze (dim=2)
            # C_t: [Batch, 2] -> [Batch, 2, 1]
            C_column = C_t.unsqueeze(2)
            
            # 18. readout_product: matmul
            # updated: [Batch, 3, 2], C_column: [Batch, 2, 1]
            # Result: [Batch, 3, 1]
            readout_product = torch.matmul(updated, C_column)
            
            # 19. readout_vector: squeeze (dim=2)
            # [Batch, 3, 1] -> [Batch, 3]
            readout_vector = readout_product.squeeze(2)
            
            # Store readout
            readouts[:, :, t] = readout_vector
            
            # Update state
            state = updated
        
        final_state = state
        
        # 20. D_batch: unsqueeze (dim=0)
        # p_D: [3] -> [1, 3]
        D_batch = self.p_D.unsqueeze(0)
        
        # 21. D_time: unsqueeze (dim=2)
        # [1, 3] -> [1, 3, 1]
        D_time = D_batch.unsqueeze(2)
        
        # 22. skip: mul
        # u: [Batch, 3, Time], D_time: [1, 3, 1]
        # Broadcast to: [Batch, 3, Time]
        skip = u * D_time
        
        # 23. pre_gate: add
        # readouts: [Batch, 3, Time], skip: [Batch, 3, Time]
        pre_gate = readouts + skip
        
        # 24. gate_activation: silu
        gate_activation = F.silu(gate)
        
        # 25. gated: mul
        # pre_gate: [Batch, 3, Time], gate_activation: [Batch, 3, Time]
        gated = pre_gate * gate_activation
        
        return gated, final_state

# Factory function to match entrypoint metadata
def create_selective_state_subunit():
    return SelectiveStateSubunit()