import torch
import torch.nn as nn

class ALiBiCore(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        H = num_heads
        P = 1
        while P * 2 <= H:
            P *= 2
        
        slopes = []
        for h in range(H):
            if h < P:
                # m[h] = 2^(-8*(h+1)/P)
                exponent = -8.0 * (h + 1) / P
            else:
                r = h - P
                # m[P+r] = 2^(-8*(2*r+1)/(2*P))
                exponent = -8.0 * (2 * r + 1) / (2 * P)
            
            # Compute in real arithmetic (float64) then store float32
            val = 2.0 ** exponent
            slopes.append(val)
        
        slopes_tensor = torch.tensor(slopes, dtype=torch.float32)
        self.register_buffer('slopes', slopes_tensor, persistent=True)

    def forward(self, q, k, v, q_offset=0, k_offset=0):
        B, H, T, D = q.shape
        _, _, S, _ = k.shape
        _, _, _, Dv = v.shape
        
        # 1. Compute slopes (already in buffer)
        slopes = self.slopes.clone() # [H]
        
        # 2. Compute distance delta[i,j] = (k_offset + j) - (q_offset + i)
        # i ranges [0, T), j ranges [0, S)
        # delta[i, j] = k_offset - q_offset + j - i
        # We can construct this using broadcasting.
        # i is [0, T), j is [0, S)
        i_idx = torch.arange(T, device=q.device, dtype=torch.float32).view(1, 1, T, 1)
        j_idx = torch.arange(S, device=q.device, dtype=torch.float32).view(1, 1, 1, S)
        
        # delta[i,j] = (k_offset + j) - (q_offset + i)
        # delta = (k_offset - q_offset) + (j - i)
        offset_diff = float(k_offset - q_offset)
        delta = offset_diff + (j_idx - i_idx) # [1, 1, T, S]
        
        # 3. Compute bias[h,i,j] = m[h] * delta[i,j]
        # slopes is [H], delta is [1, 1, T, S]
        # We need bias [H, T, S]
        # slopes.view(H, 1, 1) * delta.view(1, T, S) -> [H, T, S]
        bias = slopes.view(H, 1, 1) * delta.view(1, T, S) # [H, T, S]
        
        # 4. Compute scaled dot product score[b,h,i,j] = sum_d(q[b,h,i,d]*k[b,h,j,d]) / sqrt(D)
        # q: [B, H, T, D]
        # k: [B, H, S, D]
        # We need [B, H, T, S]
        # Using einsum or matmul
        # q @ k^T -> [B, H, T, S]
        scores = torch.einsum('bhid,bhjd->bhij', q, k)
        scores = scores / (D ** 0.5)
        
        # 5. Add bias
        # scores: [B, H, T, S]
        # bias: [H, T, S]
        # broadcast bias to [1, H, T, S]
        z = scores + bias.view(1, H, T, S)
        
        # 6. Causal mask
        # logits[b,h,i,j] = z[b,h,i,j] if kj <= qi else -inf
        # kj = k_offset + j
        # qi = q_offset + i
        # Condition: k_offset + j <= q_offset + i  =>  j - i <= q_offset - k_offset
        # Let C = q_offset - k_offset
        # Mask is True where j - i <= C
        # delta[i,j] = k_offset - q_offset + j - i = -C + (j - i)
        # So j - i = delta + C
        # Condition: delta + C <= C  => delta <= 0
        # Wait, let's re-verify.
        # kj <= qi
        # k_offset + j <= q_offset + i
        # j - i <= q_offset - k_offset
        # Let C = q_offset - k_offset
        # j - i <= C
        # 
        # Our delta[i,j] = (k_offset + j) - (q_offset + i) = j - i - (q_offset - k_offset) = (j - i) - C
        # So j - i = delta + C
        # Condition: delta + C <= C  => delta <= 0
        # 
        # So we mask where delta > 0.
        # delta is [1, 1, T, S]
        mask = delta > 0 # [1, 1, T, S]
        
        # Apply mask to z
        # z: [B, H, T, S]
        # mask: [1, 1, T, S] -> broadcast to [B, H, T, S]
        logits = torch.where(mask, torch.tensor(float('-inf'), device=q.device), z)
        
        # 7. Softmax
        # weights = softmax(logits, dim=-1)
        # PyTorch softmax is stable.
        weights = torch.softmax(logits, dim=-1)
        
        # 8. Value mix
        # output[b,h,i,a] = sum_j weights[b,h,i,j] * v[b,h,j,a]
        # weights: [B, H, T, S]
        # v: [B, H, S, Dv]
        # output: [B, H, T, Dv]
        output = torch.einsum('bhij,bhjd->bhid', weights, v)
        
        return {
            'slopes': slopes,
            'bias': bias,
            'logits': logits,
            'weights': weights,
            'output': output
        }