import torch
import torch.nn as nn

class ElementwiseBranch(nn.Module):
    def forward(self, x):
        neg_x = torch.neg(x)
        neg_mag = torch.nn.functional.relu(neg_x, inplace=False)
        minimum = torch.neg(neg_mag)
        exp_val = torch.exp(minimum)
        one = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        minus_one = torch.sub(exp_val, one)
        positive = torch.nn.functional.relu(x, inplace=False)
        return torch.add(positive, minus_one)

class UnitC(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = ElementwiseBranch()
    
    def forward(self, q, k, v, s, z):
        T = q.shape[1]
        y_list = []
        
        for t in range(T):
            q_t = q[:, t, :, :]
            k_t = k[:, t, :, :]
            v_t = v[:, t, :, :]
            
            q_elu = self.branch(q_t)
            q_phi = torch.add(q_elu, torch.tensor(1.0, dtype=torch.float32, device=q.device))
            
            k_elu = self.branch(k_t)
            k_phi = torch.add(k_elu, torch.tensor(1.0, dtype=torch.float32, device=k.device))
            
            key_expand = torch.unsqueeze(k_phi, dim=3)
            value_expand = torch.unsqueeze(v_t, dim=2)
            
            outer = torch.mul(key_expand, value_expand)
            
            s.add_(outer)
            
            z.add_(k_phi)
            
            den_product = torch.mul(q_phi, z)
            den_sum = torch.sum(den_product, dim=2, keepdim=False)
            den_eps = torch.add(den_sum, torch.tensor(1.0e-06, dtype=torch.float32, device=q.device))
            
            reciprocal = torch.div(torch.tensor(1.0, dtype=torch.float32, device=q.device), den_eps)
            
            query_expand = torch.unsqueeze(q_phi, dim=3)
            read_product = torch.mul(query_expand, s)
            read_sum = torch.sum(read_product, dim=2, keepdim=False)
            
            scale_expand = torch.unsqueeze(reciprocal, dim=2)
            
            y_t = torch.mul(read_sum, scale_expand)
            y_list.append(y_t)
        
        y = torch.stack(y_list, dim=1)
        return y, s, z

def create_unit_c():
    return UnitC()