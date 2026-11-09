import torch
import torch.nn.functional as F

class UnitA:
    def __init__(self):
        # Parameters are fixture-supplied, so we declare them as buffers to avoid
        # requiring initialization logic in the constructor. The evaluator will
        # overwrite these with the correct values.
        self.p0 = torch.empty(24, 8, dtype=torch.float32)
        self.p1 = torch.empty(24, dtype=torch.float32)
        self.p2 = torch.empty(8, 8, dtype=torch.float32)
        self.p3 = torch.empty(8, dtype=torch.float32)
        self.p4 = torch.empty(8, dtype=torch.float32)
        self.p5 = torch.empty(8, dtype=torch.float32)
        self.p6 = torch.empty(12, 8, dtype=torch.float32)
        self.p7 = torch.empty(12, dtype=torch.float32)
        self.p8 = torch.empty(8, 12, dtype=torch.float32)
        self.p9 = torch.empty(8, dtype=torch.float32)
        self.p10 = torch.empty(8, dtype=torch.float32)
        self.p11 = torch.empty(8, dtype=torch.float32)

    def forward(self, x, mask):
        # stage_a
        # project_a: linear(x, p0, p1)
        proj_a = F.linear(x, self.p0, self.p1)
        
        # partition_a: split(proj_a, [8, 8, 8], dim=2)
        q_raw, k_raw, v_raw = torch.split(proj_a, [8, 8, 8], dim=2)
        
        # shape_q: reshape(q_raw, [B, T, 2, 4])
        B, T, _ = q_raw.shape
        q = q_raw.reshape(B, T, 2, 4)
        
        # shape_k: reshape(k_raw, [B, T, 2, 4])
        k = k_raw.reshape(B, T, 2, 4)
        
        # shape_v: reshape(v_raw, [B, T, 2, 4])
        v = v_raw.reshape(B, T, 2, 4)
        
        # order_q: permute(q, [0, 2, 1, 3]) -> [B, 2, T, 4]
        q = q.permute(0, 2, 1, 3)
        
        # order_k: permute(k, [0, 2, 1, 3]) -> [B, 2, T, 4]
        k = k.permute(0, 2, 1, 3)
        
        # order_v: permute(v, [0, 2, 1, 3]) -> [B, 2, T, 4]
        v = v.permute(0, 2, 1, 3)
        
        # scale_value: constant 0.5
        scale = torch.tensor(0.5, dtype=torch.float32, device=x.device)
        
        # scale_q: mul(q, scale)
        q = q * scale
        
        # turn_k: transpose(k, dim0=2, dim1=3) -> [B, 2, 4, T]
        k = k.transpose(2, 3)
        
        # score_a: matmul(q, k) -> [B, 2, T, T]
        scores = torch.matmul(q, k)
        
        # mask_shape_a: unsqueeze(mask, dim=0) -> [1, T, T]
        mask_a = mask.unsqueeze(0)
        
        # mask_shape_b: unsqueeze(mask_a, dim=0) -> [1, 1, T, T]
        mask_b = mask_a.unsqueeze(0)
        
        # apply_mask: masked_fill(scores, mask_b, -inf)
        scores = scores.masked_fill(mask_b, float('-inf'))
        
        # prob_a: softmax(scores, dim=3)
        probs = F.softmax(scores, dim=3)
        
        # mix_a: matmul(probs, v) -> [B, 2, T, 4]
        mixed = torch.matmul(probs, v)
        
        # order_mix: permute(mixed, [0, 2, 1, 3]) -> [B, T, 2, 4]
        mixed = mixed.permute(0, 2, 1, 3)
        
        # join_a: reshape(mixed, [B, T, 8])
        joined = mixed.reshape(B, T, 8)
        
        # project_b: linear(joined, p2, p3)
        stage_a_out = F.linear(joined, self.p2, self.p3)
        
        # merge_a: add(x, stage_a_out)
        merged_a = x + stage_a_out
        
        # normalize_a: layer_norm(merged_a, [8], eps=1e-5, weight=p4, bias=p5)
        norm_a = F.layer_norm(merged_a, [8], self.p4, self.p5, eps=1e-5)
        
        # stage_b
        # expand_a: linear(norm_a, p6, p7)
        expanded = F.linear(norm_a, self.p6, self.p7)
        
        # activate_a: relu(expanded)
        activated = F.relu(expanded)
        
        # contract_a: linear(activated, p8, p9)
        stage_b_out = F.linear(activated, self.p8, self.p9)
        
        # merge_b: add(norm_a, stage_b_out)
        merged_b = norm_a + stage_b_out
        
        # normalize_b: layer_norm(merged_b, [8], eps=1e-5, weight=p10, bias=p11)
        y = F.layer_norm(merged_b, [8], self.p10, self.p11, eps=1e-5)
        
        return y

def create_unit_a():
    return UnitA()