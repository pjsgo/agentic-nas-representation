import torch
import math
from typing import Dict, Any

class SparseTransformer(torch.nn.Module):
    def __init__(self, spec: Dict[str, Any]):
        super().__init__()
        self.spec = spec

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. sparse_positions: positions
        # N is size on axis 2 of q
        N = q.size(2)
        positions = torch.arange(N, device=q.device, dtype=torch.int64)

        # 2. sparse_offsets: pairwise_difference
        # Entry [i, j] = p[i] - p[j]
        # positions shape [N], offsets shape [N, N]
        offsets = positions.unsqueeze(1) - positions.unsqueeze(0)

        # 3. sparse_causal: ge (offsets >= 0)
        causal = offsets >= 0

        # 4. sparse_local_band: lt (offsets < 3)
        local_band = offsets < self.spec['factors']['sparse_local_band']['attributes']['value']

        # 5. sparse_stride_links: mod_eq (offsets % 3 == 0)
        # Note: mod_eq is defined as remainder modulo positive modulus equals value.
        # For negative numbers, Python's % behaves as floor division remainder.
        # e.g., -1 % 3 = 2, -2 % 3 = 1, -3 % 3 = 0.
        # The rule says "Elementwise remainder modulo positive attributes.modulus equals attributes.value."
        # We use torch.remainder or % operator. Torch's % follows C-style truncation? No, torch.remainder follows Python.
        # Let's check torch docs: torch.remainder(input, other) computes element-wise remainder. It follows Python's %.
        # So -1 % 3 is 2. -3 % 3 is 0.
        stride_attrs = self.spec['factors']['sparse_stride_links']['attributes']
        stride_links = (offsets % stride_attrs['modulus']) == stride_attrs['value']

        # 6. sparse_local_pattern: and (causal AND local_band)
        local_pattern = causal & local_band

        # 7. sparse_strided_pattern: and (causal AND stride_links)
        strided_pattern = causal & stride_links

        # 8. sparse_pattern: select_pattern (mode: strided)
        # Select first input for mode local, second input for mode strided.
        # Mode is 'strided', so select strided_pattern.
        mode = self.spec['factors']['sparse_pattern']['attributes']['mode']
        pattern = local_pattern if mode == 'local' else strided_pattern

        # 9. sparse_scaled_query: dimension_scale (axis: -1, power: -0.5)
        # Multiply q by (size on axis -1) ** -0.5
        D = q.size(-1)
        scale = D ** -0.5
        scaled_query = q * scale

        # 10. sparse_scores: einsum (bhid,bhjd->bhij)
        scores = torch.einsum('bhid,bhjd->bhij', scaled_query, k)

        # 11. sparse_masked_scores: masked_logits (fill: -1000000000.0)
        # Keep scores where pattern is true, replace elsewhere by fill.
        # pattern shape [N, N], scores shape [B, H, N, N]
        # Broadcast pattern to [1, 1, N, N]
        fill_value = -1000000000.0
        masked_scores = torch.where(pattern.unsqueeze(0).unsqueeze(0), scores, torch.full_like(scores, fill_value))

        # 12. sparse_probabilities: softmax (axis: -1)
        probabilities = torch.softmax(masked_scores, dim=-1)

        # 13. sparse_output: einsum (bhij,bhjv->bhiv)
        output = torch.einsum('bhij,bhjv->bhiv', probabilities, v)

        return {
            "sparse_output": output,
            "sparse_pattern": pattern
        }

def build(spec: Dict[str, Any], resources: Dict[str, torch.Tensor]) -> torch.nn.Module:
    return SparseTransformer(spec)
