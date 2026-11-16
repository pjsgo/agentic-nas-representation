import torch
import math
from typing import Dict, Any

class LinformerModule(torch.nn.Module):
    def __init__(self, spec: Dict[str, Any], resources: Dict[str, torch.Tensor]):
        super().__init__()
        # Register resources
        # E is a parameter [R, Nmax]
        # F is a parameter [R, Nmax]
        self.E = torch.nn.Parameter(resources['E'].clone())
        self.F = torch.nn.Parameter(resources['F'].clone())

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Slice E and F based on sequence length N from k and v
        # k shape: [B, H, N, D]
        # v shape: [B, H, N, V]
        N = k.shape[2]

        # slice_sequence: Given matrix A[R, Nmax] and tensor X[B, H, N, C], return first N columns of A.
        # Result shape: [R, N]
        E_slice = self.E[:, :N]
        F_slice = self.F[:, :N]

        # 2. Compute sequence_key
        # einsum: 'rn,bhnd->bhrd'
        # E_slice: [R, N]
        # k: [B, H, N, D]
        # Result: [B, H, R, D]
        sequence_key = torch.einsum('rn,bhnd->bhrd', E_slice, k)

        # 3. Compute sequence_value
        # einsum: 'rn,bhnv->bhrv'
        # F_slice: [R, N]
        # v: [B, H, N, V]
        # Result: [B, H, R, V]
        sequence_value = torch.einsum('rn,bhnv->bhrv', F_slice, v)

        # 4. Scale query
        # dimension_scale: Multiply first input by (its size on attributes.axis) raised to attributes.power.
        # axis: -1 (D), power: -0.5
        # q shape: [B, H, N, D]
        D = q.shape[-1]
        scale_factor = D ** (-0.5)
        scaled_query = q * scale_factor

        # 5. Compute scores
        # einsum: 'bhnd,bhrd->bhnr'
        # scaled_query: [B, H, N, D]
        # sequence_key: [B, H, R, D]
        # Result: [B, H, N, R]
        scores = torch.einsum('bhnd,bhrd->bhnr', scaled_query, sequence_key)

        # 6. Softmax
        # axis: -1 (R)
        probabilities = torch.softmax(scores, dim=-1)

        # 7. Compute output
        # einsum: 'bhnr,bhrv->bhnv'
        # probabilities: [B, H, N, R]
        # sequence_value: [B, H, R, V]
        # Result: [B, H, N, V]
        output = torch.einsum('bhnr,bhrv->bhnv', probabilities, sequence_value)

        return {
            "linformer_output": output,
            "linformer_sequence_key": sequence_key,
            "linformer_sequence_value": sequence_value
        }

def build(spec: Dict[str, Any], resources: Dict[str, torch.Tensor]) -> torch.nn.Module:
    return LinformerModule(spec, resources)
