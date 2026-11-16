import torch
import math
from typing import Dict, Any

class PerformerAttention(torch.nn.Module):
    def __init__(self, spec: Dict[str, Any], resources: Dict[str, torch.Tensor]):
        super().__init__()
        # Register resources
        # omega is a buffer
        self.omega = torch.nn.Parameter(resources['omega'].clone(), requires_grad=False)
        # Note: The prompt says "Register parameter resources as trainable parameters and buffer resources as nontrainable persistent buffers".
        # The spec says omega is kind: buffer. So it should be a buffer.
        # However, nn.Parameter with requires_grad=False is also a valid way to hold a tensor that is part of the module state but not trained.
        # Let's use register_buffer to be strictly compliant with "buffer resources as nontrainable persistent buffers".
        # But wait, if I use register_buffer, it's not a Parameter. The prompt says "Register parameter resources as trainable parameters and buffer resources as nontrainable persistent buffers".
        # So I should use self.register_buffer('omega', resources['omega'].clone())

        # Let's re-evaluate. If I use register_buffer, it's fine.
        # I will clear the previous assignment and use register_buffer.
        del self.omega
        self.register_buffer('omega', resources['omega'].clone())

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Query Processing
        # performer_query_scaled: dimension_scale(q, axis=-1, power=-0.25)
        # Multiply first input by (its size on attributes.axis) raised to attributes.power.
        # q shape: [B, H, N, D]. axis -1 is D.
        q_scaled = q * (q.shape[-1] ** -0.25)

        # performer_query_projected: einsum(q_scaled, omega, 'bhnd,md->bhnm')
        q_projected = torch.einsum('bhnd,md->bhnm', q_scaled, self.omega)

        # performer_query_square: square(q_scaled)
        q_square = q_scaled ** 2

        # performer_query_norm: sum(q_square, axes=[-1], keepdim=True)
        q_norm = torch.sum(q_square, dim=-1, keepdim=True)

        # performer_query_half_norm: scale(q_norm, value=0.5)
        q_half_norm = q_norm * 0.5

        # performer_query_shift: amax(q_projected, axes=[-1])
        # Maximum over all attributes.axes simultaneously, retaining every reduced dimension as size one.
        # q_projected shape: [B, H, N, M]. axes=[-1] is M.
        # Result shape: [B, H, N, 1]
        q_shift = torch.amax(q_projected, dim=-1, keepdim=True)

        # performer_query_centered: sub(q_projected, q_half_norm)
        q_centered = q_projected - q_half_norm

        # performer_query_stable: sub(q_centered, q_shift)
        q_stable = q_centered - q_shift

        # performer_query_exp: exp(q_stable)
        q_exp = torch.exp(q_stable)

        # performer_query_epsilon: add_scalar(q_exp, value=1e-06)
        q_epsilon = q_exp + 1e-06

        # performer_query_features: resource_scale(q_epsilon, omega, axis=0, power=-0.5)
        # Multiply first input by (second input size on attributes.axis) raised to attributes.power.
        # omega shape: [M, D]. axis 0 is M.
        # q_epsilon shape: [B, H, N, M].
        # Scale factor: M ** -0.5
        q_features = q_epsilon * (self.omega.shape[0] ** -0.5)

        # 2. Key Processing
        # performer_key_scaled: dimension_scale(k, axis=-1, power=-0.25)
        k_scaled = k * (k.shape[-1] ** -0.25)

        # performer_key_projected: einsum(k_scaled, omega, 'bhnd,md->bhnm')
        k_projected = torch.einsum('bhnd,md->bhnm', k_scaled, self.omega)

        # performer_key_square: square(k_scaled)
        k_square = k_scaled ** 2

        # performer_key_norm: sum(k_square, axes=[-1], keepdim=True)
        k_norm = torch.sum(k_square, dim=-1, keepdim=True)

        # performer_key_half_norm: scale(k_norm, value=0.5)
        k_half_norm = k_norm * 0.5

        # performer_key_shift: amax(k_projected, axes=[-2, -1])
        # k_projected shape: [B, H, N, M]. axes=[-2, -1] are N and M.
        # Result shape: [B, H, 1, 1]
        k_shift = torch.amax(k_projected, dim=[-2, -1], keepdim=True)

        # performer_key_centered: sub(k_projected, k_half_norm)
        k_centered = k_projected - k_half_norm

        # performer_key_stable: sub(k_centered, k_shift)
        k_stable = k_centered - k_shift

        # performer_key_exp: exp(k_stable)
        k_exp = torch.exp(k_stable)

        # performer_key_epsilon: add_scalar(k_exp, value=1e-06)
        k_epsilon = k_exp + 1e-06

        # performer_key_features: resource_scale(k_epsilon, omega, axis=0, power=-0.5)
        k_features = k_epsilon * (self.omega.shape[0] ** -0.5)

        # 3. Aggregation
        # performer_kv_summary: einsum(k_features, v, 'bhnm,bhnv->bhmv')
        kv_summary = torch.einsum('bhnm,bhnv->bhmv', k_features, v)

        # performer_key_sum: sum(k_features, axes=[-2], keepdim=False)
        # k_features shape: [B, H, N, M]. axes=[-2] is N.
        # Result shape: [B, H, M]
        key_sum = torch.sum(k_features, dim=-2, keepdim=False)

        # performer_numerator: einsum(q_features, kv_summary, 'bhnm,bhmv->bhnv')
        numerator = torch.einsum('bhnm,bhmv->bhnv', q_features, kv_summary)

        # performer_denominator: einsum(q_features, key_sum, 'bhnm,bhm->bhn')
        denominator = torch.einsum('bhnm,bhm->bhn', q_features, key_sum)

        # performer_denominator_column: unsqueeze(denominator, axis=-1)
        denominator_column = denominator.unsqueeze(-1)

        # performer_output: divide(numerator, denominator_column)
        output = numerator / denominator_column

        return {
            "performer_output": output,
            "performer_query_features": q_features,
            "performer_key_features": k_features,
            "performer_denominator": denominator
        }

def build(spec: Dict[str, Any], resources: Dict[str, torch.Tensor]) -> torch.nn.Module:
    return PerformerAttention(spec, resources)
