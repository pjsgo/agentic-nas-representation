# Switch Transformer

Top-1 routing, 용량 제한과 overflow drop, padding mask, ReLU expert.

범위: deterministic_argmax_eval; one_routing_group; full_softmax_selected_gate; explicit_capacity_rounding; ordered_dispatch; padding_mask; overflow_drop; bias_free_relu_experts.

제외: full_Transformer; residual; attention; normalization; training; gradients; load_balancing_loss; z_loss; jitter; sampling; dropout; token_embeddings_in_router; distributed_communication; mixed_precision; initialization_distribution.

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json).

전체 모델, 학습, 성능 또는 공통 스키마 전체의 인증으로 해석하지 않습니다.

## 관련 관계

관계 유형과 필드 설명은 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Sparse MoE → Switch Transformer](../../relations/tr_043.yaml) — C, tr_043
- [Transformer → Switch Transformer](../../relations/tr_044.yaml) — A, tr_044
- [Switch Transformer → Jamba](../../relations/tr_066.yaml) — C?, tr_066
