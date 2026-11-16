# switch-transformer

생성 원본은 실패했으며 별도 수정본이 독립 참조 검사를 통과했습니다. 대표 사례 3개의 선언된 범위 내 결과입니다.

범위: deterministic_argmax_eval; one_routing_group; full_softmax_selected_gate; explicit_capacity_rounding; ordered_dispatch; padding_mask; overflow_drop; bias_free_relu_experts.

제외: full_Transformer; residual; attention; normalization; training; gradients; load_balancing_loss; z_loss; jitter; sampling; dropout; token_embeddings_in_router; distributed_communication; mixed_precision; initialization_distribution.

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [생성 및 평가 이력](provenance.json).

전체 모델, 학습, 성능 또는 공통 스키마 전체의 인증으로 해석하지 않습니다.
