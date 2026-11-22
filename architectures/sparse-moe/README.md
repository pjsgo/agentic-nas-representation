# Sparse MoE

결정적 추론, top-k routing, gate 정규화, ReLU expert 결합.

범위: deterministic_inference; top_k; selected_gate_normalization; sparse_dispatch; one_hidden_relu_experts; weighted_combine.

제외: noise; W_noise; importance_loss; load_loss; training; gradients; initialization_distribution; hierarchy; padding; capacity_limit; full_language_model; distributed_execution.

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json).

전체 모델, 학습, 성능 또는 공통 스키마 전체의 인증으로 해석하지 않습니다.

## 관련 관계

관계 유형과 필드 설명은 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Sparse MoE → Switch Transformer](../../relations/tr_043.yaml) — C, tr_043
