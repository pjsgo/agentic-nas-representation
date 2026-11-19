# Synthesizer

토큰 조건부 logits, multihead value 집계, mask와 출력 투영.

범위: token_conditioned_logits_network; relu; multihead_values; fixed_max_length_prefix; boolean_mask; key_softmax; value_aggregation; output_projection.

제외: random_synthesizer; factorized_dense; factorized_random; mixtures; query_key_dot_product; transformer_stack; residual; normalization; ffn; embeddings; dropout; training; gradients; initialization; incremental_cache; cross_attention; relative_position_bias; extra_logit; grouped_query_heads; unit_scaling.

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json).

전체 모델, 학습, 성능 또는 공통 스키마 전체의 인증으로 해석하지 않습니다.

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Transformer → Synthesizer](../../relations/tr_040.yaml) — A, tr_040
