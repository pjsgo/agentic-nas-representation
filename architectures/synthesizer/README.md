# synthesizer

LLM 생성 원본을 계산 수정 없이 독립 참조와 비교하여 통과했습니다. 대표 사례 3개의 선언된 범위 내 결과입니다.

범위: token_conditioned_logits_network; relu; multihead_values; fixed_max_length_prefix; boolean_mask; key_softmax; value_aggregation; output_projection.

제외: random_synthesizer; factorized_dense; factorized_random; mixtures; query_key_dot_product; transformer_stack; residual; normalization; ffn; embeddings; dropout; training; gradients; initialization; incremental_cache; cross_attention; relative_position_bias; extra_logit; grouped_query_heads; unit_scaling.

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [생성 및 평가 이력](provenance.json).

전체 모델, 학습, 성능 또는 공통 스키마 전체의 인증으로 해석하지 않습니다.
