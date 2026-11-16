# pre-ln

LLM 생성 원본을 계산 수정 없이 독립 참조와 비교하여 통과했습니다. 대표 사례 3개의 선언된 범위 내 결과입니다.

범위: single_encoder_block; pre_attention_layernorm; multihead_self_attention; attention_residual; pre_ffn_layernorm; relu_ffn; ffn_residual; key_padding_mask.

제외: stack; stack_final_normalization; embeddings; positional_encoding; decoder; cross_attention; incremental_cache; dropout_training; gradients; optimizer; initialization; module_registration; mixed_precision.

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [생성 및 평가 이력](provenance.json).

전체 모델, 학습, 성능 또는 공통 스키마 전체의 인증으로 해석하지 않습니다.
