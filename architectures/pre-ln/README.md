# Pre-LN

Pre-LN encoder block, attention과 FFN residual, key padding mask.

범위: single_encoder_block; pre_attention_layernorm; multihead_self_attention; attention_residual; pre_ffn_layernorm; relu_ffn; ffn_residual; key_padding_mask.

제외: stack; stack_final_normalization; embeddings; positional_encoding; decoder; cross_attention; incremental_cache; dropout_training; gradients; optimizer; initialization; module_registration; mixed_precision.

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json).

전체 모델, 학습, 성능 또는 공통 스키마 전체의 인증으로 해석하지 않습니다.

## 관련 관계

관계 유형은 [관계 안내](../../relations/README.md)를 참고합니다. `?`는 추론 관계입니다.

- [Transformer → Pre-LN](../../relations/tr_042.yaml) — C, tr_042
