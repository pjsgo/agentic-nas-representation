# FNet

단일 encoder block, 2축 Fourier mixing, post-norm residual과 FFN.

범위: single_encoder_block; two_axis_fourier_mixing; post_norm_residuals; biased_ffn; tanh_gelu; deterministic_inference.

제외: embeddings; positional_embeddings; tokenizer; task_heads; stacked_model; training; gradients; initialization; stochastic_dropout; hybrid_attention; hardware_performance; official_float32_rounding.

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json).

전체 모델, 학습, 성능 또는 공통 스키마 전체의 인증으로 해석하지 않습니다.

## 관련 관계

관계 유형은 [관계 안내](../../relations/README.md)를 참고합니다. `?`는 추론 관계입니다.

- [Transformer → FNet](../../relations/tr_045.yaml) — A, tr_045
