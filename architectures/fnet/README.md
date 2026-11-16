# FNet

LLM 생성 원본을 계산 수정 없이 독립 참조와 비교하여 통과했습니다. 대표 사례 3개의 선언된 범위 내 결과입니다.

범위: single_encoder_block; two_axis_fourier_mixing; post_norm_residuals; biased_ffn; tanh_gelu; deterministic_inference.

제외: embeddings; positional_embeddings; tokenizer; task_heads; stacked_model; training; gradients; initialization; stochastic_dropout; hybrid_attention; hardware_performance; official_float32_rounding.

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [생성 및 평가 이력](provenance.json).

전체 모델, 학습, 성능 또는 공통 스키마 전체의 인증으로 해석하지 않습니다.

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Transformer → FNet](../../relations/tr_045.yaml) — A, tr_045
