# RMSNorm

LLM 생성 원본을 계산 수정 없이 독립 참조와 비교하여 통과했습니다. 대표 사례 3개의 선언된 범위 내 결과입니다.

범위: full_last_axis_rms; outside_sqrt_epsilon; scale_only.

제외: partial_rmsnorm; offset_bias; mean_centering; mixed_precision; training; gradients; initialization; module_registration.

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [생성 및 평가 이력](provenance.json).

전체 모델, 학습, 성능 또는 공통 스키마 전체의 인증으로 해석하지 않습니다.

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [LayerNorm → RMSNorm](../../relations/tr_041.yaml) — A, tr_041
