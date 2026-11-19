# RMSNorm

마지막 축 RMS 정규화, 제곱근 밖 epsilon, scale-only.

범위: full_last_axis_rms; outside_sqrt_epsilon; scale_only.

제외: partial_rmsnorm; offset_bias; mean_centering; mixed_precision; training; gradients; initialization; module_registration.

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json).

전체 모델, 학습, 성능 또는 공통 스키마 전체의 인증으로 해석하지 않습니다.

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [LayerNorm → RMSNorm](../../relations/tr_041.yaml) — A, tr_041
