# Jamba

생성 원문 실패 후 별도 수정본 통과. 독립 참조 대표 사례 3개에서 선언 범위 내 검증을 통과했습니다.

범위: one_original_eight_layer_cycle; pre_norm_residuals; causal_grouped_query_attention; full_mamba_mixer; normalized_dt_B_C; dense_swiglu; full_capacity_top2_swiglu_experts; explicit_continuation_state

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처와 한계](sources.yaml), [평가 이력](provenance.json)

전체 모델, 학습, 성능 재현은 주장하지 않습니다. [복원 범위와 검증](../../docs/reconstruction-status.md)을 함께 읽습니다.

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Mamba → Jamba](../../relations/tr_064.yaml) — C, tr_064
- [Transformer → Jamba](../../relations/tr_065.yaml) — A, tr_065
- [Switch Transformer → Jamba](../../relations/tr_066.yaml) — C?, tr_066
