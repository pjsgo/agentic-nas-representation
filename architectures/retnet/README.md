# RetNet

생성 원문 실패 후 별도 수정본 통과. 독립 참조 대표 사례 3개에서 선언 범위 내 검증을 통과했습니다.

범위: multi_scale_fixed_head_decay; independent_projections; real_position_rotation; key_scaling; causal_retention; parallel_sum; recurrent_matrix_state; state_continuation

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처와 한계](sources.yaml), [평가 이력](provenance.json)

전체 모델, 학습, 성능 재현은 주장하지 않습니다. [복원 범위와 검증](../../docs/reconstruction-status.md)을 함께 읽습니다.

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Transformer → RetNet](../../relations/tr_058.yaml) — X, tr_058
- [Linear Transformer → RetNet](../../relations/tr_060.yaml) — I?, tr_060
