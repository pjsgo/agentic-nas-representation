# LSTMN

생성 원문 실패 후 별도 수정본 통과. 독립 참조 대표 사례 3개에서 선언 범위 내 검증을 통과했습니다.

범위: hidden_tape; cell_tape; previous_read_feedback; causal_additive_attention; weighted_hidden_and_cell_read; gates_from_read_hidden; cell_update_from_read_cell; append_states

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처와 한계](sources.yaml), [평가 이력](provenance.json)

전체 모델, 학습, 성능 재현은 주장하지 않습니다. [복원 범위와 검증](../../docs/reconstruction-status.md)을 함께 읽습니다.

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [LSTM → LSTMN](../../relations/tr_010.yaml) — A, tr_010
- [E2E Memory Network → LSTMN](../../relations/tr_011.yaml) — I, tr_011
- [LSTMN → Transformer](../../relations/tr_071.yaml) — I, tr_071
