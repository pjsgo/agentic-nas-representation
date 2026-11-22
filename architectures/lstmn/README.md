# LSTMN

Hidden/cell tape, causal additive attention, read 기반 gate와 상태 갱신.

범위: hidden_tape; cell_tape; previous_read_feedback; causal_additive_attention; weighted_hidden_and_cell_read; gates_from_read_hidden; cell_update_from_read_cell; append_states

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처와 한계](sources.yaml), [구현 기록](provenance.json)

## 관련 관계

관계 유형과 필드 설명은 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [LSTM → LSTMN](../../relations/tr_010.yaml) — A, tr_010
- [E2E Memory Network → LSTMN](../../relations/tr_011.yaml) — I, tr_011
- [LSTMN → Transformer](../../relations/tr_071.yaml) — I, tr_071
