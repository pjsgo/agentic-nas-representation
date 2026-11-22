# E2E Memory Network

Memory와 query embedding, soft addressing, multi-hop read, tied 답변 투영.

범위: input_memory_embedding; output_memory_embedding; query_embedding; soft_addressing; weighted_read; hop_query_update; tied_answer_projection

[YAML](representation.yaml), [최종 코드](reconstructed.py), [출처와 한계](sources.yaml), [구현 기록](provenance.json)

## 관련 관계

관계 유형과 필드 설명은 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Memory Network → E2E Memory Network](../../relations/tr_009.yaml) — A, tr_009
- [E2E Memory Network → LSTMN](../../relations/tr_011.yaml) — I, tr_011
- [E2E Memory Network → Transformer](../../relations/tr_070.yaml) — I, tr_070
