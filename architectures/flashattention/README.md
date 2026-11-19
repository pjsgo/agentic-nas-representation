# FlashAttention

CPU 타일 전방 계산과 online softmax 누적, causal mask.

[복원 YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json)을 함께 읽습니다.

입력, dtype, shape, 파라미터와 상태의 계약은 YAML의 `interface`, `parameters`, `state`를 따릅니다. 제외 범위는 `scope.excluded`에 명시했습니다.

각 타일의 최대값에 맞춰 정규화 합과 가중합의 스케일을 조정합니다.

[아키텍처 목록](../README.md)

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Transformer → FlashAttention](../../relations/tr_054.yaml) — A, tr_054
