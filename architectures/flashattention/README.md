# FlashAttention

CPU 타일 전방 계산과 online softmax 누적, causal mask. 생성 원본 실패, 계산 수정 후 독립 참조 3건 비교 통과.

[복원 YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [생성 및 판정 기록](provenance.json)을 함께 읽습니다.

입력, dtype, shape, 파라미터와 상태의 계약은 YAML의 `interface`, `parameters`, `state`를 따릅니다. 제외 범위는 `scope.excluded`에 명시했습니다. 전체 모델 학습, gradient 동등성이나 성능 재현을 뜻하지 않습니다.

타일 점수와 누적 최대값, 정규화 합, 가중합의 재척도를 확인했습니다. GPU의 HBM/SRAM 배치, 커널 융합과 역전파 재계산은 복원 범위 밖입니다.

[아키텍처 목록](../README.md)

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Transformer → FlashAttention](../../relations/tr_054.yaml) — A, tr_054
