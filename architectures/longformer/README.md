# Longformer

여섯 투영을 포함한 양방향 window/global attention core. 생성 원본 실패, 계산 수정 후 독립 참조 3건 비교 통과.

[복원 YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [생성 및 판정 기록](provenance.json)을 함께 읽습니다.

입력, dtype, shape, 파라미터와 상태의 계약은 YAML의 `interface`, `parameters`, `state`를 따릅니다. 제외 범위는 `scope.excluded`에 명시했습니다. 전체 모델 학습, gradient 동등성이나 성능 재현을 뜻하지 않습니다.

세션 간 일정 조율 문제로 동일 대상 요청이 총 2건 전송되었습니다. 가장 먼저 받은 원본의 실패와 그 수정본을 기준으로 판정했으며, 단일 시도 정책 준수를 주장하지 않습니다.

[아키텍처 목록](../README.md)

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Sparse Transformer → Longformer](../../relations/tr_034.yaml) — I, tr_034
