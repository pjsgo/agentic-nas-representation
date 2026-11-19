# Longformer

여섯 투영을 포함한 양방향 window/global attention core.

[복원 YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json)을 함께 읽습니다.

입력, dtype, shape, 파라미터와 상태의 계약은 YAML의 `interface`, `parameters`, `state`를 따릅니다. 제외 범위는 `scope.excluded`에 명시했습니다.

[아키텍처 목록](../README.md)

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Sparse Transformer → Longformer](../../relations/tr_034.yaml) — I, tr_034
