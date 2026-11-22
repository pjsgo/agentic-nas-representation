# BigBird

외부 random graph를 받는 local/global/random block attention core.

[복원 YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json)을 함께 읽습니다.

입력, dtype, shape, 파라미터와 상태의 계약은 YAML의 `interface`, `parameters`, `state`를 따릅니다. 제외 범위는 `scope.excluded`에 명시했습니다.

## 관련 관계

관계 유형은 [관계 안내](../../relations/README.md)를 참고합니다. `?`는 추론 관계입니다.

- [Sparse Transformer → BigBird](../../relations/tr_035.yaml) — I, tr_035
