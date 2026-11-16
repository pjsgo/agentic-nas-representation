# BigBird

외부 random graph를 받는 local/global/random block attention core. 생성 원본 실패, 계산 수정 후 독립 참조 3건 비교 통과.

[복원 YAML](representation.yaml), [최종 코드](reconstructed.py), [출처](sources.yaml), [생성 및 판정 기록](provenance.json)을 함께 읽습니다.

입력, dtype, shape, 파라미터와 상태의 계약은 YAML의 `interface`, `parameters`, `state`를 따릅니다. 제외 범위는 `scope.excluded`에 명시했습니다. 전체 모델 학습, gradient 동등성이나 성능 재현을 뜻하지 않습니다.

[아키텍처 목록](../README.md)
