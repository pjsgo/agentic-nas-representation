# 관계 YAML 읽기

[관계 목록](../relations/README.md)에서 연결을 선택한 뒤, 출발·도착 아키텍처와 변경 이유를 읽습니다. `scope.source_yaml`과 `scope.target_yaml`을 열면 실제 부품 대응을 확인할 수 있습니다.

- `changes`와 `preserve`: 무엇을 바꾸고 무엇을 유지하는지
- `conditions`, `expected_effects`, `tradeoffs`: 적용 조건, 기대 효과, 비용
- `mapping_status`, `unresolved`: 부품 대응의 범위와 남아 있는 한계
- `application`, `historical_execution`: 공개 실행 가능 여부와 과거 연구 실행 근거의 구분

`complete_scoped`는 선택한 계산 범위의 대응이고, `partial`은 주요 개념 일부가 범위 밖에 있다는 뜻입니다. 설명된 관계가 모두 실행 가능한 변환인 것은 아닙니다. `?`로 표시된 관계는 추론이며 역사적 직접 계승으로 해석하지 않습니다.

아키텍처의 `genealogy/`는 관계에서 사용하는 부품 ID를 보존한 별도 표현입니다. 관계 파일 자체는 최상위 `relations/`에서 관리합니다.

[관계 필드·유형·부품 참조 규칙](../specs/relations/local-transition-v1.md) · [복원 YAML 읽기](representation-guide.md)
