# 복원 YAML 읽기

아키텍처 폴더의 README에서 복원 범위와 사용 계약을 먼저 확인하고, `representation.yaml`과 `reconstructed.py`를 함께 읽습니다.

1. YAML의 표현 버전과 연산 규칙을 확인합니다. [명세 목록](../specs/README.md)에서 공통 규칙을 찾을 수 있습니다.
2. 입출력의 크기·축·dtype와 초기 상태를 확인합니다. 입력·가중치의 실제 값은 호출자가 준비합니다.
3. 연산과 연결을 따라가며 반복·파라미터 공유·상태 갱신을 읽습니다.
4. `provenance.json`에서 확인 범위와 생성 원본/수정본을 구분합니다. 논문·공식 코드의 근거는 해당 자료의 출처 기록을 확인합니다.

복원 YAML은 어떤 계산을 연결할지 정하고, 연산 명세는 그 계산의 정확한 의미를 정합니다. 예를 들어 `scan`의 반복 축과 상태 전달은 이름에서 추측하지 않고 선언된 규칙을 따릅니다.

[복원 방법과 판정 기준](reconstruction-method.md) · [표현 읽기 규칙](../specs/representation/reading-rules-v1.md) · [관계 읽기](genealogy-guide.md) · [아키텍처 목록](../architectures/README.md)
