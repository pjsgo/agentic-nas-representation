# Linear Transformer 실험 자료

순환형 선형 어텐션 핵심부와 별도로 구성한 연산 분해 표현 · **BLOCKED**

[YAML](representation.yaml) → 후보 생성 미실행 → [검증 코드](validator.py) → [기록된 결과](recorded/results.json)

원본·독립 기준 구현 간 비교 158개에서 최대 절대·상대 오차 0, 생성 후보는 없음

기존 연산만 사용한 YAML 분해 표현은 반복 계산 조건 4개에서 허용오차 내 검증을 통과했다. 현재 기능적 스키마는 원본에서 공유하는 feature-map Module의 객체 동일성을 표현하지 못한다.

## 재현 범위

현재 폴더는 실제 실험 자료를 읽고 비교하는 스냅샷입니다. 검증기는 원래 작업 공간의 참조 모듈·fixture·정책 파일에 의존하므로 이 폴더에서 직접 실행하지 않습니다. 전체 평가를 독립 실행 가능한 묶음으로 만드는 작업은 후속 단계입니다. [실행 가능한 ResNet 예제](../resnet/README.md)는 별도로 제공합니다.

[후속 연산 분해 YAML](followup_20261105_schema_numerical_v1/representation_decomposed_v0_3.yaml)과 [후속 결과](followup_20261105_schema_numerical_v1/validation/results.json)는 최초 표현·평가와 구분합니다.

[사례 해설](../../case-studies/linear-transformer.md) · [실험 목록](../README.md)
