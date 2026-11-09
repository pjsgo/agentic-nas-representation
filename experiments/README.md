# 실험 자료

YAML → 생성 입력 → 후보 코드 → 검증 → 결과를 따라 읽을 수 있도록 기존 산출물을 묶었습니다. `recorded/`는 당시 보고서이며, 서로 다른 범위의 판정을 하나의 성공률로 합치지 않습니다.

| 실험 | 당시 판정 | 포함 자료 | 현재 이용 방식 |
|---|---|---|---|
| [ResNet](resnet/README.md) | PASS | YAML·복원 코드·검증 코드·결과 | 오프라인 검증 실행 지원 |
| [Transformer](transformer/README.md) | FAIL | YAML·복원 코드·검증 코드·결과 | 자료 열람 |
| [LSTM](lstm/README.md) | FAIL | YAML·복원 코드·검증 코드·결과 | 자료 열람 |
| [Seq2Seq](seq2seq/README.md) | INCONCLUSIVE | YAML·복원 코드·검증 코드·결과 | 자료 열람 |
| [Linear Transformer](linear-transformer/README.md) | BLOCKED | YAML·사전 검증 코드·결과 (후보 없음) | 자료 열람 |
| [S4](s4/README.md) | NOT_RUN | YAML·사전 검증 코드·결과 (후보 없음) | 자료 열람 |
| [Mamba](mamba/README.md) | INCONCLUSIVE | YAML·복원 코드·검증 코드·결과 | 자료 열람 |
| [ResNet GELU 편집](resnet-gelu/README.md) | PASS | 편집 YAML·diff·복원 코드·검증 코드·결과 | 자료 열람 |

첫 예제는 [ResNet](resnet/README.md), 실패 분석 예제는 [Transformer](transformer/README.md)를 권합니다. 생성 입력에는 후보 생성 당시의 프롬프트와 모델 설정이 들어 있습니다. S4는 준비된 요청만 있고, Linear Transformer는 생성 후보가 없습니다.

각 폴더의 `validator.py`는 역사적 평가 구현입니다. ResNet 실행 안내 외의 스크립트는 원래 연구 환경의 의존성이 남아 있습니다. 외부 원본 소스와 바이너리 fixture를 모두 포함한 배포본은 아닙니다.

[파일 출처와 해시](manifest.json) · [표현 명세](../specs/schema_v0_3_candidate.md) · [업데이트 방식](../docs/updating.md)

Seq2Seq에는 [별도 구조 평가 v1.1 결과](seq2seq/followups/20261109_structure_v1_1/results.json)를 추가했습니다. 제한된 범위의 PASS이며 위 표의 원래 판정을 대체하지 않습니다.
