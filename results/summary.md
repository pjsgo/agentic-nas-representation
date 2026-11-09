# 검증 결과 요약

근거 확인일: 2026-11-09 (한국 시간). 범위가 서로 다른 제한된 실험과 사전 준비 결과를 모았습니다. 동일한 조건의 벤치마크나 7개 전체 모델의 성공률을 뜻하지 않습니다.

## 판정 표시의 의미

| 표시 | 한국어 의미 | 해석 |
|---|---|---|
| PASS | 통과 | 선언한 평가 범위 안에서 요구 조건을 확인함 |
| FAIL | 실패 | 기록된 평가에서 필수 조건을 충족하지 못함 |
| INCONCLUSIVE | 판정 유보 | 선행 조건이나 검증기 문제가 해결되지 않아 해당 주장을 확정할 수 없음 |
| BLOCKED | 진행 조건 미충족 | 필요한 조건이 충족되지 않아 선언한 목표로 진행할 수 없음 |
| NOT_RUN | 미실행 | 해당 단계를 아직 수행하지 않음 |

## 아키텍처별 결과

| 아키텍처 | 검증 범위 | 현재 판정 | 주요 결과 |
|---|---|---|---|
| [ResNet](../case-studies/resnet.md) | BasicBlock 및 지정된 활성화 함수 편집 | 통과 (PASS) | 기준 모델 7/7개, 편집 기준 구현과의 비교 8/8개 통과 |
| [Transformer](../case-studies/transformer.md) | 연산 단위 인코더 블록 | 실패 (FAIL) | 조건부 수치 비교는 통과했지만 Parameter 요건 위반 |
| [LSTM](../case-studies/lstm.md) | 분해된 셀의 시퀀스 반복 | 실패 (FAIL) | T≥1 별도 평가는 통과, 원래 T=0 실패는 유지 |
| [Seq2Seq](../case-studies/seq2seq.md) | 임베딩된 입력을 받는 인코더·디코더 핵심부 | 판정 유보 (INCONCLUSIVE) | 수치·상태 검사는 통과, 고정된 실행 그래프 검사는 실패 |
| [Linear Transformer](../case-studies/linear-transformer.md) | 순환형 어텐션 핵심부 | 진행 조건 미충족 (BLOCKED) | 수치 사전 검증은 통과, 원본 객체 동일성 표현은 미해결 |
| [S4](../case-studies/s4.md) | 제한된 rank-1 DPLR 핵심부 | 미실행 (NOT_RUN) | 사전 검증 통과, 생성 코드 복원은 미실행 |
| [Mamba](../case-studies/mamba.md) | Selective-scan 핵심부 | 판정 유보 (INCONCLUSIVE) | 진단용 수치 검사는 통과, 고정된 구조 특징 검사는 실패 |

## 결과를 읽을 때의 주의점

- ResNet의 통과는 하나의 BasicBlock 기준 실험과 별도 편집 실험에 해당하며, 전체 ResNet의 복원을 뜻하지 않습니다.
- Transformer의 조건부 수치 비교가 통과했어도 Parameter 요건 위반으로 전체 판정은 실패입니다.
- LSTM의 원래 전체 범위 실패와 이후 T≥1 범위 통과는 서로 다른 평가 기록입니다.
- Seq2Seq와 Mamba는 사전 고정한 구조 검사 실패와 전체 판정 유보를 그대로 유지합니다.
- Linear Transformer의 수치 사전 검증 통과와 원본 전체 구조 표현의 진행 조건 미충족은 함께 성립할 수 있습니다.
- S4의 사전 검증 통과는 생성 코드 복원 성공을 뜻하지 않습니다.

전체 성공률은 계산하지 않습니다. 각 사례 문서에서 제외 범위와 다음 작업을 설명합니다. [summary.json](summary.json)은 프로그램에서도 읽을 수 있도록 작성한 요약이며, 검증기의 원본 출력은 아닙니다. JSON의 키와 상태 코드는 연결성을 위해 영문으로 유지하고 설명은 한국어로 작성했습니다.

[프로젝트 소개](../README.md)

## 공개 예제 재실행

2026-11-09에 ResNet 기준 후보를 원본 검증기로 다시 실행했고 7개 입력과 구조 검사를 통과했습니다. [재실행 결과](reproduction/resnet-20261109/summary.json)는 당시 기록과 별도로 보존합니다. 다른 계열과 GELU 편집은 이번에 재실행하지 않았습니다.

## Seq2Seq 후속 평가

2026-11-09의 별도 구조 평가 v1.1에서는 동일 후보가 고정 CPU float32 조건 3개에서 PASS를 받았습니다. 새로 생성한 0 상수 내부의 backend detach만 제한적으로 다루며, 원래 INCONCLUSIVE 판정은 유지합니다. [보고서](../experiments/seq2seq/followups/20261109_structure_v1_1/results.json)
