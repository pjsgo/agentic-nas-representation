# 작업 요약

2026-11-13 기준 **7종 아키텍처의 지정 핵심 계산 단위에서 복원 검증 완료**. Factorization으로 계산 흐름·반복·상태·객체 공유를 YAML에 표현하고 복원 코드와 비교했습니다.

| 아키텍처 | 완료 범위 | 판정·조건 |
|---|---|---|
| [ResNet](../experiments/resnet/README.md) | BasicBlock | 단일 BasicBlock·7개 입력 검증. 전체 ResNet 검증 아님. |
| [LSTM](../experiments/lstm/README.md) | 단방향 LSTM core, T≥1 | T≥1·어댑터 기반 검증 PASS. T=0 포함 원래 FAIL 및 CLI 연결 INCONCLUSIVE 유지. |
| [Seq2Seq](../experiments/seq2seq/README.md) | teacher-forcing core | 별도 구조 동등성 v1.1 PASS. 원래 raw DAG FAIL·Common v0.3 INCONCLUSIVE 유지. |
| [Transformer](../experiments/transformer/README.md) | 단일 encoder block | 새 응답 전에 동결한 runtime_trace_v1 PASS. 하네스는 이전 응답을 보고 개발됐고 새 코드도 이전 Treatment와 동일. |
| [Linear Transformer](../experiments/linear-transformer/README.md) | recurrent attention core | 실용 core PASS. 원본 Common Schema의 stateless Module identity BLOCKED 유지. |
| [S4](../experiments/s4/README.md) | recurrent/impulse-kernel core | 원문 대비 3줄 수정 후 엄격·실용 PASS. 원문 FAIL 보존. p0…p6 외부 Parameter 주입 필요; 전체 FFT/cache·학습 미검증. |
| [Mamba](../experiments/mamba/README.md) | selective-scan core | 실용 selective-scan PASS. 기존 exact graph FAIL·Common v0.3 INCONCLUSIVE 유지; 전체 Mamba block 미검증. |

각 평가 기준과 범위가 다르며, S4는 원문 대비 3줄 수정 후 엄격·실용 검사를 통과했습니다. 이전 실패 기록은 보존합니다. 전체 모델·학습/gradient나 공통 단일 엄격 기준의 7종 성공을 뜻하지 않습니다.

다음 단계는 계보의 변화 맥락과 Factorization을 에이전트에 함께 제공하고 EvoScientist 기반 구조 편집·실험 흐름으로 연결하는 것입니다.
