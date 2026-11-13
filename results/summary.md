# 복원 결과 요약

**7개 아키텍처의 핵심 블록·코어 복원 완료.**

신경망 구조의 연산·연결·반복·상태 전달을 YAML로 표현하고, 실행 가능한 PyTorch 코드로 복원해 참조 구현과 비교했습니다.

| 아키텍처 | 복원 범위 | 검증 결과 | 표현 | 최종 구현 |
|---|---|---|---|---|
| [ResNet](../experiments/resnet/README.md#구조도) | Identity-shortcut BasicBlock | 7개 입력 검증 통과 | [YAML](../experiments/resnet/generator/input.yaml) | [코드](../experiments/resnet/response_review/generated_model.py) |
| [LSTM](../experiments/lstm/README.md#구조도) | 단방향 LSTM core, T≥1 | T=1/3/5 출력·상태 및 파라미터 공유 확인 | [YAML](../experiments/lstm/representation.yaml) | [코드](../experiments/lstm/generated_model.py) |
| [Seq2Seq](../experiments/seq2seq/README.md#구조도) | 단일 layer teacher-forcing core | 3개 fixture의 계산·상태 전달 검증 통과 | [YAML](../experiments/seq2seq/representation.yaml) | [코드](../experiments/seq2seq/generated_model.py) |
| [Transformer](../experiments/transformer/README.md#구조도) | 단일 encoder block | 9개 fixture의 출력·구조 검증 통과 | [YAML](../experiments/transformer/20261113-core-completion/representation.yaml) | [코드](../experiments/transformer/20261113-core-completion/generated_model.py) |
| [Linear Transformer](../experiments/linear-transformer/README.md#구조도) | ELU+1 recurrent attention | 출력·S/Z 상태 전달 검증 통과 | [YAML](../experiments/linear-transformer/20261113-core-completion/representation.yaml) | [코드](../experiments/linear-transformer/20261113-core-completion/generated_model.py) |
| [S4](../experiments/s4/README.md#구조도) | Rank-1 DPLR/NPLR recurrent·impulse core | 엄격·실용 검증 통과 | [YAML](../experiments/s4/20261113-core-completion/representation.yaml) | [코드](../experiments/s4/20261113-core-completion/repaired_model.py) |
| [Mamba](../experiments/mamba/README.md#구조도) | Selective-scan core | 4개 fixture의 출력·시점별 상태 검증 통과 | [YAML](../experiments/mamba/representation.yaml) | [코드](../experiments/mamba/generated_model.py) |

완료 범위는 표에 명시한 블록·코어와 평가 조건입니다. 전체 모델 학습 성능은 포함하지 않습니다.

검증은 핵심 계산, 참조 출력과의 일치, 파라미터 관계 및 상태 전달을 대상으로 했습니다. 다음 단계는 이 구현들을 기반으로 계보 연결과 복원 범위를 확장하는 것입니다.

[평가 조건·출처·해시](summary.json) · [프로젝트 소개](../README.md)
