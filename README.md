# Agentic NAS · Architecture Reconstruction

**7개 아키텍처의 핵심 블록·코어 복원 완료.**

신경망 구조의 연산·연결·반복·상태 전달을 YAML로 표현하고, 실행 가능한 PyTorch 코드로 복원해 참조 구현과 비교했습니다.

## 복원 결과

| 아키텍처 | 복원 범위 | 검증 결과 | 표현 | 최종 구현 |
|---|---|---|---|---|
| **ResNet** | Identity-shortcut BasicBlock | 7개 입력 검증 통과 | [YAML](experiments/resnet/generator/input.yaml) | [코드](experiments/resnet/response_review/generated_model.py) |
| **LSTM** | 단방향 LSTM core, T≥1 | T=1/3/5 출력·상태 및 파라미터 공유 확인 | [YAML](experiments/lstm/representation.yaml) | [코드](experiments/lstm/generated_model.py) |
| **Seq2Seq** | 단일 layer teacher-forcing core | 3개 fixture의 계산·상태 전달 검증 통과 | [YAML](experiments/seq2seq/representation.yaml) | [코드](experiments/seq2seq/generated_model.py) |
| **Transformer** | 단일 encoder block | 9개 fixture의 출력·구조 검증 통과 | [YAML](experiments/transformer/20261113-core-completion/representation.yaml) | [코드](experiments/transformer/20261113-core-completion/generated_model.py) |
| **Linear Transformer** | ELU+1 recurrent attention core | 출력·S/Z 상태 전달 검증 통과 | [YAML](experiments/linear-transformer/20261113-core-completion/representation.yaml) | [코드](experiments/linear-transformer/20261113-core-completion/generated_model.py) |
| **S4** | Rank-1 DPLR/NPLR recurrent·impulse core | 엄격·실용 검증 통과 | [YAML](experiments/s4/20261113-core-completion/representation.yaml) | [코드](experiments/s4/20261113-core-completion/repaired_model.py) |
| **Mamba** | Selective-scan core | 4개 fixture의 출력·시점별 상태 검증 통과 | [YAML](experiments/mamba/representation.yaml) | [코드](experiments/mamba/generated_model.py) |

완료 범위는 표에 명시한 블록·코어와 평가 조건입니다. 전체 모델 학습 성능은 포함하지 않습니다.

## 계보 복원으로 확장

검증된 7개 구현을 기반으로 아키텍처 계보의 연결과 복원 범위를 넓힙니다. 계보의 변화 맥락과 Factorization을 함께 활용해 구조 선택·편집과 실험으로 이어가는 것이 다음 목표입니다.

[아키텍처별 안내](experiments/README.md) · [결과 요약](results/summary.md) · [자료 출처](docs/artifact-provenance.md)

*결과 기준일: 2026-11-13*
