# 7개 아키텍처 복원 결과

**7개 아키텍처의 핵심 블록·코어 복원 완료.**

신경망 구조의 연산·연결·반복·상태 전달을 YAML로 표현하고, 실행 가능한 PyTorch 코드로 복원해 참조 구현과 비교했습니다.

| 아키텍처 | 복원 범위 | 검증 결과 | 표현 | 최종 구현 |
|---|---|---|---|---|
| **ResNet** | Identity-shortcut BasicBlock | 7개 입력 검증 통과 | [YAML](resnet/generator/input.yaml) | [코드](resnet/response_review/generated_model.py) |
| **LSTM** | 단방향 LSTM core, T≥1 | T=1/3/5 출력·상태 및 파라미터 공유 확인 | [YAML](lstm/representation.yaml) | [코드](lstm/generated_model.py) |
| **Seq2Seq** | 단일 layer teacher-forcing core | 3개 fixture의 계산·상태 전달 검증 통과 | [YAML](seq2seq/representation.yaml) | [코드](seq2seq/generated_model.py) |
| **Transformer** | 단일 encoder block | 9개 fixture의 출력·구조 검증 통과 | [YAML](transformer/20261113-core-completion/representation.yaml) | [코드](transformer/20261113-core-completion/generated_model.py) |
| **Linear Transformer** | ELU+1 recurrent attention core | 출력·S/Z 상태 전달 검증 통과 | [YAML](linear-transformer/20261113-core-completion/representation.yaml) | [코드](linear-transformer/20261113-core-completion/generated_model.py) |
| **S4** | Rank-1 DPLR/NPLR recurrent·impulse core | 엄격·실용 검증 통과 | [YAML](s4/20261113-core-completion/representation.yaml) | [코드](s4/20261113-core-completion/repaired_model.py) |
| **Mamba** | Selective-scan core | 4개 fixture의 출력·시점별 상태 검증 통과 | [YAML](mamba/representation.yaml) | [코드](mamba/generated_model.py) |

완료 범위는 표에 명시한 블록·코어와 평가 조건입니다. 전체 모델 학습 성능은 포함하지 않습니다.

각 코드는 평가에 사용한 최종 구현입니다. 실행에 필요한 파라미터와 입력 조건은 아키텍처별 README 및 코드의 인터페이스를 참고하세요.

[프로젝트 소개](../README.md) · [결과 요약](../results/summary.md) · [파일 출처](manifest.json)
