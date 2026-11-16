# 복원, 관계 해석 명세

현재 YAML을 읽거나 같은 범위의 계산으로 다시 구현할 때 사용하는 규칙입니다. 과거 실험을 보관하는 archive가 아닙니다. 사람을 위한 읽는 순서는 [사용 안내](../docs/representation-guide.md)에 있습니다.

| 명세 | 사용하는 자료 | 역할 |
|---|---|---|
| [표현 읽기 규칙 v1](representation/reading-rules-v1.md) | 전체 자료를 읽는 공통 기준 | 연산과 연결, 자원, 상태, 범위 해석 |
| [RNN expression v1](operators/rnn-expression-v1.yaml) | Simple RNN, GRU Encoder–Decoder, Bahdanau, Luong, GNMT | expression, scan, 상태 전달, LSTM 연산 의미 |
| [Sparse attention v1](operators/sparse-transformer-v1.json) | Sparse Transformer | local/strided 연결과 mask 의미 |
| [Linformer attention v1](operators/linformer-v1.json) | Linformer | sequence projection, 압축 축 attention |
| [Performer attention v1](operators/performer-v1.json) | Performer | random feature, 집계 연산 |
| [Tensor trigonometric extension v1](operators/tensor-pytorch-trig-v1.md) | Transformer 위치 계산 | 기존 tensor profile에 cast, sin, cos 추가 |
| [Tensor PyTorch 2.2.1 / v0.3-r1](operators/tensor-pytorch-2.2.1-v0.3-r1.txt) | H3의 primitive 복원 표현 | tensor 연산, 호출, scan, 상태 의미 |
| [local_transition_v1](relations/local-transition-v1.md) | 관계 YAML | 관계 유형, 부품 참조, 범위, 실행 표시 |

이 목록은 기존 표현을 단일 범용 스키마로 통합하거나 인증하지 않습니다. 각 YAML의 버전과 선언이 기준이며, LayerNorm, Transformer-XL, Compressive Transformer, LSSL, SSD 등은 계산 수식과 계약을 YAML 내부에 명시합니다. 기존 다른 표현의 범위와 계약은 해당 아키텍처 자료를 함께 읽습니다.

모델별 크기, dtype, 외부 입력과 제외 범위는 아키텍처 YAML에 둡니다. 여러 모델이 공유하는 해석 규칙만 이곳에 모읍니다. 새로운 복원에서 일부를 제외했다고 해서 공통 명세를 반드시 바꾸지는 않습니다.

연산이나 상태의 계산 의미를 바꿀 때는 새 명세 버전을 만들고, 해당 YAML의 참조를 명시적으로 변경합니다. 설명과 링크를 정정하는 작업은 계산 의미의 변경과 구분합니다. 명세 확장만으로 미복원 계산을 완료 처리하지 않습니다.
