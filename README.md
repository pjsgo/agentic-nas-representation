# Agentic NAS · Architecture Reconstruction

신경망 구조를 YAML로 기술하고 코드로 복원하며, 아키텍처 사이의 계승·변경 관계를 구조 탐색에 활용합니다.

## 자료 읽기

- [복원 YAML 안내](docs/representation-guide.md): 모델 내부의 연산·연결·파라미터·입출력·상태
- [관계 YAML 안내](docs/genealogy-guide.md): 모델 사이의 변경·보존·부품 대응과 근거
- [관계 목록](relations/README.md): 출발·도착 아키텍처별 관계 YAML

복원 범위는 전체 모델 또는 명시된 block/core입니다. 코드의 생성·수정 출처와 입력 계약은 각 아키텍처 자료를 따릅니다. 설명용 관계를 실행 가능한 변환으로 해석하지 않습니다.

## 아키텍처 자료

| 아키텍처 | 대상 범위 | 복원 YAML | 구현 |
|---|---|---|---|
| [ResNet](experiments/resnet/README.md) | Identity shortcut BasicBlock, channels=64, CPU float32 eval | [YAML](experiments/resnet/generator/input.yaml) | [코드](experiments/resnet/response_review/generated_model.py) |
| [LSTM](experiments/lstm/README.md) | Single-layer unidirectional core, T>=1; tested T=1,3,5, B=2,I=4,H=5 | [YAML](experiments/lstm/representation.yaml) | [코드](experiments/lstm/generated_model.py) |
| [Seq2Seq](experiments/seq2seq/README.md) | Pre-embedded one-layer LSTM encoder/decoder, 100% teacher forcing, three CPU float32 fixtures | [YAML](experiments/seq2seq/representation.yaml) | [코드](experiments/seq2seq/generated_model.py) |
| [Transformer](experiments/transformer/README.md) | Single encoder block, nine CPU float32 eval fixtures | [YAML](experiments/transformer/20261113-core-completion/representation.yaml) | [코드](experiments/transformer/20261113-core-completion/generated_model.py) |
| [Linear Transformer](experiments/linear-transformer/README.md) | ELU+1 recurrent linear-attention core; external q/k/v and in-place S/Z state, four CPU float32 fixtures | [YAML](experiments/linear-transformer/20261113-core-completion/representation.yaml) | [코드](experiments/linear-transformer/20261113-core-completion/generated_model.py) |
| [S4](experiments/s4/README.md) | Original rank-1 DPLR/NPLR recurrent/impulse-kernel core, Nfull=4,H=channels=1, CPU eval | [YAML](experiments/s4/20261113-core-completion/representation.yaml) | [코드](experiments/s4/20261113-core-completion/repaired_model.py) |
| [Mamba](experiments/mamba/README.md) | Mamba-1 selective-scan core, channels=3,state_size=2, four CPU float32 fixtures | [YAML](experiments/mamba/representation.yaml) | [코드](experiments/mamba/generated_model.py) |
| [Simple RNN](experiments/simple-rnn/README.md) | Elman tanh 순환 core | [YAML](experiments/simple-rnn/representation.yaml) | [코드](experiments/simple-rnn/generated_model.py) |
| [RNN Encoder–Decoder / GRU](experiments/gru-encoder-decoder/README.md) | Cho 2014 encoder·고정 context·조건부 GRU decoder | [YAML](experiments/gru-encoder-decoder/representation.yaml) | [코드](experiments/gru-encoder-decoder/generated_model.py) |
| [Bahdanau Attention](experiments/bahdanau-attention/README.md) | 제공된 encoder memory에 대한 additive attention·조건부 GRU | [YAML](experiments/bahdanau-attention/representation.yaml) | [코드](experiments/bahdanau-attention/repaired_model.py) |
| [Luong Attention](experiments/luong-attention/README.md) | global/general attention·LSTM·input feeding | [YAML](experiments/luong-attention/representation.yaml) | [코드](experiments/luong-attention/repaired_model.py) |
| [GNMT](experiments/gnmt/README.md) | 3층 encoder–decoder·양방향 하단 encoder·residual·attention | [YAML](experiments/gnmt/representation.yaml) | [코드](experiments/gnmt/repaired_model.py) |
| [LayerNorm](experiments/layernorm/README.md) | 마지막 축의 population 통계·affine 정규화 | [YAML](experiments/layernorm/representation.yaml) | [코드](experiments/layernorm/generated_model.py) |
| [Transformer-XL](experiments/transformer-xl/README.md) | segment recurrence·상대 위치 attention sublayer | [YAML](experiments/transformer-xl/representation.yaml) | [코드](experiments/transformer-xl/repaired_model.py) |
| [Compressive Transformer](experiments/compressive-transformer/README.md) | 평균 풀링 압축·두 단계 FIFO memory attention | [YAML](experiments/compressive-transformer/representation.yaml) | [코드](experiments/compressive-transformer/repaired_model.py) |
| [Sparse Transformer](experiments/sparse-transformer/README.md) | local/strided causal attention core | [YAML](experiments/sparse-transformer/representation.yaml) | [코드](experiments/sparse-transformer/repaired_model.py) |
| [Linformer](experiments/linformer/README.md) | sequence 축 low-rank projection attention core | [YAML](experiments/linformer/representation.yaml) | [코드](experiments/linformer/generated_model.py) |
| [Performer](experiments/performer/README.md) | 고정 random features의 정규화 attention core | [YAML](experiments/performer/representation.yaml) | [코드](experiments/performer/generated_model.py) |
| [LSSL](experiments/lssl/README.md) | 고정 HiPPO-LegS·bilinear discretization·dense recurrence | [YAML](experiments/lssl/representation.yaml) | [코드](experiments/lssl/repaired_model.py) |
| [Mamba-2 / SSD](experiments/mamba2-ssd/README.md) | scalar-identity multihead SSD chunk core | [YAML](experiments/mamba2-ssd/representation.yaml) | [코드](experiments/mamba2-ssd/repaired_model.py) |
| [H3](experiments/h3/README.md) | scalar-head shift·diagonal SSM gated block | [YAML](experiments/h3/representation.yaml) | [코드](experiments/h3/generated_model.py) |
