# 아키텍처 자료

각 폴더는 계산 구조, 사용 범위, 입출력 계약과 복원 YAML, 코드를 제공합니다. 관계는 별도 [관계 목록](../relations/README.md)에서 관리합니다.

| 아키텍처 | 대상 범위 | 복원 YAML | 구현 |
|---|---|---|---|
| [ResNet](resnet/README.md) | Identity shortcut BasicBlock, channels=64, CPU float32 eval | [YAML](resnet/representation.yaml) | [코드](resnet/reconstructed.py) |
| [LSTM](lstm/README.md) | Single-layer unidirectional core, T>=1; tested T=1,3,5, B=2,I=4,H=5 | [YAML](lstm/representation.yaml) | [코드](lstm/reconstructed.py) |
| [Seq2Seq](seq2seq/README.md) | Pre-embedded one-layer LSTM encoder/decoder, 100% teacher forcing, three CPU float32 fixtures | [YAML](seq2seq/representation.yaml) | [코드](seq2seq/reconstructed.py) |
| [Transformer](transformer/README.md) | Single encoder block, nine CPU float32 eval fixtures | [YAML](transformer/representation.yaml) | [코드](transformer/reconstructed.py) |
| [Linear Transformer](linear-transformer/README.md) | ELU+1 recurrent linear-attention core; external q/k/v and in-place S/Z state, four CPU float32 fixtures | [YAML](linear-transformer/representation.yaml) | [코드](linear-transformer/reconstructed.py) |
| [S4](s4/README.md) | Original rank-1 DPLR/NPLR recurrent/impulse-kernel core, Nfull=4,H=channels=1, CPU eval | [YAML](s4/representation.yaml) | [코드](s4/reconstructed.py) |
| [Mamba](mamba/README.md) | Mamba-1 selective-scan core, channels=3,state_size=2, four CPU float32 fixtures | [YAML](mamba/representation.yaml) | [코드](mamba/reconstructed.py) |
| [Simple RNN](simple-rnn/README.md) | Elman tanh 순환 core | [YAML](simple-rnn/representation.yaml) | [코드](simple-rnn/reconstructed.py) |
| [RNN Encoder–Decoder / GRU](gru-encoder-decoder/README.md) | Cho 2014 encoder, 고정 context, 조건부 GRU decoder | [YAML](gru-encoder-decoder/representation.yaml) | [코드](gru-encoder-decoder/reconstructed.py) |
| [Bahdanau Attention](bahdanau-attention/README.md) | 제공된 encoder memory에 대한 additive attention, 조건부 GRU | [YAML](bahdanau-attention/representation.yaml) | [코드](bahdanau-attention/reconstructed.py) |
| [Luong Attention](luong-attention/README.md) | global/general attention, LSTM, input feeding | [YAML](luong-attention/representation.yaml) | [코드](luong-attention/reconstructed.py) |
| [GNMT](gnmt/README.md) | 3층 encoder–decoder, 양방향 하단 encoder, residual, attention | [YAML](gnmt/representation.yaml) | [코드](gnmt/reconstructed.py) |
| [LayerNorm](layernorm/README.md) | 마지막 축의 population 통계, affine 정규화 | [YAML](layernorm/representation.yaml) | [코드](layernorm/reconstructed.py) |
| [Transformer-XL](transformer-xl/README.md) | segment recurrence, 상대 위치 attention sublayer | [YAML](transformer-xl/representation.yaml) | [코드](transformer-xl/reconstructed.py) |
| [Compressive Transformer](compressive-transformer/README.md) | 평균 풀링 압축, 두 단계 FIFO memory attention | [YAML](compressive-transformer/representation.yaml) | [코드](compressive-transformer/reconstructed.py) |
| [Sparse Transformer](sparse-transformer/README.md) | local/strided causal attention core | [YAML](sparse-transformer/representation.yaml) | [코드](sparse-transformer/reconstructed.py) |
| [Linformer](linformer/README.md) | sequence 축 low-rank projection attention core | [YAML](linformer/representation.yaml) | [코드](linformer/reconstructed.py) |
| [Performer](performer/README.md) | 고정 random features의 정규화 attention core | [YAML](performer/representation.yaml) | [코드](performer/reconstructed.py) |
| [LSSL](lssl/README.md) | 고정 HiPPO-LegS, bilinear discretization, dense recurrence | [YAML](lssl/representation.yaml) | [코드](lssl/reconstructed.py) |
| [Mamba-2 / SSD](mamba2-ssd/README.md) | scalar-identity multihead SSD chunk core | [YAML](mamba2-ssd/representation.yaml) | [코드](mamba2-ssd/reconstructed.py) |
| [H3](h3/README.md) | scalar-head shift, diagonal SSM gated block | [YAML](h3/representation.yaml) | [코드](h3/reconstructed.py) |
