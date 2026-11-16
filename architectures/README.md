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
| [Longformer](longformer/README.md) | 여섯 투영을 포함한 양방향 window/global attention core | [YAML](longformer/representation.yaml) | [코드](longformer/reconstructed.py) |
| [BigBird](bigbird/README.md) | 외부 random graph를 받는 local/global/random block attention core | [YAML](bigbird/representation.yaml) | [코드](bigbird/reconstructed.py) |
| [FlashAttention](flashattention/README.md) | CPU 타일 전방 계산과 online softmax 누적, causal mask | [YAML](flashattention/representation.yaml) | [코드](flashattention/reconstructed.py) |
| [GQA](gqa/README.md) | 외부 QKV의 query-to-KV 그룹 대응, causal prefix와 G-head cache | [YAML](gqa/representation.yaml) | [코드](gqa/reconstructed.py) |

<!-- phase3 -->

3차 위치 표현 3개는 연구용 실현과 외부 LLM 생성 코드의 독립 참조 검사를 통과했습니다. 생성 원본 통과 3개, 수정 후 통과 0개이며 세부 이력은 각 provenance에 기록했습니다.

| 노드 | 복원 범위 | 자료 |
|---|---|---|
| Relative Position | 부호 있는 거리 clipping과 상대 key/value를 포함한 projected self-attention core | [자료](relative-position/README.md) |
| RoPE | 명시 위치의 인접 feature 쌍 회전과 attention core | [자료](rope/README.md) |
| ALiBi | 고정 head slope와 causal 거리 bias attention core | [자료](alibi/README.md) |

- [rmsnorm](rmsnorm/README.md): LLM 생성 원본을 계산 수정 없이 독립 참조와 비교하여 통과했습니다.

- [pre-ln](pre-ln/README.md): LLM 생성 원본을 계산 수정 없이 독립 참조와 비교하여 통과했습니다.

- [sparse-moe](sparse-moe/README.md): 생성 원본은 실패했으며 별도 수정본이 독립 참조 검사를 통과했습니다.

- [switch-transformer](switch-transformer/README.md): 생성 원본은 실패했으며 별도 수정본이 독립 참조 검사를 통과했습니다.

- [reformer](reformer/README.md): 생성 원본은 실패했으며 별도 수정본이 독립 참조 검사를 통과했습니다.

- [synthesizer](synthesizer/README.md): LLM 생성 원본을 계산 수정 없이 독립 참조와 비교하여 통과했습니다.

- [fnet](fnet/README.md): LLM 생성 원본을 계산 수정 없이 독립 참조와 비교하여 통과했습니다.
