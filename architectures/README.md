# 아키텍처 목록

53개 모델의 `representation.yaml`에는 모두 `architecture`(모델)와 `interface`(입출력 규격)가 있습니다. 나머지 필드는 모델과 표현 형식에 따라 달라집니다.

| 주요 필드 | 의미 |
|---|---|
| `scope`, `architecture_entry` | 복원 범위와 계산 시작점 |
| `representation_version`, `operator_profile` | 표현 버전과 연산 해석 규칙 |
| `factors`, `graph`, `functions` | 계산을 구성하는 연산, 그래프 또는 함수 |
| `contains`, `dataflow`, `scan` | 부품의 포함 관계, 값이 흐르는 연결, 반복 계산 |
| `parameters`, `buffers`, `state` | 학습 파라미터, 보관하는 값, 호출 간 전달하는 상태 |

포함 관계는 `contains`에서, 계산 순서는 데이터 의존성과 명시된 호출·반복에서 읽습니다.

## 모델 목록

| 아키텍처 | 대상 범위 | 복원 YAML | 구현 |
|---|---|---|---|
| [ALiBi](alibi/README.md) | 고정 head slope와 causal 거리 bias attention core | [YAML](alibi/representation.yaml) | [코드](alibi/reconstructed.py) |
| [Bahdanau Attention](bahdanau-attention/README.md) | 제공된 encoder memory에 대한 additive attention, 조건부 GRU | [YAML](bahdanau-attention/representation.yaml) | [코드](bahdanau-attention/reconstructed.py) |
| [BigBird](bigbird/README.md) | 외부 random graph를 받는 local/global/random block attention core | [YAML](bigbird/representation.yaml) | [코드](bigbird/reconstructed.py) |
| [ByteNet](bytenet/README.md) | ReLU bottleneck residual feature stack, causal 또는 centered convolution | [YAML](bytenet/representation.yaml) | [코드](bytenet/reconstructed.py) |
| [Compressive Transformer](compressive-transformer/README.md) | 평균 풀링 압축, 두 단계 FIFO memory attention | [YAML](compressive-transformer/representation.yaml) | [코드](compressive-transformer/reconstructed.py) |
| [ConvS2S](convs2s/README.md) | 단일 encoder/decoder GLU 블록, attention과 residual, 명시적 학습 위치 임베딩 | [YAML](convs2s/representation.yaml) | [코드](convs2s/reconstructed.py) |
| [Decomposable Attention](decomposable-attention/README.md) | 공유 F/G, 양방향 정렬과 masked sum, 분류기 제외 | [YAML](decomposable-attention/representation.yaml) | [코드](decomposable-attention/reconstructed.py) |
| [E2E Memory Network](e2e-memory-network/README.md) | Memory와 query embedding, soft addressing, multi-hop read, tied 답변 투영 | [YAML](e2e-memory-network/representation.yaml) | [코드](e2e-memory-network/reconstructed.py) |
| [FlashAttention](flashattention/README.md) | CPU 타일 전방 계산과 online softmax 누적, causal mask | [YAML](flashattention/representation.yaml) | [코드](flashattention/reconstructed.py) |
| [FNet](fnet/README.md) | 단일 encoder block, 2축 Fourier mixing, post-norm residual과 FFN | [YAML](fnet/representation.yaml) | [코드](fnet/reconstructed.py) |
| [GNMT](gnmt/README.md) | 3층 encoder–decoder, 양방향 하단 encoder, residual, attention | [YAML](gnmt/representation.yaml) | [코드](gnmt/reconstructed.py) |
| [GQA](gqa/README.md) | 외부 QKV의 query-to-KV 그룹 대응, causal prefix와 G-head cache | [YAML](gqa/representation.yaml) | [코드](gqa/reconstructed.py) |
| [H3](h3/README.md) | scalar-head shift, diagonal SSM gated block | [YAML](h3/representation.yaml) | [코드](h3/reconstructed.py) |
| [HiPPO](hippo/README.md) | HiPPO-LegS 연산자, 시간 가변 bilinear 이산화, 계수 상태 scan | [YAML](hippo/representation.yaml) | [코드](hippo/reconstructed.py) |
| [Hyena](hyena/README.md) | 입력 gating, short filter와 implicit long filter, causal FFT convolution core | [YAML](hyena/representation.yaml) | [코드](hyena/reconstructed.py) |
| [Jamba](jamba/README.md) | 축소 8층 주기, GQA와 Mamba mixer, dense 및 top-2 SwiGLU expert, 상태 이어 계산 | [YAML](jamba/representation.yaml) | [코드](jamba/reconstructed.py) |
| [LayerNorm](layernorm/README.md) | 마지막 축의 population 통계, affine 정규화 | [YAML](layernorm/representation.yaml) | [코드](layernorm/reconstructed.py) |
| [Linear Transformer](linear-transformer/README.md) | ELU+1 recurrent linear-attention core; external q/k/v and in-place S/Z state | [YAML](linear-transformer/representation.yaml) | [코드](linear-transformer/reconstructed.py) |
| [Linformer](linformer/README.md) | sequence 축 low-rank projection attention core | [YAML](linformer/representation.yaml) | [코드](linformer/reconstructed.py) |
| [Longformer](longformer/README.md) | 여섯 투영을 포함한 양방향 window/global attention core | [YAML](longformer/representation.yaml) | [코드](longformer/reconstructed.py) |
| [LSSL](lssl/README.md) | 고정 HiPPO-LegS, bilinear discretization, dense recurrence | [YAML](lssl/representation.yaml) | [코드](lssl/reconstructed.py) |
| [LSTM](lstm/README.md) | Single-layer unidirectional core, T>=1; tested T=1,3,5, B=2,I=4,H=5 | [YAML](lstm/representation.yaml) | [코드](lstm/reconstructed.py) |
| [LSTMN](lstmn/README.md) | Hidden/cell tape, causal additive attention, read 기반 gate와 상태 갱신 | [YAML](lstmn/representation.yaml) | [코드](lstmn/reconstructed.py) |
| [Luong Attention](luong-attention/README.md) | global/general attention, LSTM, input feeding | [YAML](luong-attention/representation.yaml) | [코드](luong-attention/reconstructed.py) |
| [Mamba](mamba/README.md) | Mamba-1 selective-scan core, channels=3,state_size=2 | [YAML](mamba/representation.yaml) | [코드](mamba/reconstructed.py) |
| [Mamba-2 / SSD](mamba2-ssd/README.md) | scalar-identity multihead SSD chunk core | [YAML](mamba2-ssd/representation.yaml) | [코드](mamba2-ssd/reconstructed.py) |
| [Memory Network](memory-network/README.md) | 제공된 count feature, fact 저장, 두 argmax support, 답변 ranking | [YAML](memory-network/representation.yaml) | [코드](memory-network/reconstructed.py) |
| [Neural LM](neural-lm/README.md) | 고정 문맥 lookup, tanh hidden, direct 경로와 vocabulary readout | [YAML](neural-lm/representation.yaml) | [코드](neural-lm/reconstructed.py) |
| [Performer](performer/README.md) | 고정 random features의 정규화 attention core | [YAML](performer/representation.yaml) | [코드](performer/reconstructed.py) |
| [Pre-LN](pre-ln/README.md) | Pre-LN encoder block, attention과 FFN residual, key padding mask | [YAML](pre-ln/representation.yaml) | [코드](pre-ln/reconstructed.py) |
| [QRNN](qrnn/README.md) | 단일 causal gate 계층, fo-pooling, 외부 cell과 입력 이력 | [YAML](qrnn/representation.yaml) | [코드](qrnn/reconstructed.py) |
| [Reformer](reformer/README.md) | 제공된 회전의 LSH attention, chunk와 이전 chunk 참조, multi-round 결합 | [YAML](reformer/representation.yaml) | [코드](reformer/reconstructed.py) |
| [Relative Position](relative-position/README.md) | 부호 있는 거리 clipping과 상대 key/value를 포함한 projected self-attention core | [YAML](relative-position/representation.yaml) | [코드](relative-position/reconstructed.py) |
| [ResNet](resnet/README.md) | Identity shortcut BasicBlock, channels=64, CPU float32 eval | [YAML](resnet/representation.yaml) | [코드](resnet/reconstructed.py) |
| [RetNet](retnet/README.md) | Multi-scale causal retention, 병렬 합산과 순환 행렬 상태, 상태 이어 계산 | [YAML](retnet/representation.yaml) | [코드](retnet/reconstructed.py) |
| [RETRO](retro/README.md) | 제공된 검색 표현의 chunked cross-attention, joint softmax와 residual core | [YAML](retro/representation.yaml) | [코드](retro/reconstructed.py) |
| [RMSNorm](rmsnorm/README.md) | 마지막 축 RMS 정규화, 제곱근 밖 epsilon, scale-only | [YAML](rmsnorm/representation.yaml) | [코드](rmsnorm/reconstructed.py) |
| [RNN Encoder–Decoder / GRU](gru-encoder-decoder/README.md) | Cho 2014 encoder, 고정 context, 조건부 GRU decoder | [YAML](gru-encoder-decoder/representation.yaml) | [코드](gru-encoder-decoder/reconstructed.py) |
| [RNN LM](rnn-lm/README.md) | 2010 sigmoid recurrent core와 flat vocabulary softmax, 명시적 초기 상태 | [YAML](rnn-lm/representation.yaml) | [코드](rnn-lm/reconstructed.py) |
| [RoPE](rope/README.md) | 명시 위치의 인접 feature 쌍 회전과 attention core | [YAML](rope/representation.yaml) | [코드](rope/reconstructed.py) |
| [RWKV](rwkv/README.md) | RWKV v4 time-mixing core, 안정화 WKV, 명시적 상태 이어 계산 | [YAML](rwkv/representation.yaml) | [코드](rwkv/reconstructed.py) |
| [S4](s4/README.md) | Original rank-1 DPLR/NPLR recurrent/impulse-kernel core, Nfull=4,H=channels=1, CPU eval | [YAML](s4/representation.yaml) | [코드](s4/reconstructed.py) |
| [Seq2Seq](seq2seq/README.md) | Pre-embedded one-layer LSTM encoder/decoder, 100% teacher forcing | [YAML](seq2seq/representation.yaml) | [코드](seq2seq/reconstructed.py) |
| [Simple RNN](simple-rnn/README.md) | Elman tanh 순환 core | [YAML](simple-rnn/representation.yaml) | [코드](simple-rnn/reconstructed.py) |
| [Sparse MoE](sparse-moe/README.md) | 결정적 추론, top-k routing, gate 정규화, ReLU expert 결합 | [YAML](sparse-moe/representation.yaml) | [코드](sparse-moe/reconstructed.py) |
| [Sparse Transformer](sparse-transformer/README.md) | local/strided causal attention core | [YAML](sparse-transformer/representation.yaml) | [코드](sparse-transformer/reconstructed.py) |
| [Structured Self-Attention](structured-self-attention/README.md) | bias 없는 다중 hop attention과 행렬 sentence embedding, encoder 제외 | [YAML](structured-self-attention/representation.yaml) | [코드](structured-self-attention/reconstructed.py) |
| [Switch Transformer](switch-transformer/README.md) | Top-1 routing, 용량 제한과 overflow drop, padding mask, ReLU expert | [YAML](switch-transformer/representation.yaml) | [코드](switch-transformer/reconstructed.py) |
| [Synthesizer](synthesizer/README.md) | 토큰 조건부 logits, multihead value 집계, mask와 출력 투영 | [YAML](synthesizer/representation.yaml) | [코드](synthesizer/reconstructed.py) |
| [Transformer](transformer/README.md) | 단일 encoder block, 명시적 sinusoidal 위치 인코딩과 토큰 스케일링 | [YAML](transformer/representation.yaml) | [코드](transformer/reconstructed.py) |
| [Transformer-XL](transformer-xl/README.md) | segment recurrence, 상대 위치 attention sublayer | [YAML](transformer-xl/representation.yaml) | [코드](transformer-xl/reconstructed.py) |
| [TTT](ttt/README.md) | 단일 head TTT-Linear 내부 학습기, chunk 내 causal 갱신과 경계 상태 이어 계산 | [YAML](ttt/representation.yaml) | [코드](ttt/reconstructed.py) |
| [WaveNet](wavenet/README.md) | 조건 없는 dilated gated residual/skip feature stack, audio 입출력 제외 | [YAML](wavenet/representation.yaml) | [코드](wavenet/reconstructed.py) |
