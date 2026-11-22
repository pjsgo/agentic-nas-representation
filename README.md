# Agentic NAS: Architecture Reconstruction

**신경망 아키텍처의 핵심 계산을 YAML로 기술하고 PyTorch 코드로 구현했습니다.** YAML에는 연산과 연결, 가중치 공유 방식, 입력과 상태가 흐르는 순서가 담겨 있습니다. 계보 자료에서는 두 아키텍처가 어떤 계산을 공유하고 어디에서 달라지는지 볼 수 있습니다.

## 폴더 구조

```text
architectures/  아키텍처별 YAML, 구현 코드와 사용법
relations/      아키텍처 사이의 관계와 근거
```

[아키텍처 YAML 필드](architectures/README.md)와 [관계 YAML 필드 및 계보도](relations/README.md)를 참고하세요.

## 아키텍처 자료

53개 아키텍처를 이름순으로 정리했습니다. 모델 이름을 누르면 입력과 사용법을, YAML과 코드 링크를 누르면 해당 파일을 볼 수 있습니다.

| 아키텍처 | 대상 범위 | 복원 YAML | 구현 |
|---|---|---|---|
| [ALiBi](architectures/alibi/README.md) | 고정 head slope와 causal 거리 bias attention core | [YAML](architectures/alibi/representation.yaml) | [코드](architectures/alibi/reconstructed.py) |
| [Bahdanau Attention](architectures/bahdanau-attention/README.md) | 제공된 encoder memory에 대한 additive attention, 조건부 GRU | [YAML](architectures/bahdanau-attention/representation.yaml) | [코드](architectures/bahdanau-attention/reconstructed.py) |
| [BigBird](architectures/bigbird/README.md) | 외부 random graph를 받는 local/global/random block attention core | [YAML](architectures/bigbird/representation.yaml) | [코드](architectures/bigbird/reconstructed.py) |
| [ByteNet](architectures/bytenet/README.md) | ReLU bottleneck residual feature stack, causal 또는 centered convolution | [YAML](architectures/bytenet/representation.yaml) | [코드](architectures/bytenet/reconstructed.py) |
| [Compressive Transformer](architectures/compressive-transformer/README.md) | 평균 풀링 압축, 두 단계 FIFO memory attention | [YAML](architectures/compressive-transformer/representation.yaml) | [코드](architectures/compressive-transformer/reconstructed.py) |
| [ConvS2S](architectures/convs2s/README.md) | 단일 encoder/decoder GLU 블록, attention과 residual, 명시적 학습 위치 임베딩 | [YAML](architectures/convs2s/representation.yaml) | [코드](architectures/convs2s/reconstructed.py) |
| [Decomposable Attention](architectures/decomposable-attention/README.md) | 공유 F/G, 양방향 정렬과 masked sum, 분류기 제외 | [YAML](architectures/decomposable-attention/representation.yaml) | [코드](architectures/decomposable-attention/reconstructed.py) |
| [E2E Memory Network](architectures/e2e-memory-network/README.md) | Memory와 query embedding, soft addressing, multi-hop read, tied 답변 투영 | [YAML](architectures/e2e-memory-network/representation.yaml) | [코드](architectures/e2e-memory-network/reconstructed.py) |
| [FlashAttention](architectures/flashattention/README.md) | CPU 타일 전방 계산과 online softmax 누적, causal mask | [YAML](architectures/flashattention/representation.yaml) | [코드](architectures/flashattention/reconstructed.py) |
| [FNet](architectures/fnet/README.md) | 단일 encoder block, 2축 Fourier mixing, post-norm residual과 FFN | [YAML](architectures/fnet/representation.yaml) | [코드](architectures/fnet/reconstructed.py) |
| [GNMT](architectures/gnmt/README.md) | 3층 encoder–decoder, 양방향 하단 encoder, residual, attention | [YAML](architectures/gnmt/representation.yaml) | [코드](architectures/gnmt/reconstructed.py) |
| [GQA](architectures/gqa/README.md) | 외부 QKV의 query-to-KV 그룹 대응, causal prefix와 G-head cache | [YAML](architectures/gqa/representation.yaml) | [코드](architectures/gqa/reconstructed.py) |
| [H3](architectures/h3/README.md) | scalar-head shift, diagonal SSM gated block | [YAML](architectures/h3/representation.yaml) | [코드](architectures/h3/reconstructed.py) |
| [HiPPO](architectures/hippo/README.md) | HiPPO-LegS 연산자, 시간 가변 bilinear 이산화, 계수 상태 scan | [YAML](architectures/hippo/representation.yaml) | [코드](architectures/hippo/reconstructed.py) |
| [Hyena](architectures/hyena/README.md) | 입력 gating, short filter와 implicit long filter, causal FFT convolution core | [YAML](architectures/hyena/representation.yaml) | [코드](architectures/hyena/reconstructed.py) |
| [Jamba](architectures/jamba/README.md) | 축소 8층 주기, GQA와 Mamba mixer, dense 및 top-2 SwiGLU expert, 상태 이어 계산 | [YAML](architectures/jamba/representation.yaml) | [코드](architectures/jamba/reconstructed.py) |
| [LayerNorm](architectures/layernorm/README.md) | 마지막 축의 population 통계, affine 정규화 | [YAML](architectures/layernorm/representation.yaml) | [코드](architectures/layernorm/reconstructed.py) |
| [Linear Transformer](architectures/linear-transformer/README.md) | ELU+1 recurrent linear-attention core; external q/k/v and in-place S/Z state | [YAML](architectures/linear-transformer/representation.yaml) | [코드](architectures/linear-transformer/reconstructed.py) |
| [Linformer](architectures/linformer/README.md) | sequence 축 low-rank projection attention core | [YAML](architectures/linformer/representation.yaml) | [코드](architectures/linformer/reconstructed.py) |
| [Longformer](architectures/longformer/README.md) | 여섯 투영을 포함한 양방향 window/global attention core | [YAML](architectures/longformer/representation.yaml) | [코드](architectures/longformer/reconstructed.py) |
| [LSSL](architectures/lssl/README.md) | 고정 HiPPO-LegS, bilinear discretization, dense recurrence | [YAML](architectures/lssl/representation.yaml) | [코드](architectures/lssl/reconstructed.py) |
| [LSTM](architectures/lstm/README.md) | Single-layer unidirectional core, T>=1; tested T=1,3,5, B=2,I=4,H=5 | [YAML](architectures/lstm/representation.yaml) | [코드](architectures/lstm/reconstructed.py) |
| [LSTMN](architectures/lstmn/README.md) | Hidden/cell tape, causal additive attention, read 기반 gate와 상태 갱신 | [YAML](architectures/lstmn/representation.yaml) | [코드](architectures/lstmn/reconstructed.py) |
| [Luong Attention](architectures/luong-attention/README.md) | global/general attention, LSTM, input feeding | [YAML](architectures/luong-attention/representation.yaml) | [코드](architectures/luong-attention/reconstructed.py) |
| [Mamba](architectures/mamba/README.md) | Mamba-1 selective-scan core, channels=3,state_size=2 | [YAML](architectures/mamba/representation.yaml) | [코드](architectures/mamba/reconstructed.py) |
| [Mamba-2 / SSD](architectures/mamba2-ssd/README.md) | scalar-identity multihead SSD chunk core | [YAML](architectures/mamba2-ssd/representation.yaml) | [코드](architectures/mamba2-ssd/reconstructed.py) |
| [Memory Network](architectures/memory-network/README.md) | 제공된 count feature, fact 저장, 두 argmax support, 답변 ranking | [YAML](architectures/memory-network/representation.yaml) | [코드](architectures/memory-network/reconstructed.py) |
| [Neural LM](architectures/neural-lm/README.md) | 고정 문맥 lookup, tanh hidden, direct 경로와 vocabulary readout | [YAML](architectures/neural-lm/representation.yaml) | [코드](architectures/neural-lm/reconstructed.py) |
| [Performer](architectures/performer/README.md) | 고정 random features의 정규화 attention core | [YAML](architectures/performer/representation.yaml) | [코드](architectures/performer/reconstructed.py) |
| [Pre-LN](architectures/pre-ln/README.md) | Pre-LN encoder block, attention과 FFN residual, key padding mask | [YAML](architectures/pre-ln/representation.yaml) | [코드](architectures/pre-ln/reconstructed.py) |
| [QRNN](architectures/qrnn/README.md) | 단일 causal gate 계층, fo-pooling, 외부 cell과 입력 이력 | [YAML](architectures/qrnn/representation.yaml) | [코드](architectures/qrnn/reconstructed.py) |
| [Reformer](architectures/reformer/README.md) | 제공된 회전의 LSH attention, chunk와 이전 chunk 참조, multi-round 결합 | [YAML](architectures/reformer/representation.yaml) | [코드](architectures/reformer/reconstructed.py) |
| [Relative Position](architectures/relative-position/README.md) | 부호 있는 거리 clipping과 상대 key/value를 포함한 projected self-attention core | [YAML](architectures/relative-position/representation.yaml) | [코드](architectures/relative-position/reconstructed.py) |
| [ResNet](architectures/resnet/README.md) | Identity shortcut BasicBlock, channels=64, CPU float32 eval | [YAML](architectures/resnet/representation.yaml) | [코드](architectures/resnet/reconstructed.py) |
| [RetNet](architectures/retnet/README.md) | Multi-scale causal retention, 병렬 합산과 순환 행렬 상태, 상태 이어 계산 | [YAML](architectures/retnet/representation.yaml) | [코드](architectures/retnet/reconstructed.py) |
| [RETRO](architectures/retro/README.md) | 제공된 검색 표현의 chunked cross-attention, joint softmax와 residual core | [YAML](architectures/retro/representation.yaml) | [코드](architectures/retro/reconstructed.py) |
| [RMSNorm](architectures/rmsnorm/README.md) | 마지막 축 RMS 정규화, 제곱근 밖 epsilon, scale-only | [YAML](architectures/rmsnorm/representation.yaml) | [코드](architectures/rmsnorm/reconstructed.py) |
| [RNN Encoder–Decoder / GRU](architectures/gru-encoder-decoder/README.md) | Cho 2014 encoder, 고정 context, 조건부 GRU decoder | [YAML](architectures/gru-encoder-decoder/representation.yaml) | [코드](architectures/gru-encoder-decoder/reconstructed.py) |
| [RNN LM](architectures/rnn-lm/README.md) | 2010 sigmoid recurrent core와 flat vocabulary softmax, 명시적 초기 상태 | [YAML](architectures/rnn-lm/representation.yaml) | [코드](architectures/rnn-lm/reconstructed.py) |
| [RoPE](architectures/rope/README.md) | 명시 위치의 인접 feature 쌍 회전과 attention core | [YAML](architectures/rope/representation.yaml) | [코드](architectures/rope/reconstructed.py) |
| [RWKV](architectures/rwkv/README.md) | RWKV v4 time-mixing core, 안정화 WKV, 명시적 상태 이어 계산 | [YAML](architectures/rwkv/representation.yaml) | [코드](architectures/rwkv/reconstructed.py) |
| [S4](architectures/s4/README.md) | Original rank-1 DPLR/NPLR recurrent/impulse-kernel core, Nfull=4,H=channels=1, CPU eval | [YAML](architectures/s4/representation.yaml) | [코드](architectures/s4/reconstructed.py) |
| [Seq2Seq](architectures/seq2seq/README.md) | Pre-embedded one-layer LSTM encoder/decoder, 100% teacher forcing | [YAML](architectures/seq2seq/representation.yaml) | [코드](architectures/seq2seq/reconstructed.py) |
| [Simple RNN](architectures/simple-rnn/README.md) | Elman tanh 순환 core | [YAML](architectures/simple-rnn/representation.yaml) | [코드](architectures/simple-rnn/reconstructed.py) |
| [Sparse MoE](architectures/sparse-moe/README.md) | 결정적 추론, top-k routing, gate 정규화, ReLU expert 결합 | [YAML](architectures/sparse-moe/representation.yaml) | [코드](architectures/sparse-moe/reconstructed.py) |
| [Sparse Transformer](architectures/sparse-transformer/README.md) | local/strided causal attention core | [YAML](architectures/sparse-transformer/representation.yaml) | [코드](architectures/sparse-transformer/reconstructed.py) |
| [Structured Self-Attention](architectures/structured-self-attention/README.md) | bias 없는 다중 hop attention과 행렬 sentence embedding, encoder 제외 | [YAML](architectures/structured-self-attention/representation.yaml) | [코드](architectures/structured-self-attention/reconstructed.py) |
| [Switch Transformer](architectures/switch-transformer/README.md) | Top-1 routing, 용량 제한과 overflow drop, padding mask, ReLU expert | [YAML](architectures/switch-transformer/representation.yaml) | [코드](architectures/switch-transformer/reconstructed.py) |
| [Synthesizer](architectures/synthesizer/README.md) | 토큰 조건부 logits, multihead value 집계, mask와 출력 투영 | [YAML](architectures/synthesizer/representation.yaml) | [코드](architectures/synthesizer/reconstructed.py) |
| [Transformer](architectures/transformer/README.md) | 단일 encoder block, 명시적 sinusoidal 위치 인코딩과 토큰 스케일링 | [YAML](architectures/transformer/representation.yaml) | [코드](architectures/transformer/reconstructed.py) |
| [Transformer-XL](architectures/transformer-xl/README.md) | segment recurrence, 상대 위치 attention sublayer | [YAML](architectures/transformer-xl/representation.yaml) | [코드](architectures/transformer-xl/reconstructed.py) |
| [TTT](architectures/ttt/README.md) | 단일 head TTT-Linear 내부 학습기, chunk 내 causal 갱신과 경계 상태 이어 계산 | [YAML](architectures/ttt/representation.yaml) | [코드](architectures/ttt/reconstructed.py) |
| [WaveNet](architectures/wavenet/README.md) | 조건 없는 dilated gated residual/skip feature stack, audio 입출력 제외 | [YAML](architectures/wavenet/representation.yaml) | [코드](architectures/wavenet/reconstructed.py) |
