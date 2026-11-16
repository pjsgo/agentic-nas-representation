# Agentic NAS: Architecture Reconstruction

**YAML에 명시된 계산을 LLM이 코드로 복원하도록 요청합니다. 생성 코드의 핵심 계산과 입출력, 파라미터, 상태 계약은 독립 참조와 비교해 확인합니다.**

아키텍처 이름이나 익숙한 구조로 빠진 계산을 보충하지 않도록 지시합니다. 생성 입력은 YAML과 그 해석 규칙, 구현 계약이며, 원본 아키텍처 코드와 평가 정답은 제공하지 않습니다. 생성 원본과 수정본, 복원 범위와 미해결 부분은 구분해 기록합니다. 아키텍처 사이의 계승과 변경 관계는 별도 관계 YAML로 연결합니다.

## 폴더 구조

```text
architectures/  각 아키텍처의 복원 YAML, 코드, 사용 계약, 출처
relations/      아키텍처 사이의 변경과 보존, 부품 대응
specs/          YAML과 관계를 해석하는 규칙과 버전
docs/           읽는 순서, 예시, 자료 관리 안내
```

공유 저장소에서는 이 프로젝트가 개인별 폴더 안에, 개인 저장소에서는 루트에 놓입니다. 내부 경로와 내용은 동일합니다.

## 자료 읽기

- [아키텍처 복원 방법](docs/reconstruction-method.md): 제공한 정보, 확인 방법, 결과의 의미
- [복원 YAML 안내](docs/representation-guide.md): 모델 내부의 연산과 연결, 파라미터, 입출력과 상태
- [관계 YAML 안내](docs/genealogy-guide.md): 모델 사이의 변경과 보존, 부품 대응과 근거
- [해석 명세](specs/README.md): 표현, 연산, 관계의 규칙과 버전
- [관계 목록](relations/README.md): 출발 및 도착 아키텍처별 관계 YAML

복원 범위는 전체 모델 또는 명시된 block/core입니다. 코드의 생성 및 수정 출처와 입력 계약은 각 아키텍처 자료를 따릅니다. 설명용 관계를 실행 가능한 변환으로 해석하지 않습니다.

## 아키텍처 자료

| 아키텍처 | 대상 범위 | 복원 YAML | 구현 |
|---|---|---|---|
| [ResNet](architectures/resnet/README.md) | Identity shortcut BasicBlock, channels=64, CPU float32 eval | [YAML](architectures/resnet/representation.yaml) | [코드](architectures/resnet/reconstructed.py) |
| [LSTM](architectures/lstm/README.md) | Single-layer unidirectional core, T>=1; tested T=1,3,5, B=2,I=4,H=5 | [YAML](architectures/lstm/representation.yaml) | [코드](architectures/lstm/reconstructed.py) |
| [Seq2Seq](architectures/seq2seq/README.md) | Pre-embedded one-layer LSTM encoder/decoder, 100% teacher forcing, three CPU float32 fixtures | [YAML](architectures/seq2seq/representation.yaml) | [코드](architectures/seq2seq/reconstructed.py) |
| [Transformer](architectures/transformer/README.md) | Single encoder block, nine CPU float32 eval fixtures | [YAML](architectures/transformer/representation.yaml) | [코드](architectures/transformer/reconstructed.py) |
| [Linear Transformer](architectures/linear-transformer/README.md) | ELU+1 recurrent linear-attention core; external q/k/v and in-place S/Z state, four CPU float32 fixtures | [YAML](architectures/linear-transformer/representation.yaml) | [코드](architectures/linear-transformer/reconstructed.py) |
| [S4](architectures/s4/README.md) | Original rank-1 DPLR/NPLR recurrent/impulse-kernel core, Nfull=4,H=channels=1, CPU eval | [YAML](architectures/s4/representation.yaml) | [코드](architectures/s4/reconstructed.py) |
| [Mamba](architectures/mamba/README.md) | Mamba-1 selective-scan core, channels=3,state_size=2, four CPU float32 fixtures | [YAML](architectures/mamba/representation.yaml) | [코드](architectures/mamba/reconstructed.py) |
| [Simple RNN](architectures/simple-rnn/README.md) | Elman tanh 순환 core | [YAML](architectures/simple-rnn/representation.yaml) | [코드](architectures/simple-rnn/reconstructed.py) |
| [RNN Encoder–Decoder / GRU](architectures/gru-encoder-decoder/README.md) | Cho 2014 encoder, 고정 context, 조건부 GRU decoder | [YAML](architectures/gru-encoder-decoder/representation.yaml) | [코드](architectures/gru-encoder-decoder/reconstructed.py) |
| [Bahdanau Attention](architectures/bahdanau-attention/README.md) | 제공된 encoder memory에 대한 additive attention, 조건부 GRU | [YAML](architectures/bahdanau-attention/representation.yaml) | [코드](architectures/bahdanau-attention/reconstructed.py) |
| [Luong Attention](architectures/luong-attention/README.md) | global/general attention, LSTM, input feeding | [YAML](architectures/luong-attention/representation.yaml) | [코드](architectures/luong-attention/reconstructed.py) |
| [GNMT](architectures/gnmt/README.md) | 3층 encoder–decoder, 양방향 하단 encoder, residual, attention | [YAML](architectures/gnmt/representation.yaml) | [코드](architectures/gnmt/reconstructed.py) |
| [LayerNorm](architectures/layernorm/README.md) | 마지막 축의 population 통계, affine 정규화 | [YAML](architectures/layernorm/representation.yaml) | [코드](architectures/layernorm/reconstructed.py) |
| [Transformer-XL](architectures/transformer-xl/README.md) | segment recurrence, 상대 위치 attention sublayer | [YAML](architectures/transformer-xl/representation.yaml) | [코드](architectures/transformer-xl/reconstructed.py) |
| [Compressive Transformer](architectures/compressive-transformer/README.md) | 평균 풀링 압축, 두 단계 FIFO memory attention | [YAML](architectures/compressive-transformer/representation.yaml) | [코드](architectures/compressive-transformer/reconstructed.py) |
| [Sparse Transformer](architectures/sparse-transformer/README.md) | local/strided causal attention core | [YAML](architectures/sparse-transformer/representation.yaml) | [코드](architectures/sparse-transformer/reconstructed.py) |
| [Linformer](architectures/linformer/README.md) | sequence 축 low-rank projection attention core | [YAML](architectures/linformer/representation.yaml) | [코드](architectures/linformer/reconstructed.py) |
| [Performer](architectures/performer/README.md) | 고정 random features의 정규화 attention core | [YAML](architectures/performer/representation.yaml) | [코드](architectures/performer/reconstructed.py) |
| [LSSL](architectures/lssl/README.md) | 고정 HiPPO-LegS, bilinear discretization, dense recurrence | [YAML](architectures/lssl/representation.yaml) | [코드](architectures/lssl/reconstructed.py) |
| [Mamba-2 / SSD](architectures/mamba2-ssd/README.md) | scalar-identity multihead SSD chunk core | [YAML](architectures/mamba2-ssd/representation.yaml) | [코드](architectures/mamba2-ssd/reconstructed.py) |
| [H3](architectures/h3/README.md) | scalar-head shift, diagonal SSM gated block | [YAML](architectures/h3/representation.yaml) | [코드](architectures/h3/reconstructed.py) |
| [Longformer](architectures/longformer/README.md) | 여섯 투영을 포함한 양방향 window/global attention core | [YAML](architectures/longformer/representation.yaml) | [코드](architectures/longformer/reconstructed.py) |
| [BigBird](architectures/bigbird/README.md) | 외부 random graph를 받는 local/global/random block attention core | [YAML](architectures/bigbird/representation.yaml) | [코드](architectures/bigbird/reconstructed.py) |
| [FlashAttention](architectures/flashattention/README.md) | CPU 타일 전방 계산과 online softmax 누적, causal mask | [YAML](architectures/flashattention/representation.yaml) | [코드](architectures/flashattention/reconstructed.py) |
| [GQA](architectures/gqa/README.md) | 외부 QKV의 query-to-KV 그룹 대응, causal prefix와 G-head cache | [YAML](architectures/gqa/representation.yaml) | [코드](architectures/gqa/reconstructed.py) |

<!-- phase3 -->

3차 위치 표현 3개는 연구용 실현과 외부 LLM 생성 코드의 독립 참조 검사를 통과했습니다. 생성 원본 통과 3개, 수정 후 통과 0개이며 세부 이력은 각 provenance에 기록했습니다.

| 노드 | 복원 범위 | 자료 |
|---|---|---|
| Relative Position | 부호 있는 거리 clipping과 상대 key/value를 포함한 projected self-attention core | [자료](architectures/relative-position/README.md) |
| RoPE | 명시 위치의 인접 feature 쌍 회전과 attention core | [자료](architectures/rope/README.md) |
| ALiBi | 고정 head slope와 causal 거리 bias attention core | [자료](architectures/alibi/README.md) |

- [rmsnorm](architectures/rmsnorm/README.md): LLM 생성 원본을 계산 수정 없이 독립 참조와 비교하여 통과했습니다.

- [pre-ln](architectures/pre-ln/README.md): LLM 생성 원본을 계산 수정 없이 독립 참조와 비교하여 통과했습니다.

- [sparse-moe](architectures/sparse-moe/README.md): 생성 원본은 실패했으며 별도 수정본이 독립 참조 검사를 통과했습니다.

- [switch-transformer](architectures/switch-transformer/README.md): 생성 원본은 실패했으며 별도 수정본이 독립 참조 검사를 통과했습니다.

- [reformer](architectures/reformer/README.md): 생성 원본은 실패했으며 별도 수정본이 독립 참조 검사를 통과했습니다.

- [synthesizer](architectures/synthesizer/README.md): LLM 생성 원본을 계산 수정 없이 독립 참조와 비교하여 통과했습니다.

- [fnet](architectures/fnet/README.md): LLM 생성 원본을 계산 수정 없이 독립 참조와 비교하여 통과했습니다.

<!-- phase5-completion -->

## 5차 통합

45개 아키텍처·64개 관계. 신규 10개 배포 코드의 독립 참조 대표 사례 30개 통과. 생성 원문 통과 2개·수정본 통과 8개.

[검증과 한계](docs/phase5-readiness.md) · [전체 관계도](docs/genealogy-map.md)

- [hippo](architectures/hippo/README.md)
- [hyena](architectures/hyena/README.md)
- [retnet](architectures/retnet/README.md)
- [rwkv](architectures/rwkv/README.md)
- [memory-network](architectures/memory-network/README.md)
- [e2e-memory-network](architectures/e2e-memory-network/README.md)
- [lstmn](architectures/lstmn/README.md)
- [retro](architectures/retro/README.md)
- [jamba](architectures/jamba/README.md)
- [ttt](architectures/ttt/README.md)
