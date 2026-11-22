# ByteNet

ReLU bottleneck residual feature stack, causal 또는 centered convolution.

[계산 YAML](representation.yaml), [최종 구현](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json)

## 사용 계약

```yaml
constructor: 'ByteNetStack(channels: int, bottleneck: int, dilations: tuple[int,...],
  causal: bool); torch.nn.Module'
forward: forward(x) -> dict[str, Tensor]
inputs:
  x: CPU float64 [B,C,T], axes batch/channel/time; supplied internal features with
    all positions valid
constraints: 1<=B<=2; 1<=D=bottleneck<=4; C=channels=2*D; 1<=T<=16; 1<=L=len(dilations)<=4;
  integer dilations 1..8; finite inputs/parameters abs<=2. causal is bool. Empty sequences/invalid-input
  diagnostics excluded.
returns:
  states: float64 [L+1,B,C,T], x then all block outputs
  normalized1: float64 [L,B,C,T]
  reduced: float64 [L,B,D,T]
  normalized2: float64 [L,B,D,T]
  dilated: float64 [L,B,D,T]
  normalized3: float64 [L,B,D,T]
  expanded: float64 [L,B,C,T]
  output: float64 [B,C,T], states[L]
```

## 구현 범위

```yaml
variant: ReLU bottleneck residual feature stack; causal decoder-style or centered
  encoder-style selected at construction
included:
- per_position_layernorm
- relu_preactivation
- pointwise_bottleneck
- dilated_convolution
- residual_expansion
excluded:
- multiplicative_unit_LM_variant
- embedding
- encoder_decoder_connection
- dynamic_unfolding
- initial_token_shift
- output_head
- softmax
- decoding
- training
- gradients
- streaming_cache
```

## 관련 관계

관계 유형과 필드 설명은 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [WaveNet → ByteNet](../../relations/tr_015.yaml) — I, tr_015
- [ByteNet → ConvS2S](../../relations/tr_016.yaml) — X?, tr_016
