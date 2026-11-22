# WaveNet

조건 없는 dilated gated residual/skip feature stack, audio 입출력 제외.

[계산 YAML](representation.yaml), [최종 구현](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json)

## 사용 계약

```yaml
constructor: 'WaveNetStack(channels: int, hidden_channels: int, skip_channels: int,
  dilations: tuple[int,...]); torch.nn.Module'
forward: forward(x) -> dict[str, Tensor]
inputs:
  x: CPU float64 [B,C,T], axes batch/channel/time; supplied hidden features, not raw
    audio
constraints: 1<=B<=2; 1<=C,H,S<=8; 1<=T<=16; 1<=L=len(dilations)<=4; each dilation
  integer 1..8; finite inputs and parameter magnitudes <=2; all positions valid. No
  implicit sequence shift. Empty sequences and invalid-input diagnostics excluded.
symbols: C=channels, H=hidden_channels, S=skip_channels, L=len(dilations)
returns:
  states: float64 [L+1,B,C,T], x followed by each residual state
  filter_pre: float64 [L,B,H,T]
  gate_pre: float64 [L,B,H,T]
  gated: float64 [L,B,H,T]
  skips: float64 [L,B,S,T]
  skip_sum: float64 [B,S,T]
  output: float64 [B,C,T], states[L]
```

## 구현 범위

```yaml
variant: Unconditioned bias-free dilated gated residual and skip feature stack
included:
- causal_dilation
- tanh_sigmoid_gate
- residual_projection
- skip_projection
- skip_sum
excluded:
- initial_audio_convolution
- mu_law
- quantization
- output_head
- softmax
- conditioning
- audio_sampling
- streaming_cache
- training
- gradients
```

## 관련 관계

관계 유형은 [관계 안내](../../relations/README.md)를 참고합니다. `?`는 추론 관계입니다.

- [WaveNet → ByteNet](../../relations/tr_015.yaml) — I, tr_015
- [WaveNet → Hyena](../../relations/tr_052.yaml) — I?, tr_052
