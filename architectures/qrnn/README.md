# QRNN

단일 causal gate 계층, fo-pooling, 외부 cell과 입력 이력.

[계산 YAML](representation.yaml), [최종 구현](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json)

## 사용 계약

```yaml
constructor: Model(I=3,H=4,window=2), torch.nn.Module
forward: forward(x, c0=None, prev_x=None) -> dict
dtype: torch.float64
device: cpu
domain: finite inputs; positive B,T; all positions are real tokens; no padding mask
inputs:
  x: '[B,T,I]'
  c0: None or [B,H]; None means zeros
  prev_x: None or [B,I]; None means zeros; must be None for window=1
outputs:
  z: '[B,T,H]'
  update_gate: '[B,T,H]'
  output_gate: '[B,T,H]'
  cells: '[B,T,H]'
  output: '[B,T,H]'
  final_cell: '[B,H], cells[:,-1,:]'
  final_output: '[B,H], output[:,-1,:]'
  final_x: '[B,I], x[:,-1,:]'
guards: No additional runtime guards required; caller must satisfy domain and shapes
```

## 구현 범위

```yaml
included:
- single_causal_gate_layer
- fo_pooling_source_gate_convention
- external_cell_and_input_history
excluded:
- f_pooling
- ifo_pooling
- stacked_layers
- embeddings
- readout
- zoneout
- dropout
- training
- CUDA_kernel
- padding_mask
```

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [QRNN → Transformer](../../relations/tr_027.yaml) — X?, tr_027
