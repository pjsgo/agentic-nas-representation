# ConvS2S

단일 encoder/decoder GLU 블록, attention과 residual, 사전 합산 입력 임베딩.

[계산 YAML](representation.yaml), [최종 구현](reconstructed.py), [출처](sources.yaml), [검증 및 생성 이력](provenance.json)

## 사용 계약

```yaml
constructor: Model(D=3,C=4,K=3), torch.nn.Module
forward: forward(source_embedding, target_embedding, source_padding_mask) -> dict
dtype: torch.float64
device: cpu
inputs:
  source_embedding: '[B,S,D]; finite sum of token and absolute position embeddings
    supplied by caller'
  target_embedding: '[B,T,D]; finite embeddings already shifted by caller; index t
    contains information for predicting next token'
  source_padding_mask: '[B,S] bool; True excludes position; at least one False per
    row; positive B,S,T'
outputs:
  encoder: '[B,S,D], projected encoder states, zero at masked positions'
  values: '[B,S,D], (encoder+source_embedding)/sqrt(2); masked values may be nonzero
    but never attended'
  decoder_glu: '[B,T,C]'
  attention: '[B,T,S]'
  context: '[B,T,D], weighted values times sqrt(valid source count)'
  output: '[B,T,C], block output before vocabulary projection'
guards: No extra runtime guards required; caller satisfies shape/domain; all-masked
  source and target padding outside scope
```

## 복원 범위와 평가

```yaml
included:
- one_encoder_GLU_block
- one_causal_decoder_GLU_block
- input_and_encoder_output_projections
- single_decoder_attention
- source_padding_mask
- residual_scaling
excluded:
- token_and_position_lookup
- stacked_blocks
- vocabulary_projection
- softmax_readout
- incremental_cache
- beam_search
- training
- dropout
- weight_norm_parameterization
- backward_gradient_scaling
```

생성 원본: FAIL. 최종 코드: PASS_WITHIN_DECLARED_SCOPE. float32 valid-count 제곱근 계산을 float64 입력 dtype에 맞춰 수정. 기존 독립 참조 대표 사례 3개 결과를 재사용했습니다. 전체 모델과 공식 runtime 동등성은 검증하지 않았습니다.

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [ByteNet → ConvS2S](../../relations/tr_016.yaml) — X?, tr_016
- [Bahdanau Attention → ConvS2S](../../relations/tr_017.yaml) — C, tr_017
- [ConvS2S → Transformer](../../relations/tr_025.yaml) — X, tr_025
- [ConvS2S → Transformer](../../relations/tr_072.yaml) — C, tr_072
