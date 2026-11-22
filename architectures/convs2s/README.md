# ConvS2S

단일 encoder/decoder GLU 블록, attention과 residual, 명시적 학습 위치 임베딩.

[계산 YAML](representation.yaml), [최종 구현](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json)

## 사용 계약

기본 통합 진입점은 `model.forward_with_positions(source_words, source_positions, source_table, target_words, target_positions, target_table, source_padding_mask)`입니다. CPU float64 token 벡터 `[B,S,D]`/`[B,T,D]`, int64 위치 ID `[B,S]`/`[B,T]`, 독립 learned table `[Ps,D]`/`[Pt,D]`을 받습니다. ID는 각 table 범위 내이며 입력은 유한해야 합니다. 위치 조회와 합산 후 기존 블록으로 연결합니다. target은 caller가 미리 shift합니다. 자동 위치 번호나 padding 합산 제거는 하지 않습니다.

기존 사전 합산 입력 호출 `model(source_embedding, target_embedding, source_padding_mask)`도 사용할 수 있습니다. 아래는 이 블록 호출의 계약입니다.

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

## 구현 범위

```yaml
included:
- one_encoder_GLU_block
- one_causal_decoder_GLU_block
- input_and_encoder_output_projections
- single_decoder_attention
- source_padding_mask
- residual_scaling
- explicit_position_table_lookup
- token_position_addition
excluded:
- token_lookup
- automatic_position_numbering
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

위치 계산은 같은 YAML의 `source_position_lookup/add`, `target_position_lookup/add`와 같은 구현 파일에 정의됩니다.

## 관련 관계

관계 유형과 필드 설명은 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [ByteNet → ConvS2S](../../relations/tr_016.yaml) — X?, tr_016
- [Bahdanau Attention → ConvS2S](../../relations/tr_017.yaml) — C, tr_017
- [ConvS2S → Transformer](../../relations/tr_025.yaml) — X, tr_025
- [ConvS2S → Transformer](../../relations/tr_072.yaml) — C, tr_072
