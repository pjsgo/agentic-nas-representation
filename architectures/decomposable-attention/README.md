# Decomposable Attention

공유 F/G, 양방향 정렬과 masked sum, 분류기 제외.

[계산 YAML](representation.yaml), [최종 구현](reconstructed.py), [출처](sources.yaml), [검증 및 생성 이력](provenance.json)

## 사용 계약

```yaml
signature: decompose(a, b, mask_a, mask_b, params) -> dict
class_contract: No class or constructor; export this pure function in reconstructed.py
inputs:
  a: CPU float64[B,M,D], finite supplied embedded/projected sequence
  b: CPU float64[B,N,D], finite supplied embedded/projected sequence
  mask_a: CPU bool[B,M]; True is a valid token
  mask_b: CPU bool[B,N]; True is a valid token
  params: dict with exactly fw1[F,D], fb1[F], fw2[F,F], fb2[F], gw1[G,2*D], gb1[G],
    gw2[G,G], gb2[G]; finite CPU float64 tensors
constraints:
- B,M,N,D,F,G positive; at least one valid token in each sequence of every batch item
- All inputs and computed intermediates finite, except internal masked logits may
  be -inf
- Arbitrary boolean padding patterns supported; empty and all-masked sequences raise
  ValueError
outputs:
  scores: float64[B,M,N], unscaled dot products, zero at invalid token pairs
  attention_ab: float64[B,M,N], normalized over valid b tokens for valid a queries;
    invalid rows and columns zero
  attention_ba: float64[B,N,M], normalized over valid a tokens for valid b queries;
    invalid rows and columns zero
  aligned_b: float64[B,M,D], b aligned to a; invalid a rows zero
  aligned_a: float64[B,N,D], a aligned to b; invalid b rows zero
  compare_a: float64[B,M,G], invalid rows zero
  compare_b: float64[B,N,G], invalid rows zero
  aggregate_a: float64[B,G]
  aggregate_b: float64[B,G]
  output: float64[B,2*G], concatenation aggregate_a then aggregate_b
```

## 복원 범위와 평가

```yaml
included:
- two_layer_shared_F
- unscaled_cross_scores
- bidirectional_alignment
- padding_masks
- two_layer_shared_G
- masked_sum_aggregation
excluded:
- intra_sentence_attention
- classifier_H
- label_probabilities
- NULL_token_creation
- embeddings
- input_projection
- training_dropout
- loss
- gradients
- full_NLI_model
```

생성 원본: FAIL. 최종 코드: PASS_WITHIN_DECLARED_SCOPE. 길이가 다른 두 문장의 역방향 mask 축 수정. 기존 독립 참조 대표 사례 3개 결과를 재사용했습니다. 전체 모델과 공식 runtime 동등성은 검증하지 않았습니다.

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Bahdanau Attention → Decomposable Attention](../../relations/tr_012.yaml) — I, tr_012
- [Decomposable Attention → Transformer](../../relations/tr_023.yaml) — I, tr_023
