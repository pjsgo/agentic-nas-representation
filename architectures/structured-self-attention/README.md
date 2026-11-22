# Structured Self-Attention

bias 없는 다중 hop attention과 행렬 sentence embedding, encoder 제외.

[계산 YAML](representation.yaml), [최종 구현](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json)

## 사용 계약

```yaml
signature: pool(hidden, ws1, ws2, mask) -> dict
class_contract: No class or constructor; export this pure function in reconstructed.py
inputs:
  hidden: finite CPU float64[B,T,D] supplied contextual states
  ws1: finite CPU float64[A,D]
  ws2: finite CPU float64[R,A]
  mask: CPU bool[B,T]; True means valid
constraints:
- B,T,D,A,R positive; at least one valid token per batch item
- Computed intermediates must be finite except internal masked logits
- Empty or all-masked sequences raise ValueError; arbitrary boolean padding patterns
  supported
outputs:
  features: float64[B,T,A], zero on padding
  scores: float64[B,R,T], zero on padding
  attention: float64[B,R,T], zero on padding; sum over T is one for each batch and
    hop
  output: float64[B,R,D], one weighted sum per hop, no flatten or hop averaging
```

## 구현 범위

```yaml
included:
- bias_free_tanh_projection
- multiple_hop_scores
- masked_token_softmax
- attention_matrix
- matrix_sentence_embedding
excluded:
- BiLSTM_encoder
- embeddings
- diversity_penalty
- classifier
- training_dropout
- gradients
- initialization_distribution
- full_sentence_model
```

## 관련 관계

관계 유형은 [관계 안내](../../relations/README.md)를 참고합니다. `?`는 추론 관계입니다.

- [Structured Self-Attention → Transformer](../../relations/tr_024.yaml) — I?, tr_024
