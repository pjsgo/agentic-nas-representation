# Neural LM

고정 문맥 lookup, tanh hidden, direct 경로와 vocabulary readout.

[계산 YAML](representation.yaml), [최종 구현](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json)

## 사용 계약

```yaml
function: neural_lm(tokens, C, H, d, U, W, b) -> dict
class_contract: No class or constructor required. Export this top-level function;
  caller supplies all weights. No global state or random initialization.
dimensions: B=batch, K=context count=n-1, E=embedding width, D=hidden width, V=vocabulary;
  all positive integers. Q=K*E.
tensors: tokens int64 CPU [B,K], values in [0,V). All other inputs finite float64
  CPU tensors; finite intermediate arithmetic assumed.
inputs:
  tokens: '[B,K], position j is word w_(t-1-j), most recent first; caller builds complete
    context'
  C: '[V,E], shared token embedding rows'
  H: '[D,Q], hidden projection, source column-vector convention'
  d: '[D], hidden bias'
  U: '[V,D], hidden-to-vocabulary projection'
  W: '[V,Q], direct feature-to-vocabulary projection; required, zero matrix selects
    no-direct variant'
  b: '[V], vocabulary bias'
outputs:
  features: '[B,Q], feature index q=j*E+e'
  hidden: '[B,D]'
  logits: '[B,V], unnormalized next-word logits'
  probabilities: '[B,V], softmax over vocabulary only'
invalid_inputs: Raise ValueError for non-tensors, wrong dtype/device/rank/shape, nonfinite
  floating inputs, empty dimensions, or out-of-range token IDs.
```

## 구현 범위

```yaml
included:
- shared_token_lookup
- ordered_context_concatenation
- biased_tanh_hidden
- direct_feature_to_vocabulary
- biased_hidden_to_vocabulary
- full_vocabulary_softmax
excluded:
- training
- gradients
- corpus_processing
- context_window_construction
- padding
- unknown_word_policy
- sampling
- ngram_interpolation
- distributed_execution
```

## 관련 관계

관계 유형은 [관계 안내](../../relations/README.md)를 참고합니다. `?`는 추론 관계입니다.

- [Neural LM → RNN LM](../../relations/tr_002.yaml) — P?, tr_002
