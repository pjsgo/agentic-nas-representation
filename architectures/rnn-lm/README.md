# RNN LM

2010 sigmoid recurrent core와 flat vocabulary softmax, 명시적 초기 상태.

[계산 YAML](representation.yaml), [최종 구현](reconstructed.py), [출처](sources.yaml), [구현 기록](provenance.json)

## 사용 계약

```yaml
function: rnn_lm(tokens, h0, A, R, O) -> dict
class_contract: No class or constructor required. Export this top-level function with
  all weights and initial state supplied. No global state or random initialization.
dimensions: T=time, B=batch, V=vocabulary, D=hidden width; all positive integers.
tensors: tokens int64 CPU [T,B], values in [0,V). h0,A,R,O finite float64 CPU; finite
  intermediate arithmetic assumed.
inputs:
  tokens: '[T,B], current observed word at each t; output predicts the next word'
  h0: '[B,D], caller-supplied state before first token, e.g. all 0.1 for source initialization'
  A: '[V,D], rows of one-hot input-to-hidden map, not a separate embedding followed
    by another projection'
  R: '[D,D], row-vector previous-state to new-state map'
  O: '[D,V], row-vector hidden-to-vocabulary map'
outputs:
  hidden: '[T,B,D], post-update state for each observed token'
  logits: '[T,B,V]'
  probabilities: '[T,B,V], full vocabulary distribution per time and batch'
  final_state: '[B,D], equals hidden[T-1], independent copy (no storage alias)'
invalid_inputs: Raise ValueError for non-tensors, wrong dtype/device/rank/shape, nonfinite
  floating inputs, empty dimensions including T=0, or out-of-range token IDs.
```

## 구현 범위

```yaml
included:
- one_hot_input_equivalent_lookup
- sigmoid_recurrence
- explicit_initial_state
- hidden_to_vocabulary
- full_vocabulary_softmax
- returned_final_state
excluded:
- training
- bptt
- dynamic_test_adaptation
- class_based_output
- toolkit_fast_exp_and_clipping
- compression_layer
- direct_maxent_features
- corpus_processing
- padding
- automatic_sentence_reset
- sampling
- ngram_interpolation
```

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Simple RNN → RNN LM](../../relations/tr_001.yaml) — A, tr_001
- [Neural LM → RNN LM](../../relations/tr_002.yaml) — P?, tr_002
