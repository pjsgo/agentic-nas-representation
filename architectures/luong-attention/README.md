# Luong Attention

global/general attention, LSTM, input feeding.

[복원 YAML](representation.yaml) / [최종 구현](reconstructed.py) / [논문과 공식 코드](sources.yaml) / [범위, 코드 출처](provenance.json)

## 계산 흐름

이전 attentional vector → LSTM → bilinear score → context → attentional vector.

## 사용 계약

`build()`

`forward(self, memory, mask, target, h0, c0, feed0)`

CPU float64, YAML에 선언된 크기, dtype, 파라미터, 외부 상태 계약을 따릅니다. `build()`의 0 초기화는 학습된 가중치가 아닙니다. 비교, 사용 전에 YAML의 파라미터 값을 `model.p`에 주입합니다.

## 범위

제외: training/gradients, tokenization/embedding lookup, beam search, full translation system, original initializer fidelity, local-m/local-p/dot/concat/location alternatives, encoder execution and vocabulary projection, stacked decoder layers. 세부 연산과 상태 계약은 YAML을 기준으로 읽습니다.

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Bahdanau Attention → Luong Attention](../../relations/tr_008.yaml) — A, tr_008
- [Luong Attention → Transformer](../../relations/tr_022.yaml) — C, tr_022
