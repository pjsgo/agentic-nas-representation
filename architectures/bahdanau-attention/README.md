# Bahdanau Attention

제공된 encoder memory에 대한 additive attention, 조건부 GRU.

[복원 YAML](representation.yaml) / [최종 구현](reconstructed.py) / [논문과 공식 코드](sources.yaml) / [범위, 코드 출처](provenance.json)

## 계산 흐름

이전 decoder 상태, source memory → additive alignment → context → GRU 갱신.

## 사용 계약

`build()`

`forward(self, memory: torch.Tensor, mask: torch.Tensor, target: torch.Tensor, h0: torch.Tensor)`

CPU float64, YAML에 선언된 크기, dtype, 파라미터, 외부 상태 계약을 따릅니다. `build()`의 0 초기화는 학습된 가중치가 아닙니다. 비교, 사용 전에 YAML의 파라미터 값을 `model.p`에 주입합니다.

## 범위

제외: training/gradients, tokenization/embedding lookup, beam search, full translation system, original initializer fidelity, bidirectional encoder execution; annotations supplied at boundary, initial-state projection from backward encoder, deep maxout vocabulary head. 세부 연산과 상태 계약은 YAML을 기준으로 읽습니다.

## 관련 관계

관계 유형은 [관계 안내](../../relations/README.md)를 참고합니다. `?`는 추론 관계입니다.

- [RNN Encoder–Decoder / GRU → Bahdanau Attention](../../relations/tr_006.yaml) — A, tr_006
- [Seq2Seq → Bahdanau Attention](../../relations/tr_007.yaml) — P, tr_007
- [Bahdanau Attention → Luong Attention](../../relations/tr_008.yaml) — A, tr_008
- [Bahdanau Attention → Decomposable Attention](../../relations/tr_012.yaml) — I, tr_012
- [Bahdanau Attention → GNMT](../../relations/tr_014.yaml) — C, tr_014
- [Bahdanau Attention → ConvS2S](../../relations/tr_017.yaml) — C, tr_017
- [Bahdanau Attention → Transformer](../../relations/tr_021.yaml) — I, tr_021
