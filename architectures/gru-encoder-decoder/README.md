# RNN Encoder–Decoder / GRU

Cho 2014 encoder, 고정 context, 조건부 GRU decoder.

[복원 YAML](representation.yaml) / [최종 구현](reconstructed.py) / [논문과 공식 코드](sources.yaml) / [범위, 코드 출처](provenance.json)

## 계산 흐름

encoder scan → 고정 context → decoder 초기화, 조건부 scan.

## 사용 계약

`build()`

`forward(self, source: torch.Tensor, target: torch.Tensor)`

CPU float64, YAML에 선언된 크기, dtype, 파라미터, 외부 상태 계약을 따릅니다. `build()`의 0 초기화는 학습된 가중치가 아닙니다. 비교, 사용 전에 YAML의 파라미터 값을 `model.p`에 주입합니다.

## 범위

제외: training/gradients, tokenization/embedding lookup, beam search, full translation system, original initializer fidelity, maxout output head and factorized vocabulary classifier. 세부 연산과 상태 계약은 YAML을 기준으로 읽습니다.

## 관련 관계

관계 유형은 [관계 안내](../../relations/README.md)를 참고합니다. `?`는 추론 관계입니다.

- [RNN Encoder–Decoder / GRU → Seq2Seq](../../relations/tr_005.yaml) — X?, tr_005
- [RNN Encoder–Decoder / GRU → Bahdanau Attention](../../relations/tr_006.yaml) — A, tr_006
- [LSTM → RNN Encoder–Decoder / GRU](../../relations/tr_073.yaml) — I, tr_073
