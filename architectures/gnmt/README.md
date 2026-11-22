# GNMT

3층 encoder–decoder, 양방향 하단 encoder, residual, attention.

[복원 YAML](representation.yaml) / [최종 구현](reconstructed.py) / [논문과 공식 코드](sources.yaml) / [범위, 코드 출처](provenance.json)

## 계산 흐름

양방향 encoder → 단방향 상위층, residual → 하단 decoder attention → 상위 decoder.

## 사용 계약

`build()`

`forward(self, source, target)`

CPU float64, YAML에 선언된 크기, dtype, 파라미터, 외부 상태 계약을 따릅니다. `build()`의 0 초기화는 학습된 가중치가 아닙니다. 비교, 사용 전에 YAML의 파라미터 값을 `model.p`에 주입합니다.

## 범위

제외: training/gradients, tokenization/embedding lookup, beam search, full translation system, original initializer fidelity, 8-layer/1024-wide configuration, gnmt_v2 current-attention broadcast alternative, wordpieces, quantization, length/coverage beam penalties, multi-GPU placement, padding/variable lengths within a batch, vocabulary projection/softmax. 세부 연산과 상태 계약은 YAML을 기준으로 읽습니다.

## 관련 관계

관계 유형과 필드 설명은 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [LSTM → GNMT](../../relations/tr_013.yaml) — A, tr_013
- [Bahdanau Attention → GNMT](../../relations/tr_014.yaml) — C, tr_014
- [ResNet → GNMT](../../relations/tr_018.yaml) — C, tr_018
- [GNMT → Transformer](../../relations/tr_069.yaml) — P, tr_069
