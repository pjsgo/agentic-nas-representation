# Simple RNN

Elman tanh 순환 core.

[복원 YAML](representation.yaml) / [최종 구현](reconstructed.py) / [논문과 공식 코드](sources.yaml) / [범위, 코드 출처](provenance.json)

## 계산 흐름

입력, 이전 hidden → affine 합산 → tanh → 다음 hidden.

## 사용 계약

`build()`

`forward(self, x: torch.Tensor, h0: torch.Tensor)`

CPU float64, YAML에 선언된 크기, dtype, 파라미터, 외부 상태 계약을 따릅니다. `build()`의 0 초기화는 학습된 가중치가 아닙니다. 비교, 사용 전에 YAML의 파라미터 값을 `model.p`에 주입합니다.

## 범위

제외: training/gradients, tokenization/embedding lookup, beam search, full translation system, original initializer fidelity. 세부 연산과 상태 계약은 YAML을 기준으로 읽습니다.

## 관련 관계

관계 유형과 필드 설명은 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Simple RNN → RNN LM](../../relations/tr_001.yaml) — A, tr_001
- [Simple RNN → LSTM](../../relations/tr_003.yaml) — A, tr_003
