# Transformer-XL

segment recurrence, 상대 위치 attention sublayer.

[복원 YAML](representation.yaml) / [최종 구현](reconstructed.py) / [논문과 공식 코드](sources.yaml) / [범위, 코드 출처](provenance.json)

## 계산 흐름

이전 segment memory, 현재 입력 → 상대 위치 causal attention → postnorm → memory 갱신.

## 사용 계약

`build(spec, parameters)`

`forward(self, x, memory)`


CPU float64, YAML에 선언된 크기, dtype, 파라미터, 외부 상태 계약을 따릅니다. 생성자에 필요한 파라미터/자원은 호출자가 준비하며, `spec`이 필요한 경우 이 YAML을 파싱해 전달합니다.

외부 LLM 생성 후 계산 오류를 수정한 코드입니다. 원본 실패와 수정본의 판정은 출처 기록에서 구분합니다.

## 범위

제외: ffn, token_embedding, adaptive_softmax, training, dropout, pre_layernorm, same_length_mask, ext_len_nonzero, position_clamp. 전체 모델 학습, 속도, 품질 재현을 주장하지 않습니다. 세부 연산과 상태 계약은 YAML을 기준으로 읽습니다.

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [Transformer → Transformer-XL](../../relations/tr_029.yaml) — A, tr_029
- [Relative Position → Transformer-XL](../../relations/tr_030.yaml) — C, tr_030
- [Transformer-XL → Compressive Transformer](../../relations/tr_031.yaml) — A, tr_031
