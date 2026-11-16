# Linformer

sequence 축 low-rank projection attention core.

[복원 YAML](representation.yaml) / [최종 구현](reconstructed.py) / [논문과 공식 코드](sources.yaml) / [범위, 코드 출처](provenance.json)

## 계산 흐름

K/V sequence projection → QK score → 압축 축 softmax → value 집계.

## 사용 계약

`build(spec: Dict[str, Any], resources: Dict[str, torch.Tensor])`

`forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor)`


CPU float64, YAML에 선언된 크기, dtype, 파라미터, 외부 상태 계약을 따릅니다. 생성자에 필요한 파라미터/자원은 호출자가 준비하며, `spec`이 필요한 경우 이 YAML을 파싱해 전달합니다.

평가에 사용한 외부 LLM 생성 코드를 계산 수정 없이 제공합니다.

## 범위

제외: full Transformer blocks, QKV and output feature projections, training, dropout, original class hierarchy, performance claims, feature projection biases (commuting projection boundary only valid without bias), incremental cache, padding, causal masking. 전체 모델 학습, 속도, 품질 재현을 주장하지 않습니다. 세부 연산과 상태 계약은 YAML을 기준으로 읽습니다.

## 관련 관계

- [Transformer → Linformer](../../relations/tr_036.yaml) — A (tr_036)

[연산 해석 규칙](../../specs/operators/linformer-v1.json)
