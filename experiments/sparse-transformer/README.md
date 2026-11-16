# Sparse Transformer

local/strided causal attention core.

[복원 YAML](representation.yaml) · [최종 구현](repaired_model.py) · [논문·공식 코드](sources.yaml) · [범위·코드 출처](provenance.json)

## 계산 흐름

local 또는 strided 연결 → causal mask → scaled softmax → value 집계.

## 사용 계약

`build(spec: Dict[str, Any], resources: Dict[str, torch.Tensor])`

`forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor)`


CPU float64, YAML에 선언된 크기·dtype·파라미터·외부 상태 계약을 따릅니다. 생성자에 필요한 파라미터/자원은 호출자가 준비하며, `spec`이 필요한 경우 이 YAML을 파싱해 전달합니다.

외부 LLM 생성 후 계산 오류를 수정한 코드입니다. 원본 실패와 수정본의 판정은 출처 기록에서 구분합니다.

## 범위

제외: full Transformer blocks, QKV and output feature projections, training, dropout, original class hierarchy, performance claims, fixed pattern, block-sparse CUDA kernel, layer alternation schedule. 전체 모델 학습·속도·품질 재현을 주장하지 않습니다. 세부 연산과 상태 계약은 YAML을 기준으로 읽습니다.

## 관련 관계

- [Transformer → Sparse Transformer](../../relations/tr_032.yaml) — A · tr_032

[연산 해석 규칙](operator-semantics.json)
