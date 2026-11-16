# H3

scalar-head shift·diagonal SSM gated block.

[복원 YAML](representation.yaml) · [최종 구현](generated_model.py) · [논문·공식 코드](sources.yaml) · [범위·코드 출처](provenance.json)

## 계산 흐름

Q/K/V projection → K shift memory → V gate → diagonal memory → Q gate → output projection.

## 사용 계약

`build(values)`

`forward(self, u, shift0, diagonal0)`


CPU float32, YAML에 선언된 고정 크기와 외부 상태 계약을 따릅니다. 생성자에 필요한 파라미터/자원은 호출자가 준비하며, `spec`이 필요한 경우 이 YAML을 파싱해 전달합니다.

평가에 사용한 외부 LLM 생성 코드를 계산 수정 없이 제공합니다.

## 범위

제외: full model, training/gradients, FFT/cache backend, initialization equivalence, head_dim>1, automatic relation conversion, performance claims. 전체 모델 학습·속도·품질 재현을 주장하지 않습니다. 세부 연산과 상태 계약은 YAML을 기준으로 읽습니다.

관계 부품 ID는 별도 `genealogy/representation.yaml`을 참조합니다. 기본 복원 YAML은 primitive 표현이며 두 표현의 ID를 혼용하지 않습니다.

## 관련 관계

- [S4 → H3](../../relations/tr_050.yaml) — C · tr_050
- [H3 → Mamba](../../relations/tr_062.yaml) — A · tr_062

[연산 해석 규칙](operator-semantics.txt)
