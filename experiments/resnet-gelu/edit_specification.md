# Controlled Factor Edit — 호출 전 명세

## 연구 질문과 Edit
사람이 지정한 단일 Factor Edit의 전달 정확성을 평가한다. Agent 자율 탐색이나 NAS 성능 향상을 시험하지 않는다.

[원래 YAML](original_input.yaml)에서 [Edited YAML](generator/input.yaml)로 다음 네 key path만 변경했다.

| 경로 | 원래 값 | Edited 값 | 이유 |
|---|---|---|---|
| factors.activation.operator | relu | gelu | 마지막 활성화 계산 변경 |
| factors.activation.attributes | inplace=true, target=input | approximate=none | GELU는 inplace를 지원하지 않음. 실제 라이브러리 기본값인 exact GELU를 명시 |
| resource.activation | shared_relu_module | post_add_gelu_module | 첫 ReLU와 독립 Module 객체로 분리 |
| operator_profile | pytorch_nn_2_2_1_v0_2 | pytorch_nn_2_2_1_v0_2_gelu1 | 공통 GELU operator 정의를 추가한 additive profile revision |

기존 첫 ReLU의 resource 문자열 `shared_relu_module`은 불필요한 이름 변경을 피하려고 유지했다. ID 문자열 자체는 공유를 뜻하지 않으며, 같은 ID의 반복 참조가 공유를 뜻한다. Edited YAML에서는 그 ID가 한 번만 사용된다.

## 반드시 보존할 항목
첫 ReLU의 operator·inplace·target·resource, 두 Conv와 BN의 모든 속성, identity Shortcut, main-target in-place Add, 11개 Dataflow edge, contains 계층, 입력 shape 계약과 static same_shape를 보존한다. 원래 dependency 항목은 없으며 새 dependency를 추가할 필요도 없다. GELU가 stateless Module이므로 Parameter/Buffer 추가는 없다.

## Independent Reference
`reference/edited_reference.py` (원본 연구 경로; 배포본 미포함)는 설치된 Torchvision BasicBlock 생성자를 상속하고, 실제 원본 forward의 마지막 `self.relu(out)` 호출만 별도 `self.post_add_activation(out)`으로 바꿨다. 첫 ReLU를 교체하거나 기존 Qwen 코드를 수정해 만들지 않았다.

원본 경로·SHA-256·행 번호는 `reference/reference_provenance.json` (원본 연구 경로; 배포본 미포함)에 있다. 호출 전 AST 대조로 마지막 호출 하나만 바뀌었음을 확인한다. 이 자료와 Reference Python은 LLM에게 제공하지 않는다.

## 수치 기준
[evaluation_plan.json](evaluation_plan.json)에 API 호출 전에 CPU·float32·eval·1 thread, rtol=1e-5·atol=1e-6을 고정했다. 기존 source-reference의 12개 상태 tensor와 7개 입력을 값과 layout까지 재사용하고, `[1,64,1,1]`에 -0.5를 채운 음수 witness를 추가했다.

비교는 두 가지로 분리한다.

1. **Generated ↔ Edited Reference:** 전체 상태를 명확히 일대일 대응한 뒤 8개 모두 assert_close를 통과해야 한다.
2. **Edited Reference ↔ Original:** 출력 차이가 관찰되어야 한다. 두 모델이 같아야 한다는 조건을 적용하지 않는다. 음수 witness에서는 GELU의 음수 출력도 확인한다.

추가로 main 내부 5개 operator 출력과 Add 직후·최종 활성화 직전 텐서는 원본/Edited Reference 사이에 같아야 한다. 이 검사는 의도하지 않은 앞 경로 변경을 분리한다.

## Edit 전용 구조 검증
[validate_edit.py](validate_edit.py)는 candidate 클래스명·속성명·정확한 AST 문장을 정답으로 고정하지 않는다. 수동 전체 소스 검토 후 공개된 class/factory를 adapter metadata로 지정하고, 실행 hook의 operator 종류·순서·tensor 연결에서 semantic role을 찾아 매핑한다.

Resource ID와 실제 객체 동일성, 첫 ReLU in-place, Add의 main tensor 갱신, 마지막 GELU의 out-of-place와 approximate=none을 확인한다. 상태 대응은 semantic role과 Module 내부 field 이름으로 정하며, shape만 맞추거나 Parameter를 임의로 생략하지 않는다. 비어 있거나 누락된 테스트 집합은 실패다.

이 검증기는 이 Edit에 필요한 graph pattern을 평가하는 전용 도구이며 일반 NAS 검증기는 아니다. API 응답 이후에는 생성 코드의 실행 적합성 검토와 entrypoint 선택만 허용하며, Reference·평가 계획·검사 구현을 생성 결과에 맞춰 바꾸지 않는다.

## LLM 요청 경계
기존 중립 Prompt 바이트는 유지한다. Qwen3.8-27B, temperature=0, max_tokens=8192로 1회만 호출한다. payload는 **중립 Prompt + Edited YAML + 공통 Semantics 및 GELU operator 확장**뿐이다. Source Mapping, 이 문서, 원본 Python, Reference Python, fixture, 기존 생성 코드는 제외한다.

GELU 확장은 위치나 edit 정답을 지시하지 않는 공통 operator 규칙이다. 기존 Semantics 파일은 수정하지 않았고, 별도 사본에 additive 규칙을 붙였다.

## 성공 판정과 한계
형식·구문·Edit/구조·Resource·실행·상태 매핑·Edited Reference 수치 일치·실제 edit 관측을 각각 기록한다. 생성 코드 수정 없이 모두 통과해야 전체 성공이다. 단일 생성·단일 블록·하나의 상태 fixture와 eval CPU 범위이며, gradient·train mode·일반적인 Agent Edit 능력·NAS 성능은 주장하지 않는다.
