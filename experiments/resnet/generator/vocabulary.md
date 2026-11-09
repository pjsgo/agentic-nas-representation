# Representation Semantics v0.2

대상 입력: [input_resnet_source_grounded_v0_2.yaml](input.yaml). 이 문서는 이후 Reconstruction LLM에 YAML과 함께 제공하는 공통 해석 규칙이다. 원본 경로·Source Mapping·검증 보고서는 제공하지 않는다. 현재 명세는 확정했으나 복원·Forward·수치 동등성은 아직 검증하지 않았다.

## 구조와 참조

- `representation_version: "0.2"`와 `operator_profile: pytorch_nn_2_2_1_v0_2`를 함께 해석한다. `architecture`는 설명용 이름이며 암묵적인 연산·연결을 추가하는 근거가 아니다.
- `factors`의 키는 문서 안에서 유일한 factor ID다. `contains`는 부모–자식 membership만 정한다. 배열 순서는 실행 순서가 아니다. 각 자식은 정확히 한 부모에 속한다.
- `dataflow`의 `[source, destination]`은 직접적인 텐서 전달이다. 배열의 나열 순서 대신 edge의 의존관계가 실행을 정한다. 암묵적인 clone·detach·cast·reshape를 추가하지 않는다.
- 컨테이너 `main.input`과 `main.output`은 경계 포트다. 전자는 외부에서 받은 텐서를 자식에게 전달하고, 후자는 자식의 결과를 외부로 전달한다. 자체 연산이나 Module 객체가 아니다. `block`은 전체 그래프의 루트다.
- 단항 operator의 포트는 `input`, `output`; `add`의 포트는 `main`, `shortcut`, `output`이다. 전역 입출력 `x`, `y`는 `interface.block`이 선언한다. factor ID가 전역 유일하므로 내부 factor 참조에 부모 접두사를 중복하지 않는다.

## Resource Identity

`resource`는 **연산 호출 factor → Module 객체 ID**의 매핑이다. 동일 ID를 가진 factor는 호출 위치가 달라도 정확히 같은 Module 인스턴스를 사용한다. 다른 ID는 별도 인스턴스이며, 동일한 operator·attributes만으로 객체나 파라미터를 공유하지 않는다.

Module 자원은 파라미터뿐 아니라 buffers와 train/eval 상태를 포함한다. 동일 ID의 operator·attributes는 일치해야 한다. 파라미터가 없는 ReLU도 객체 동일성을 표현할 수 있다. 컨테이너와 기능적 `add`에는 Module 자원을 만들지 않는다. 자원 ID는 학습된 weight 값이나 `state_dict` 키 이름을 뜻하지 않는다.

## In-place와 target

`attributes.inplace: true`이면 `target`은 **해당 factor의 입력 포트명**이며, 그 포트에 도착한 텐서의 저장공간을 갱신한다. 출력은 갱신된 텐서를 alias한다. Module 객체나 원본 graph 입력을 임의로 갱신한다는 뜻이 아니다.

- `relu`의 `target: input`: 도착한 텐서에 in-place ReLU를 적용한다. 공유 ReLU 객체를 사용해도 두 호출의 입력 텐서가 같아지는 것은 아니다.
- `add`의 `target: main`: `main`에 도착한 텐서에 `shortcut`을 더해 갱신한다. PyTorch에서는 `main.add_(shortcut)`에 대응한다. `shortcut`을 갱신하거나 autograd 의미가 다른 `out=` 연산으로 대체하지 않는다.
- 이 프로파일의 `add`는 일반 PyTorch의 덧셈·broadcast 의미를 따른다. 추가적인 shape 계약은 `interface`가 정한다. mutation이 여러 소비자의 읽기 순서에 영향을 주는 그래프는 별도 순서 정보가 필요하다. 현재 입력에는 그런 미결정 의존관계가 없다.

## Interface

`interface.block.input`은 외부 포트명과 shape 계약이다. `[N,64,H,W]`는 NCHW 축 순서이며, N·H·W는 양의 정수 심볼이다. 구체 배치·공간 크기·dtype·device를 새로 고정하지 않는다. 출력 shape는 operator와 edge에서 유도하며 입력과 같다.

`interface.merge.require: same_shape(main, shortcut)`는 해당 merge의 두 입력 포트 shape가 같다는 **정적 제약**이다. 자동 Runtime Guard, assert, 보정 projection, padding, reshape, clone을 추가하라는 지시가 아니다. 라이브러리 operator 자체의 기존 입력 검사는 유지한다. BN의 4차원 입력 요구와 학습 시 채널당 값이 하나인 입력의 거부도 이에 포함된다.

## 공통 Operator 프로파일과 버전

프로파일 `pytorch_nn_2_2_1_v0_2`는 **설치된 PyTorch 2.2.1**의 다음 의미를 고정한다. Torchvision 0.17.1은 검증 대상의 출처 버전이며, operator 정의를 대신하지 않는다. 버전 변경 시 기본값·상태·mutation 의미를 다시 검토하고 프로파일을 갱신한다.

| Operator | 정의 |
|---|---|
| `conv2d` | `torch.nn.Conv2d` Module. attributes는 같은 이름의 생성자 인자다. 공간 속성의 스칼라는 두 공간 축에 동일하게 적용된다. |
| `batch_norm2d` | `torch.nn.BatchNorm2d` Module. affine 파라미터, running_mean·running_var·num_batches_tracked, train/eval별 계산·갱신을 포함한다. 단순한 고정 수식으로 축약하지 않는다. |
| `relu` | `torch.nn.ReLU` Module. `inplace`는 생성자 속성이고 `target`은 이 표현에서 추가한 mutation 포트 지정이다. |
| `add` | 두 텐서의 원소별 덧셈. Module을 요구하지 않는다. `inplace`와 `target`은 앞 절의 규칙을 적용한다. |

명시 attributes가 우선한다. 이 프로파일의 Module 생성은 해당 라이브러리의 기본 파라미터·buffer 초기화와 기본 training 상태를 유지한다. BN은 affine weight=1·bias=0, running mean=0·variance=1·counter=0으로 시작하며 Conv는 라이브러리의 기본 초기화를 사용한다. dtype·device는 라이브러리 기본 생성 문맥을 따른다. 별도 전체 모델 생성자의 초기화나 pretrained weights를 추가하지 않는다.

## 평가 경계

이 명세는 계산 구조·연산 설정·객체/상태 동일성·mutation 의미를 정의한다. 고정 weight 값, RNG seed, 실제 입력, dtype/device, 평가 모드는 이번 Representation에 임의로 넣지 않는다. 이후 수치 비교에서는 양쪽 모델의 파라미터와 BN buffers, 모드, 입력 및 실행 조건을 별도 평가 fixture에서 맞춰야 한다. 코드의 클래스 이름·속성 이름·정확한 문자열 일치는 요구하지 않는다.
