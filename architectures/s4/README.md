# S4

**복원 범위: original rank-1 DPLR/NPLR recurrent, impulse-kernel core, Nfull=4, H=channels=1.** 해당 범위에서 검증을 마친 YAML과 복원 코드입니다.

[YAML](representation.yaml) / [복원 코드](reconstructed.py) / [출처, 확인 범위](provenance.json)

## 구조도
```mermaid
flowchart LR
    params["외부 Parameter"] --> prep["계수 준비, 이산화"]
    prep --> seq["Sequence 상태 반복, readout"]
    u["입력 u"] --> seq
    h0["초기 h0"] --> seq
    seq --> add["Add"]
    u --> skip["D * u"] --> add
    params -->|D| skip
    add --> y["출력 y"]
    seq --> hf["최종 상태"]
    prep --> impulse["Impulse-kernel 반복"]
    impulse --> kernel["Kernel"]
```

동일한 준비 계수에서 sequence와 impulse-kernel 경로가 나뉩니다. 각 반복 내부의 상태 갱신은 하나의 노드로 요약했습니다. 주요 계산을 묶어 보여주는 요약도이며, 세부 연산과 연결은 위 YAML과 코드를 기준으로 확인합니다.

## YAML이 표현하는 내용

`representation_version: 0.3-candidate.1`, 진입점 `unit`.

raw Parameter 준비, 이산화 → sequence scan / impulse-kernel scan → direct feedthrough D 합산.

| YAML에서 볼 곳 | 복원 코드에서의 구현 |
|---|---|
| `prepare`, `f0000`… | `_prepare()`의 계수 계산 |
| `sequence_scan`, `sequence_body` | `_sequence_body()`와 시간축 `select(2,t)` |
| `f0167`, `kernel_scan`, `kernel_body` | batch 첫 항목 slice 및 `_kernel_body()`의 impulse recurrence |
| `f0170`…`f0172` | D, u direct feedthrough와 최종 y |

YAML은 연산과 연결, 속성, 공유, 상태를 기술합니다. 학습된 가중치나 실행 fixture를 저장하는 파일은 아닙니다. [공통 읽기 안내](../../docs/representation-guide.md)를 함께 참고하세요.

## 입력과 출력

| 방향 | 포트 | Shape |
|---|---|---|
| 입력 | `u` | `['B', 1, 'T']` |
| 입력 | `h0` | `['B', 1, 2, 2]` |
| 출력 | `kernel` | `[1, 1, 'T']` |
| 출력 | `y` | `['B', 1, 'T']` |
| 출력 | `h_final` | `['B', 1, 2, 2]` |

## 복원 코드 사용 조건

`model = create_model().eval()` → `kernel, y, h_final = model(u, h0)`.

p0…p6의 역할은 inv_dt, A_real, A_imag, B, C, P, D 순서입니다. 생성 시 `torch.empty`이므로 7개 값을 모두 주입해야 합니다. float64의 마지막 길이 2 축은 복소수 실수부/허수부 표현에 사용됩니다.

| YAML Parameter | Shape |
|---|---|
| `p0` | `[1, 1]` |
| `p1` | `[1, 2]` |
| `p2` | `[1, 2]` |
| `p3` | `[1, 1, 2, 2]` |
| `p4` | `[1, 1, 2, 2]` |
| `p5` | `[1, 1, 2, 2]` |
| `p6` | `[1, 1]` |

코드의 import, 생성, forward 이름은 위 계약을 따릅니다. 원본과 수치 비교하려면 입력, 초기 상태, Parameter/Buffer 값, dtype과 eval 조건을 동일하게 맞춰야 합니다. 저장된 검증 환경은 CPU, PyTorch 2.2.1입니다.

## 확인한 결과와 범위

기존 엄격 검사에서 그래프, 수치 PASS, 실용 검사 90회 PASS, primitive 중간값 152개와 매 시점 carry 비교 PASS입니다.

원문 생성 코드에서 slice 축 1곳과 scan select 표현 2곳을 수정한 최종본입니다. 전체 FFT/cache 경로, S4D, 일반 rank, 학습은 포함하지 않습니다.

이 폴더는 최신 사용 자료이며 상세 실험 로그, 검사기, fixture 묶음은 포함하지 않습니다. 검증 근거의 연구 작업 공간 기록은 provenance.json에 있습니다.

[전체 모델 목록](../../README.md)

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [LSSL → S4](../../relations/tr_049.yaml) — A, tr_049
- [S4 → H3](../../relations/tr_050.yaml) — C, tr_050
- [S4 → Hyena](../../relations/tr_053.yaml) — X?, tr_053
- [S4 → Mamba](../../relations/tr_061.yaml) — A, tr_061
