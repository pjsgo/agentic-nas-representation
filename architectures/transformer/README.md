# Transformer

**복원 범위: model dimension 8, head 2, feed-forward dimension 12의 단일 encoder block.** 해당 범위에서 검증을 마친 YAML과 복원 코드입니다.

[YAML](representation.yaml) / [복원 코드](reconstructed.py) / [출처, 확인 범위](provenance.json)

## 구조도
```mermaid
flowchart LR
    x["입력 x"] --> qkv["Q/K/V projection, head 분리"]
    qkv --> scores["Scaled QK scores"]
    scores --> masking["Mask 적용"]
    mask["Mask 입력"] --> masking
    masking --> softmax["Softmax"] --> mix["V 가중합"]
    qkv -->|V| mix
    mix --> proj["Head 결합, output projection"] --> add1["Add"]
    x --> add1
    add1 --> norm1["LayerNorm"] --> ffn["FFN"] --> add2["Add"]
    norm1 --> add2
    add2 --> norm2["LayerNorm"] --> y["출력 y"]
```

두 residual 연결과 attention mask를 표시했습니다. FFN은 확장, 활성화, 축소 연산을 묶어 표시한 것입니다. 주요 계산을 묶어 보여주는 요약도이며, 세부 연산과 연결은 위 YAML과 코드를 기준으로 확인합니다.

## YAML이 표현하는 내용

`representation_version: 0.4-candidate.1`, 진입점 `unit`.

Q/K/V projection → masked attention → projection, residual, LayerNorm → FFN, residual, LayerNorm.

| YAML에서 볼 곳 | 복원 코드에서의 구현 |
|---|---|
| `stage_a`, `project_a`, `partition_a` | Q/K/V linear projection과 분할 |
| `score_a`, `apply_mask`, `prob_a`, `mix_a` | scaled dot-product, mask, softmax, value 혼합 |
| `stage_b`, `expand_a`, `activate_a`, `contract_a` | feed-forward 경로 |
| `merge_a/b`, `normalize_a/b`, `state_registry` | residual, LayerNorm 및 등록된 Parameter 유지 |

YAML은 연산과 연결, 속성, 공유, 상태를 기술합니다. 학습된 가중치나 실행 fixture를 저장하는 파일은 아닙니다. [공통 읽기 안내](../../docs/representation-guide.md)를 함께 참고하세요.

## 입력과 출력

| 방향 | 포트 | Shape |
|---|---|---|
| 입력 | `x` | `['B', 'T', 8]` |
| 입력 | `mask` | `['T', 'T']` |
| 출력 | `y` | `['B', 'T', 8]` |

## 복원 코드 사용 조건

`model = create_unit_a().eval()` → `y = model(x, mask)`.

p0/p1은 QKV, p2/p3는 attention output, p4/p5는 첫 LayerNorm, p6/p7, p8/p9는 FFN, p10/p11은 두 번째 LayerNorm입니다. 생성 시 `torch.empty`이므로 모든 값을 주입한 후 사용합니다. Boolean mask의 True 위치는 차단됩니다.

| YAML Parameter | Shape |
|---|---|
| `p0` | `[24, 8]` |
| `p1` | `[24]` |
| `p2` | `[8, 8]` |
| `p3` | `[8]` |
| `p4` | `[8]` |
| `p5` | `[8]` |
| `p6` | `[12, 8]` |
| `p7` | `[12]` |
| `p8` | `[8, 12]` |
| `p9` | `[8]` |
| `p10` | `[8]` |
| `p11` | `[8]` |

코드의 import, 생성, forward 이름은 위 계약을 따릅니다. 원본과 수치 비교하려면 입력, 초기 상태, Parameter/Buffer 값, dtype과 eval 조건을 동일하게 맞춰야 합니다. 저장된 검증 환경은 CPU, PyTorch 2.2.1입니다.

## 확인한 결과와 범위

CPU float32 fixture 9개에서 출력 포함 18개 추적값, 구조, 직접 연결, 연산 속성, Parameter 등록/수명을 검사했고 최대 오차 0입니다.

YAML 버전은 등록 상태를 명시하는 0.4-candidate.1입니다. 단일 블록의 eval 검증이며 전체 Transformer, 학습을 포함하지 않습니다. runtime trace 평가 기준은 이전 후보 관찰을 바탕으로 작성됐습니다.

이 폴더는 최신 사용 자료이며 상세 실험 로그, 검사기, fixture 묶음은 포함하지 않습니다. 검증 근거의 연구 작업 공간 기록은 provenance.json에 있습니다.

[전체 모델 목록](../../README.md)

## 관련 관계

유형과 대응 범위는 [관계 안내](../../docs/genealogy-guide.md)를 참고합니다. `?`는 추론 관계입니다.

- [ResNet → Transformer](../../relations/tr_019.yaml) — C, tr_019
- [LayerNorm → Transformer](../../relations/tr_020.yaml) — C, tr_020
- [Bahdanau Attention → Transformer](../../relations/tr_021.yaml) — I, tr_021
- [Luong Attention → Transformer](../../relations/tr_022.yaml) — C, tr_022
- [Seq2Seq → Transformer](../../relations/tr_026.yaml) — A, tr_026
- [Transformer → Relative Position](../../relations/tr_028.yaml) — A, tr_028
- [Transformer → Transformer-XL](../../relations/tr_029.yaml) — A, tr_029
- [Transformer → Sparse Transformer](../../relations/tr_032.yaml) — A, tr_032
- [Transformer → Reformer](../../relations/tr_033.yaml) — A, tr_033
- [Transformer → Linformer](../../relations/tr_036.yaml) — A, tr_036
- [Transformer → Linear Transformer](../../relations/tr_037.yaml) — A, tr_037
- [Transformer → Performer](../../relations/tr_038.yaml) — A, tr_038
- [Transformer → Synthesizer](../../relations/tr_040.yaml) — A, tr_040
- [Transformer → Pre-LN](../../relations/tr_042.yaml) — C, tr_042
- [Transformer → Switch Transformer](../../relations/tr_044.yaml) — A, tr_044
- [Transformer → FNet](../../relations/tr_045.yaml) — A, tr_045
- [Transformer → ALiBi](../../relations/tr_047.yaml) — A, tr_047
- [Transformer → H3](../../relations/tr_051.yaml) — X, tr_051
- [Transformer → FlashAttention](../../relations/tr_054.yaml) — A, tr_054
- [Transformer → RETRO](../../relations/tr_055.yaml) — C, tr_055
- [Transformer → GQA](../../relations/tr_056.yaml) — A, tr_056
- [Transformer → RetNet](../../relations/tr_058.yaml) — X, tr_058
- [Transformer → RWKV](../../relations/tr_059.yaml) — X, tr_059
- [Transformer → Jamba](../../relations/tr_065.yaml) — A, tr_065
- [Transformer → TTT](../../relations/tr_068.yaml) — X, tr_068
- [GNMT → Transformer](../../relations/tr_069.yaml) — P, tr_069
- [E2E Memory Network → Transformer](../../relations/tr_070.yaml) — I, tr_070
- [LSTMN → Transformer](../../relations/tr_071.yaml) — I, tr_071
- [Transformer → Hyena](../../relations/tr_076.yaml) — A, tr_076
