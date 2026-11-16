# 아키텍처 관계

관계 YAML은 변경 이유·계산·보존 부분·적용 조건·효과·대가·출처와 양쪽 복원 YAML의 부품 ID를 연결합니다.

`complete_scoped`는 선택한 핵심 계산의 대응, `partial`은 주요 개념 일부가 복원 범위 밖에 있음을 뜻합니다. 전체 모델 변환 성공 여부와는 별개입니다. `?`는 추론 관계이며 역사적 직접 계승으로 해석하지 않습니다.

공개 관계는 설명용입니다. 연구 공간에서 실행된 변환의 증거는 해당 YAML의 `historical_execution`에 별도로 보존합니다.

| 관계 | 유형 | 부품 매핑 |
|---|---|---|
| [Simple RNN → LSTM](tr_003.yaml) | A · tr_003 | partial |
| [LSTM → Seq2Seq](tr_004.yaml) | C · tr_004 | complete_scoped |
| [RNN Encoder–Decoder/GRU → Bahdanau](tr_006.yaml) | A · tr_006 | partial |
| [Seq2Seq → Bahdanau](tr_007.yaml) | P · tr_007 | partial |
| [Bahdanau → Luong](tr_008.yaml) | A · tr_008 | complete_scoped |
| [Bahdanau → GNMT](tr_014.yaml) | C · tr_014 | complete_scoped |
| [ResNet → Transformer](tr_019.yaml) | C · tr_019 | complete_scoped |
| [LayerNorm → Transformer](tr_020.yaml) | C · tr_020 | complete_scoped |
| [Luong → Transformer](tr_022.yaml) | C · tr_022 | complete_scoped |
| [Seq2Seq → Transformer](tr_026.yaml) | A · tr_026 | partial |
| [Transformer → Transformer-XL](tr_029.yaml) | A · tr_029 | complete_scoped |
| [Transformer-XL → Compressive Transformer](tr_031.yaml) | A · tr_031 | complete_scoped |
| [Transformer → Sparse Transformer](tr_032.yaml) | A · tr_032 | complete_scoped |
| [Transformer → Linformer](tr_036.yaml) | A · tr_036 | complete_scoped |
| [Transformer → Linear Transformer](tr_037.yaml) | A · tr_037 | complete_scoped |
| [Transformer → Performer](tr_038.yaml) | A · tr_038 | complete_scoped |
| [Linear Transformer → Performer](tr_039.yaml) | I? · tr_039 | complete_scoped |
| [LSSL → S4](tr_049.yaml) | A · tr_049 | complete_scoped |
| [S4 → H3](tr_050.yaml) | C · tr_050 | complete_scoped |
| [S4 → Mamba-1](tr_061.yaml) | A · tr_061 | complete_scoped |
| [H3 → Mamba](tr_062.yaml) | A · tr_062 | partial |
| [Mamba → Mamba-2/SSD](tr_063.yaml) | A · tr_063 | partial |
| [GNMT → Transformer](tr_069.yaml) | P · tr_069 | partial |

[표현 안내](../docs/genealogy-guide.md) · [기계 판독 목록](catalog.yaml)
