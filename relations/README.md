# 아키텍처 계보 관계

45개 노드, 64개 설명 관계. 기록 완료와 실행 가능 여부는 별도입니다.

[전체 관계도](../docs/genealogy-map.md) · [5차 검증과 범위](../docs/phase5-readiness.md) · [남은 범위](remaining-scope.yaml)

| ID | 출발 → 도착 | 유형 | 전체 대응 | core 대응 | 공개 변환 실행 |
|---|---|---|---|---|---|
| [tr_003](tr_003.yaml) | Simple RNN → LSTM | A | partial | partial | 불가 |
| [tr_004](tr_004.yaml) | LSTM → Seq2Seq | C | complete_scoped | complete_scoped | 불가 |
| [tr_005](tr_005.yaml) | RNN Encoder–Decoder / GRU → Seq2Seq | X? | partial | complete_scoped | 불가 |
| [tr_006](tr_006.yaml) | RNN Encoder–Decoder/GRU → Bahdanau | A | partial | partial | 불가 |
| [tr_007](tr_007.yaml) | Seq2Seq → Bahdanau | P | partial | partial | 불가 |
| [tr_008](tr_008.yaml) | Bahdanau → Luong | A | complete_scoped | complete_scoped | 불가 |
| [tr_009](tr_009.yaml) | Memory Network → E2E Memory Network | A | partial | complete_scoped | 불가 |
| [tr_010](tr_010.yaml) | LSTM → LSTMN | A | partial | complete_scoped | 불가 |
| [tr_011](tr_011.yaml) | E2E Memory Network → LSTMN | I | partial | complete_scoped | 불가 |
| [tr_013](tr_013.yaml) | LSTM → GNMT | A | partial | complete_scoped | 불가 |
| [tr_014](tr_014.yaml) | Bahdanau → GNMT | C | complete_scoped | complete_scoped | 불가 |
| [tr_018](tr_018.yaml) | ResNet → GNMT | C | partial | complete_scoped | 불가 |
| [tr_019](tr_019.yaml) | ResNet → Transformer | C | complete_scoped | complete_scoped | 불가 |
| [tr_020](tr_020.yaml) | LayerNorm → Transformer | C | complete_scoped | complete_scoped | 불가 |
| [tr_021](tr_021.yaml) | Bahdanau Attention → Transformer | I | partial | complete_scoped | 불가 |
| [tr_022](tr_022.yaml) | Luong → Transformer | C | complete_scoped | complete_scoped | 불가 |
| [tr_026](tr_026.yaml) | Seq2Seq → Transformer | A | partial | partial | 불가 |
| [tr_028](tr_028.yaml) | Transformer → Relative Position | A | partial | complete_scoped | 불가 |
| [tr_029](tr_029.yaml) | Transformer → Transformer-XL | A | complete_scoped | complete_scoped | 불가 |
| [tr_030](tr_030.yaml) | Relative Position → Transformer-XL | C | partial | complete_scoped | 불가 |
| [tr_031](tr_031.yaml) | Transformer-XL → Compressive Transformer | A | complete_scoped | complete_scoped | 불가 |
| [tr_032](tr_032.yaml) | Transformer → Sparse Transformer | A | complete_scoped | complete_scoped | 불가 |
| [tr_033](tr_033.yaml) | Transformer → Reformer | A | partial | complete_scoped | 불가 |
| [tr_034](tr_034.yaml) | Sparse Transformer → Longformer | I | partial | partial | 불가 |
| [tr_035](tr_035.yaml) | Sparse Transformer → BigBird | I | partial | partial | 불가 |
| [tr_036](tr_036.yaml) | Transformer → Linformer | A | complete_scoped | complete_scoped | 불가 |
| [tr_037](tr_037.yaml) | Transformer → Linear Transformer | A | complete_scoped | complete_scoped | 불가 |
| [tr_038](tr_038.yaml) | Transformer → Performer | A | complete_scoped | complete_scoped | 불가 |
| [tr_039](tr_039.yaml) | Linear Transformer → Performer | I? | complete_scoped | complete_scoped | 불가 |
| [tr_040](tr_040.yaml) | Transformer → Synthesizer | A | partial | complete_scoped | 불가 |
| [tr_041](tr_041.yaml) | LayerNorm → RMSNorm | A | partial | complete_scoped | 불가 |
| [tr_042](tr_042.yaml) | Transformer → Pre-LN | C | partial | complete_scoped | 불가 |
| [tr_043](tr_043.yaml) | Sparse MoE → Switch Transformer | C | partial | complete_scoped | 불가 |
| [tr_044](tr_044.yaml) | Transformer → Switch Transformer | A | partial | complete_scoped | 불가 |
| [tr_045](tr_045.yaml) | Transformer → FNet | A | partial | complete_scoped | 불가 |
| [tr_046](tr_046.yaml) | Relative Position → RoPE | I? | partial | complete_scoped | 불가 |
| [tr_047](tr_047.yaml) | Transformer → ALiBi | A | partial | complete_scoped | 불가 |
| [tr_048](tr_048.yaml) | HiPPO → LSSL | C | partial | complete_scoped | 불가 |
| [tr_049](tr_049.yaml) | LSSL → S4 | A | complete_scoped | complete_scoped | 불가 |
| [tr_050](tr_050.yaml) | S4 → H3 | C | complete_scoped | complete_scoped | 불가 |
| [tr_051](tr_051.yaml) | Transformer → H3 | X | partial | complete_scoped | 불가 |
| [tr_053](tr_053.yaml) | S4 → Hyena | X? | partial | complete_scoped | 불가 |
| [tr_054](tr_054.yaml) | Transformer → FlashAttention | A | partial | partial | 불가 |
| [tr_055](tr_055.yaml) | Transformer → RETRO | C | partial | complete_scoped | 불가 |
| [tr_056](tr_056.yaml) | Transformer → GQA | A | partial | partial | 불가 |
| [tr_057](tr_057.yaml) | Linear Transformer → RWKV | I? | partial | complete_scoped | 불가 |
| [tr_058](tr_058.yaml) | Transformer → RetNet | X | partial | complete_scoped | 불가 |
| [tr_059](tr_059.yaml) | Transformer → RWKV | X | partial | complete_scoped | 불가 |
| [tr_060](tr_060.yaml) | Linear Transformer → RetNet | I? | partial | complete_scoped | 불가 |
| [tr_061](tr_061.yaml) | S4 → Mamba-1 | A | complete_scoped | complete_scoped | 불가 |
| [tr_062](tr_062.yaml) | H3 → Mamba | A | partial | partial | 불가 |
| [tr_063](tr_063.yaml) | Mamba → Mamba-2/SSD | A | partial | partial | 불가 |
| [tr_064](tr_064.yaml) | Mamba → Jamba | C | partial | complete_scoped | 불가 |
| [tr_065](tr_065.yaml) | Transformer → Jamba | A | partial | complete_scoped | 불가 |
| [tr_066](tr_066.yaml) | Switch Transformer → Jamba | C? | partial | complete_scoped | 불가 |
| [tr_067](tr_067.yaml) | Mamba → TTT | P | partial | complete_scoped | 불가 |
| [tr_068](tr_068.yaml) | Transformer → TTT | X | partial | complete_scoped | 불가 |
| [tr_069](tr_069.yaml) | GNMT → Transformer | P | partial | partial | 불가 |
| [tr_070](tr_070.yaml) | E2E Memory Network → Transformer | I | partial | complete_scoped | 불가 |
| [tr_071](tr_071.yaml) | LSTMN → Transformer | I | partial | complete_scoped | 불가 |
| [tr_073](tr_073.yaml) | LSTM → RNN Encoder–Decoder / GRU | I | partial | complete_scoped | 불가 |
| [tr_074](tr_074.yaml) | Linear Transformer → H3 | I | partial | complete_scoped | 불가 |
| [tr_075](tr_075.yaml) | H3 → Hyena | I | partial | complete_scoped | 불가 |
| [tr_076](tr_076.yaml) | Transformer → Hyena | A | partial | complete_scoped | 불가 |
