# 아키텍처 계보 관계도

카탈로그의 45개 노드와 64개 관계로 생성했습니다. A=구조/계층 계승, C=부품 계승, I=아이디어, P=병목 대응, X=대안 비교입니다. `?`는 원본의 추론 표시입니다. 화살표는 코드 실행 경로가 아닙니다.

부분 대응 48개, 선언 범위 내 대응 완료 16개입니다. [관계별 상태와 YAML](../relations/README.md)을 함께 읽습니다.

```mermaid
flowchart LR
    resnet["ResNet"]
    lstm["LSTM"]
    seq2seq["Seq2Seq"]
    transformer["Transformer"]
    linear_transformer["Linear Transformer"]
    s4["S4"]
    mamba["Mamba"]
    simple_rnn["Simple RNN"]
    gru_encoder_decoder["RNN Encoder–Decoder / GRU"]
    bahdanau_attention["Bahdanau Attention"]
    luong_attention["Luong Attention"]
    gnmt["GNMT"]
    layernorm["LayerNorm"]
    transformer_xl["Transformer-XL"]
    compressive_transformer["Compressive Transformer"]
    sparse_transformer["Sparse Transformer"]
    linformer["Linformer"]
    performer["Performer"]
    lssl["LSSL"]
    mamba2_ssd["Mamba-2 / SSD"]
    h3["H3"]
    longformer["Longformer"]
    bigbird["BigBird"]
    flashattention["FlashAttention"]
    gqa["GQA"]
    relative_position["Relative Position"]
    rope["RoPE"]
    alibi["ALiBi"]
    rmsnorm["RMSNorm"]
    pre_ln["Pre-LN"]
    sparse_moe["Sparse MoE"]
    switch_transformer["Switch Transformer"]
    reformer["Reformer"]
    synthesizer["Synthesizer"]
    fnet["FNet"]
    hippo["HiPPO"]
    hyena["Hyena"]
    retnet["RetNet"]
    rwkv["RWKV"]
    memory_network["Memory Network"]
    e2e_memory_network["E2E Memory Network"]
    lstmn["LSTMN"]
    retro["RETRO"]
    jamba["Jamba"]
    ttt["TTT"]
    simple_rnn -->|"A 003"| lstm
    lstm -->|"C 004"| seq2seq
    gru_encoder_decoder -->|"X? 005"| seq2seq
    gru_encoder_decoder -->|"A 006"| bahdanau_attention
    seq2seq -->|"P 007"| bahdanau_attention
    bahdanau_attention -->|"A 008"| luong_attention
    memory_network -->|"A 009"| e2e_memory_network
    lstm -->|"A 010"| lstmn
    e2e_memory_network -->|"I 011"| lstmn
    lstm -->|"A 013"| gnmt
    bahdanau_attention -->|"C 014"| gnmt
    resnet -->|"C 018"| gnmt
    resnet -->|"C 019"| transformer
    layernorm -->|"C 020"| transformer
    bahdanau_attention -->|"I 021"| transformer
    luong_attention -->|"C 022"| transformer
    seq2seq -->|"A 026"| transformer
    transformer -->|"A 028"| relative_position
    transformer -->|"A 029"| transformer_xl
    relative_position -->|"C 030"| transformer_xl
    transformer_xl -->|"A 031"| compressive_transformer
    transformer -->|"A 032"| sparse_transformer
    transformer -->|"A 033"| reformer
    sparse_transformer -->|"I 034"| longformer
    sparse_transformer -->|"I 035"| bigbird
    transformer -->|"A 036"| linformer
    transformer -->|"A 037"| linear_transformer
    transformer -->|"A 038"| performer
    linear_transformer -->|"I? 039"| performer
    transformer -->|"A 040"| synthesizer
    layernorm -->|"A 041"| rmsnorm
    transformer -->|"C 042"| pre_ln
    sparse_moe -->|"C 043"| switch_transformer
    transformer -->|"A 044"| switch_transformer
    transformer -->|"A 045"| fnet
    relative_position -->|"I? 046"| rope
    transformer -->|"A 047"| alibi
    hippo -->|"C 048"| lssl
    lssl -->|"A 049"| s4
    s4 -->|"C 050"| h3
    transformer -->|"X 051"| h3
    s4 -->|"X? 053"| hyena
    transformer -->|"A 054"| flashattention
    transformer -->|"C 055"| retro
    transformer -->|"A 056"| gqa
    linear_transformer -->|"I? 057"| rwkv
    transformer -->|"X 058"| retnet
    transformer -->|"X 059"| rwkv
    linear_transformer -->|"I? 060"| retnet
    s4 -->|"A 061"| mamba
    h3 -->|"A 062"| mamba
    mamba -->|"A 063"| mamba2_ssd
    mamba -->|"C 064"| jamba
    transformer -->|"A 065"| jamba
    switch_transformer -->|"C? 066"| jamba
    mamba -->|"P 067"| ttt
    transformer -->|"X 068"| ttt
    gnmt -->|"P 069"| transformer
    e2e_memory_network -->|"I 070"| transformer
    lstmn -->|"I 071"| transformer
    lstm -->|"I 073"| gru_encoder_decoder
    linear_transformer -->|"I 074"| h3
    h3 -->|"I 075"| hyena
    transformer -->|"A 076"| hyena
```

기존 연구의 범위 제한 실행 근거는 tr_037, tr_061에 보존되어 있습니다. 배포 관계에는 실행 변환기가 포함되지 않습니다.
