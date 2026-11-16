# 아키텍처 계보 관계도

카탈로그의 53개 노드와 76개 관계로 생성했습니다. A=구조/계층 계승, C=부품 계승, I=아이디어, P=병목 대응, X=대안 비교입니다. `?`는 원본의 추론 표시입니다. 화살표는 코드 실행 경로가 아닙니다.

부분 대응 60개, 선언 범위 내 대응 완료 16개입니다. [관계별 상태와 YAML](../relations/README.md)을 함께 읽습니다.

```mermaid
flowchart LR
    alibi["ALiBi"]
    bahdanau_attention["Bahdanau Attention"]
    bigbird["BigBird"]
    bytenet["ByteNet"]
    compressive_transformer["Compressive Transformer"]
    convs2s["ConvS2S"]
    decomposable_attention["Decomposable Attention"]
    e2e_memory_network["E2E Memory Network"]
    flashattention["FlashAttention"]
    fnet["FNet"]
    gnmt["GNMT"]
    gqa["GQA"]
    h3["H3"]
    hippo["HiPPO"]
    hyena["Hyena"]
    jamba["Jamba"]
    layernorm["LayerNorm"]
    linear_transformer["Linear Transformer"]
    linformer["Linformer"]
    longformer["Longformer"]
    lssl["LSSL"]
    lstm["LSTM"]
    lstmn["LSTMN"]
    luong_attention["Luong Attention"]
    mamba["Mamba"]
    mamba2_ssd["Mamba-2 / SSD"]
    memory_network["Memory Network"]
    neural_lm["Neural LM"]
    performer["Performer"]
    pre_ln["Pre-LN"]
    qrnn["QRNN"]
    reformer["Reformer"]
    relative_position["Relative Position"]
    resnet["ResNet"]
    retnet["RetNet"]
    retro["RETRO"]
    rmsnorm["RMSNorm"]
    gru_encoder_decoder["RNN Encoder–Decoder / GRU"]
    rnn_lm["RNN LM"]
    rope["RoPE"]
    rwkv["RWKV"]
    s4["S4"]
    seq2seq["Seq2Seq"]
    simple_rnn["Simple RNN"]
    sparse_moe["Sparse MoE"]
    sparse_transformer["Sparse Transformer"]
    structured_self_attention["Structured Self-Attention"]
    switch_transformer["Switch Transformer"]
    synthesizer["Synthesizer"]
    transformer["Transformer"]
    transformer_xl["Transformer-XL"]
    ttt["TTT"]
    wavenet["WaveNet"]
    simple_rnn -->|"A tr_001"| rnn_lm
    neural_lm -->|"P? tr_002"| rnn_lm
    simple_rnn -->|"A tr_003"| lstm
    lstm -->|"C tr_004"| seq2seq
    gru_encoder_decoder -->|"X? tr_005"| seq2seq
    gru_encoder_decoder -->|"A tr_006"| bahdanau_attention
    seq2seq -->|"P tr_007"| bahdanau_attention
    bahdanau_attention -->|"A tr_008"| luong_attention
    memory_network -->|"A tr_009"| e2e_memory_network
    lstm -->|"A tr_010"| lstmn
    e2e_memory_network -->|"I tr_011"| lstmn
    bahdanau_attention -->|"I tr_012"| decomposable_attention
    lstm -->|"A tr_013"| gnmt
    bahdanau_attention -->|"C tr_014"| gnmt
    wavenet -->|"I tr_015"| bytenet
    bytenet -->|"X? tr_016"| convs2s
    bahdanau_attention -->|"C tr_017"| convs2s
    resnet -->|"C tr_018"| gnmt
    resnet -->|"C tr_019"| transformer
    layernorm -->|"C tr_020"| transformer
    bahdanau_attention -->|"I tr_021"| transformer
    luong_attention -->|"C tr_022"| transformer
    decomposable_attention -->|"I tr_023"| transformer
    structured_self_attention -->|"I? tr_024"| transformer
    convs2s -->|"X tr_025"| transformer
    seq2seq -->|"A tr_026"| transformer
    qrnn -->|"X? tr_027"| transformer
    transformer -->|"A tr_028"| relative_position
    transformer -->|"A tr_029"| transformer_xl
    relative_position -->|"C tr_030"| transformer_xl
    transformer_xl -->|"A tr_031"| compressive_transformer
    transformer -->|"A tr_032"| sparse_transformer
    transformer -->|"A tr_033"| reformer
    sparse_transformer -->|"I tr_034"| longformer
    sparse_transformer -->|"I tr_035"| bigbird
    transformer -->|"A tr_036"| linformer
    transformer -->|"A tr_037"| linear_transformer
    transformer -->|"A tr_038"| performer
    linear_transformer -->|"I? tr_039"| performer
    transformer -->|"A tr_040"| synthesizer
    layernorm -->|"A tr_041"| rmsnorm
    transformer -->|"C tr_042"| pre_ln
    sparse_moe -->|"C tr_043"| switch_transformer
    transformer -->|"A tr_044"| switch_transformer
    transformer -->|"A tr_045"| fnet
    relative_position -->|"I? tr_046"| rope
    transformer -->|"A tr_047"| alibi
    hippo -->|"C tr_048"| lssl
    lssl -->|"A tr_049"| s4
    s4 -->|"C tr_050"| h3
    transformer -->|"X tr_051"| h3
    wavenet -->|"I? tr_052"| hyena
    s4 -->|"X? tr_053"| hyena
    transformer -->|"A tr_054"| flashattention
    transformer -->|"C tr_055"| retro
    transformer -->|"A tr_056"| gqa
    linear_transformer -->|"I? tr_057"| rwkv
    transformer -->|"X tr_058"| retnet
    transformer -->|"X tr_059"| rwkv
    linear_transformer -->|"I? tr_060"| retnet
    s4 -->|"A tr_061"| mamba
    h3 -->|"A tr_062"| mamba
    mamba -->|"A tr_063"| mamba2_ssd
    mamba -->|"C tr_064"| jamba
    transformer -->|"A tr_065"| jamba
    switch_transformer -->|"C? tr_066"| jamba
    mamba -->|"P tr_067"| ttt
    transformer -->|"X tr_068"| ttt
    gnmt -->|"P tr_069"| transformer
    e2e_memory_network -->|"I tr_070"| transformer
    lstmn -->|"I tr_071"| transformer
    convs2s -->|"C tr_072"| transformer
    lstm -->|"I tr_073"| gru_encoder_decoder
    linear_transformer -->|"I tr_074"| h3
    h3 -->|"I tr_075"| hyena
    transformer -->|"A tr_076"| hyena
```

[관계 목록](../relations/README.md)
