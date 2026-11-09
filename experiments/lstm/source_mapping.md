# Source mapping — LSTM worker

## Source facts
Installed torch 2.2.1, git 6c8c5ad5eaf47a62fafbb4a2747198cbffbf1ff0. File `<site-packages>/torch/nn/modules/rnn.py`, SHA256 `9b4d515adeadd1b2e6941ce0d86e1a2d59fe5db1020dfba9ec0ded581b66c6ed`. Exact dependent files, metadata hashes: source/provenance.json. PKG-INFO declares License: BSD-3; standalone license text is absent in the installed egg-info. Source inspected as text before torch import. Backend implementation is wheel-native; Python calls _VF.lstm_cell, used only as oracle.

| Source fact / span (rnn.py) | YAML path | Evaluation |
|---|---|---|
| RNNCellBase 1130–1147: four distinct Parameter slots, both biases | parameters.p0/p1/p2/p3, dataflow resource endpoints | bijective live operand mapping and object/storage identities |
| LSTMCell 1270–1279: two affine terms, gates, c then h | factors.affine_a/affine_b/combine/partition/gate_a…gate_d/retain/write/update/readout/expose; dataflow; interface | independent equations, source h/c each step, runtime intermediate DAG |
| LSTM 700–714: exact i,f,g,o concatenation order | partition.attributes.sections, edges output0→gate_a,…output3→gate_d | swap-slices negative |
| LSTMCell 1289–1305 explicit input/state and output shapes | interface.root/body/iterate; state.carry_a/carry_b | nonzero unequal h0/c0, every h/c |
| LSTMCell 1313–1322: reuse same cell in loop, emitted hidden sequence | factors.iterate.scan, root membership; body separate definition | Parameter identities observed at each repeated linear |
| LSTMCell 1321 example stacks axis 0 | interface uses B,T layout; scan slices/stacks axis 1 (explicit adapter design) | T=1/3/5, axis and direction negatives |
| LSTMCell 1323–1354: constructor + native call with all four slots | provenance constructor; functional decomposed scope | source oracle same fixtures |
| RNNCellBase 1157–1160 default uniform initialization | parameters.*.initialization=fixture_supplied (restriction) | default initializer NOT_RUN, never claimed preserved |
| Functional linear 2008–2030 and tanh/sigmoid 1965–1987 | tensor frozen profile functional operators | dependent file pinned; runtime normalized ATen trace |

## Deliberate scope restrictions
Single unidirectional cell scan, I=4/H=5, batch-major x, external h0/c0, CPU float32 eval, both biases, no projection/dropout/training, no implicit zeros or unbatched mode. No full nn.LSTM coverage. Source default initialization replaced by explicitly declared fixture injection; numerical equivalence is conditional on aligned state.

## Representation design
Neutral IDs preserve visible computation. Functional operations create no Module resources; p0=weight_ih, p1=weight_hh, p2=bias_ih, p3=bias_hh. Module and Parameter identities are distinct: source cell is one Module reused; YAML has no computational Modules and four shared Parameters. Root/scan are structural only, body a separate forest root. Separate carries update simultaneously; next_a also emits. Dependency adds ancestry checks only. Empty scan returns input states and [B,0,5]; this is profile-defined design, not source sequence support.

## Fixtures and loss
fixtures/plan.json freezes seeds, shapes, tolerances, intermediate plan and unsupported scope before generation. Full-source constructor options, default RNG initialization, unbatched/implicit state and full LSTM output packaging are omitted explicitly. Native internal gates are not instrumented; exposed h/c cross-check each step, while decomposed reference/representation intermediates are compared separately. ATen RegistrationDeclarations.h is also pinned: add line 104, mul 737, sigmoid 910, split_with_sizes 950, tanh 1022. These are operator schemas, not native implementation source. No historical YAML or generated model is an oracle. Architecture names/prior knowledge cannot be proven irrelevant. Agent Edit is NOT_RUN.
