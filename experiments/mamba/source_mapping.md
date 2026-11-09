# Source mapping: Mamba-1 selective-scan subunit

Source: local Transformers **4.39.3**, `MambaMixer.slow_forward`, Apache-2.0.
Pinned file SHA-256: `55582c72ea0ce7ebaf809a4793b2e95e2a6580fd1c5b2b0e9393bfc2be068f49`.
Original: `<workspace>/AI4Science_2026summer_archive/.venv-ai4science/lib/python3.11/site-packages/transformers/models/mamba/modeling_mamba.py`.
Exact snapshot and package metadata: `source/`; original remains read-only.
This is Mamba-1's single real-valued selective state, not Mamba-2's grouped/head architecture. No commit is asserted; local file bytes and package metadata are the pin.

## Selected source facts

The computational boundary is **lines 263–279 inclusive**. `delta_logits` is the already-biased result of `dt_proj(time_step)` at line 262, supplied externally. `u` is the activated causal-convolution result; `gate`, B and C are also external boundary values. B/C vary by batch and token; they are not Parameters. External `initial_state` is the preexisting SSM state at line 232, before this subunit. The source can provide it from cache, but this extraction does not implement or claim a cache protocol.

| Source lines | Fact | YAML paths/factors |
|---|---|---|
| 263 | softplus(default beta=1, threshold=20), then transpose(1,2) | `delta_positive`, `delta_layout` |
| 266 | A = -exp(A_log.float()) | `parameters.p_A_log`, `rates_exp`, `rates_negative` |
| 267 | exp(A[None,:,None,:] * delta[:,:,:,None]) | `rates_batch`, `rates_time`, `delta_state`, `decay_argument`, `decay` |
| 268 | delta times per-token B with channel broadcast | `B_channel`, `discrete_B` |
| 269 | discrete B times per-token/channel u | `u_state`, `drive` |
| 273–274 | forward scan, next = decay * previous + drive | `recurrence.scan`, `state.ssm`, `step`, `retained`, `updated` |
| 275–276 | updated state readout with same-token C, remove singleton column | `C_column`, `readout_product`, `readout_vector` |
| 277 | stack readouts along last time axis | `recurrence.scan.emits.readouts.axis: 2` |
| 278 | D skip, u * D[None,:,None] | `parameters.p_D`, `D_batch`, `D_time`, `skip`, `pre_gate` |
| 279 | multiply by SiLU(gate) | `gate_activation`, `gated` |

Machine-resolvable paths and source spans: `source/mapping.json`. Every operation factor appears in that map. Membership and edge order impose no execution ordering. `step` is a separate scan-body definition with globally shared Parameter endpoints. Carries update simultaneously after body evaluation; readout explicitly uses `updated.output`.

## Restrictions chosen for this experiment

- CPU float32, channel size 3, state size 2, positive batch/time lengths. `.float()` and `.to(dtype)` in this source are identity conversions in the declared float32 domain. No casts are silently generalized to mixed precision.
- Pure **computational subunit** with explicit input/state and Parameter injection. Only A_log and D are Parameters. They are distinct objects/storage. There are no Buffers or computational Module operators in this source slice. The parent wrapper Module is an adapter, not a represented operator.
- Source A_log starts as log(arange(1,S+1)) expanded by channel, D starts as ones (118–122). Evaluation uses `fixture_supplied` values for both. Source initializer equivalence is NOT_RUN and outside the forward reconstruction claim.
- T=0 is excluded: original source stacks an empty list and is undefined there. Generic common scan semantics define empty behavior, but this entry's positive Time contract excludes it.
- Exposed final SSM state is a boundary observation of source local `ssm_state`; cache copy_ at 282 is excluded. Initial state is not mutated.

## Full-source behaviors deliberately excluded, not schema-complete claims

Source 226–228 input projection and input/gate split; 230–254 depthwise causal conv (groups=intermediate_size, padding=kernel_size-1 then first T crop), activation and separate convolution cache; 258–261 input-dependent low-rank delta/B/C generation; 262 delta bias before softplus; 281–282 cache write; 285 output projection. Full MambaBlock additionally has normalization and residual (312–329). These distinctions are recorded but omitted computations are not claimed reconstructed. In particular no fixed SSM is substituted for selectivity: the declared delta_logits, B, C remain explicit per-token tensors.

## Representation design and information loss

All computations inside this restricted subunit decompose into frozen functional operators, explicit broadcasts, one scan and two explicit Parameter identities. No new operator or field is used. No selected source computational Module identity is lowered away: dt_proj is outside the boundary. Upstream generation of selective values, bias calculation, conv causality, output projection, initializer fidelity, cache semantics and the whole block are information outside the scope; a successful core test cannot establish them.

## Independent implementation and evaluation evidence

`reference.py` was authored directly from these reviewed source facts. `source/source_adapter.py` compiles the exact 263–279 excerpt unchanged inside an input-binding/return wrapper; it imports no Transformers code and uses required local PyTorch. Tracing observes source local states without replacing the recurrence. `numeric_checks.py` is a separate evaluator and YAML interpreter, never the source of the reference. Original and snapshot hashes are rechecked before preflight. Four tiny fixture cases and all applicable controls are frozen in `fixtures/` and `negative_tests/`.

At this preparation stage numerical execution is **BLOCKED pending a coordinated compute slot** per the user's explicit reply. Syntax/static checks are separate evidence and cannot imply source/reference/numerical PASS. Candidate validation includes numerical/identity checks and a canonical runtime tensor-DAG/factor-observation check in runtime_structure.py. These implementations have not been executed because no compute slot is reserved; source/static review still precedes candidate import.
