# Original S4 restricted-core source mapping

## Source facts

Repository: https://github.com/state-spaces/s4, commit `e757cef57d89e448c413de7325ed5601aceaac13` (2024-02-21). `source/acquisition.json` pins all acquired URLs/bytes/SHA-256, including Apache-2.0 LICENSE. Main source `source/upstream/models/s4/s4.py` SHA-256: `9f3418b396ec9ccbea30d1f1de92b55e631a3380ebbb19257a73f0219e2a23a6`. Source stays byte-identical. `source/definitions.json` records exact upstream class/function/method spans.

Official README explicitly distinguishes original DPLR/NPLR from mode=diag (S4D). We instantiate the original `SSMKernelDPLR` using d_state=4, rank=1, n_ssm=1, d_model=channels=1, init=legs, backend=naive, real_transform=exp, imag_transform=none, dt_transform=exp, l_max=None. Its inheritance from SSMKernelDiag shares parameter processing, not a substitution of the diagonal transition: the P correction is retained and tested. State dimension four is a small source-valid ordinary NPLR constructor domain; choosing full dimension two would conflict with the initializer's W[-2] access after conjugate-pair halving.

Installed optional scipy/einops/pytorch_lightning are missing. No packages are installed. `source/oracle_adapter.py` executes only reviewed, unmodified AST function/class definitions from the hash-checked source. Mathematical bodies, source constructor, LegS/NPLR initialization, parameter registration, DPLR discretization, Cauchy-naive, FFT kernel/forward and recurrent steps remain upstream code. Only import scaffolding/logger, optional backend flags and a finite, explicit layout-only replacement for einops repeat/rearrange are supplied. `_conj`, complex view aliases and source global dictionaries follow upstream lines 17–25 and 602–606. Unknown layout patterns raise rather than guess. Native torch/numpy are already installed. The source's constructors really run; fixture transfer overwrites all raw parameters afterward, so initialization fidelity is not claimed.

## Selected scope and explicit restrictions

A parameterized functional core of original S4: four full conjugate-completed state coordinates, two stored complex coordinates, rank one, one feature and output channel, bilinear discretization, positive sequence length, external initial state, eval, CPU, rate=1. The raw seven trainable Parameter slots retain exactly the source shapes, real-pair storage, and separate object identities. Public forward takes u[B,1,T], h0[B,1,2,2] (real/imag last axis) and returns kernel[1,1,T], y[B,1,T], h_final[B,1,2,2]. y is emitted after the transition, with D*u; kernel[0]=C*dB and excludes D. Any batch B>=1 and T>=1 is described; the frozen concrete fixtures are B=2/1/2 and T=4/7/3. Only these cases establish numerical evidence.

The represented kernel generator uses impulse stepping through the original DPLR recurrence. This is an explicit equivalent evaluation algorithm for the same finite impulse-response transform, not the original frequency-domain implementation. We additionally execute fresh official naive Cauchy+Woodbury+IRFFT kernel and FFTConv.forward instances to verify outputs and state. Original FFT algorithm structure, C_tilde in-place reparameterization/l_kernel cache, optimizer metadata, train/grad/dropout/bidirectionality/nonlinear S4Block, larger rank/features and initializer behavior are excluded. The source cache buffer is explicitly enumerated/excluded for the restricted raw core; fresh recurrent oracle instances have l_kernel=0. A full original FFT module or architecture-wide structural PASS is not claimed.

## Fact-to-YAML mapping

| Source fact / exact span in s4.py | YAML location | Treatment |
|---|---|---|
| register/inv_dt/B/C/A_real/A_imag, 1035–1070; _get_params, 1072–1105 | parameters.p0–p4; factors.prepare | Source raw real-pair storage, exp/neg and component extraction explicitly decomposed |
| DPLR register P/Q, 1278–1325 | parameters.p5; prepare.pr/pi/qr/qi | P retained, Q=conj(P) via imaginary sign reversal |
| D parameter and additive skip, 1676 and 1738 | parameters.p6; root mul/add leaves | Independent feedthrough, absent from kernel/carry |
| rank-one D/R/E preprocessing, 1450–1471 | factors.prepare and associated leaf factors | Scalar rank-one solve expanded as division; no stored discretized trainable tensor |
| bilinear transition, 1473–1511 | factors.sequence_body, factors.kernel_body | Exact complex arithmetic expanded into real scalar primitives; 2*real contraction includes conjugate partners |
| default/step, 1572–1611 | state.sequence_scan_carry; factors.sequence_scan | Explicit external real-pair carry, simultaneous update, forward traversal, axis 2, output after state transition |
| recurrent kernel equivalence / official kernel, 1327–1427 and 1535–1611 | factors.kernel_scan and state.kernel_scan_* | Explicit initial zero state and one-then-zero pulse carry; kernel time indices start at zero |
| source full-state dense recurrence / state forwarding, 932–958, 1513–1611 | independent reference.py plus validation comparisons | Dense complex full-conjugate basis oracle and original source intermediate/state checks |

`source/factor_mapping.json` enumerates every leaf factor, exact YAML path and source span, all raw Parameter roles and the representation restrictions. `source/definitions.json` is authoritative for exact AST spans; table spans group related source facts. Leaf decomposition names are neutral. Local installed PyTorch source hashes/spans are in `source/operator_sources.json`; the unchanged PyTorch 2.2.1 tensor profile is normative. Actual YAML execution verifies shape contracts, hierarchy boundaries, edges, primitive attributes and scan timing against the independent oracle.

## Representation design and equivalence reasoning

For the stored half-complex state s, source Q=conj(P). The full conjugate-completed contraction reduces to P*(2 Re sum(Q*s)). For rank one, the source _setup_linear matrix solve is scalar: denominator 1+2 Re sum(Q*D*P), R=Q*D/denominator, D=(2/dt-A)^-1, E=2/dt+A. The source state update is D*(v-P*2 Re sum(R*v)), where v=E*s-P*2 Re sum(Q*s)+2*B*u. All complex multiplication and reciprocal operations are expanded into real-pair mul/add/sub/div, with explicit component slices and sums. These are algebraic coordinate expansions, not new YAML formulas or hidden operators. Output is 2 Re sum(C*s_next). See `source/lowering_proof.md` for the derivation and limits.

The kernel scan uses a source input slice only to obtain sequence length. Its separate pulse carry starts at one and becomes zero after each step; it begins from an explicit zeros_like raw-shaped tensor. Thus outputs are C*dB, C*dA*dB, ... with no per-fixture precomputed coefficients. Two body definitions distinguish ordinary batch B from kernel batch one; they share global raw Parameters and root preprocessing. Kernel final state and final pulse are explicit scan outputs that are intentionally unused externally (listed in mapping/schema evidence). The root has no extra Module resource: functional arithmetic plus scan boundaries create no Module under the contract. All seven Parameter identities are globally declared and fixture-supplied; no Buffers are declared in this restricted core.

## Fixtures and independence

`fixtures/plan.json` and `fixtures/cases.json` were frozen before first source/reference numerical execution: float64/complex128, rtol=1e-9, atol=1e-10, CPU eval, seeds 1701/1702/1703. Three different step sizes (0.2,0.7,1.1), three lengths, signed dense/impulse inputs and nonzero carries discriminate required errors. No tolerance widening. The independent Reference builds the dense full complex DPLR matrix and uses a matrix solve; it does not call the YAML generator/interpreter, official step code, generated code, or legacy outputs. Full rank-one scalar lowering and dense-complex reference are different computational implementations checked against actual upstream code.

Source/reference, YAML numerical fidelity, schema consistency, runtime dataflow, and future generated reconstruction are separate results. Parameter injection does not validate initializers. Prior-knowledge influence cannot be ruled out by neutral names; explicit source grounding and excluded scope are preserved here, and mapping/source/reference/tests are never included in the Qwen payload.
