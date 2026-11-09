# Schema delta assessment

Chosen restricted core: **no delta needed or used**. Representation version is
0.3-candidate.1 and profile tensor_pytorch_2_2_1_v0_3_r1. Every computation is an
approved constant/exp/neg/mul/sub/div/add/slice/squeeze/unsqueeze/sum/stack/
zeros_like operation or an explicit scan. Raw complex Parameter storage is
represented as the same real-pair tensors used by the official source; complex
arithmetic is fully expanded with real primitives. A rank-one Woodbury solve
is scalar division. Kernel generation remains computation from continuous
parameters via an impulse-driven S4 recurrence. No discretized/structured term
is hidden in a fixture or promoted to an independent Parameter.

The original full implementation's FFT/IFFT, general structured solves and
C_tilde/l_kernel in-place caching are not represented by this YAML. Those are
**scope exclusions**, not a full-implementation Common Schema PASS. The frozen
profile has no FFT/IFFT or general inverse/solve primitive; representing the
original frequency-domain algorithm directly would require decomposition
with all computation explicit or a future coordinated schema extension. Cache
mutation/control order also needs separate semantics assessment. No extension
or common file is edited here.

Using the equivalent recurrent impulse-kernel algorithm is deliberate and
source-verified; it does not preserve the original fast FFT algorithm's
execution graph or cache state transitions. The selected raw-parameter
recurrent subunit does preserve original S4 DPLR/conjugation, step size,
bilinear discretization, low-rank structure, projection, D, initial/final carry
and kernel indexing. `source_mapping.md` and `source/lowering_proof.md` identify
the exact correspondence. General S4, S4Block and Agent Edit are not covered.

No claim of general-rank matrix inversion or arbitrary S4 expressibility is
inferred from this scalar rank-one result. Numerical and common-schema
preflight outcomes are recorded separately in validation/preflight.json.
