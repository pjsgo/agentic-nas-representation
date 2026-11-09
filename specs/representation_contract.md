# Common Representation Contract — 0.3 candidate, revision 1
Status: FROZEN_FOR_THIS_RUN; candidate semantics, not a proven general schema.
Normative inputs: the exact v0.2 snapshot at `base/representation_semantics_v0_2.md` and the additive rules in `schema_v0_3_candidate.md`. SHA-256 identities are in `manifest.json` and `SHA256SUMS`. All six workers use these identical bytes. Existing v0.2 YAML is evaluated under v0.2, unchanged; it is not silently promoted to v0.3.

| Component | Meaning shared by every worker |
|---|---|
| Architecture Entry | One declared root and explicit external tensor ports; the architecture label is descriptive. |
| Recursive Region | Hierarchical membership, nested regions, and explicit boundary ports; grouping alone creates no Module and no order. |
| Operation | Leaf computation defined by a frozen operator profile, typed ports and attributes. No architecture-name completion. |
| Dataflow | Directed tensor delivery, no implicit transforms; scope and shape must resolve. |
| Interface | Named inputs/outputs, axis layout, shape equations and optional dtype/domain constraints; static contracts do not request new runtime guards. |
| Resource / Identity | `resource` continues to mean Module object identity. Explicit parameter and buffer identities are separate. |
| State | Typed, initialized recurrent carry is distinct from trainable parameters, buffers and ordinary intermediates. |
| Iteration / Scan | Explicit sequence axis, traversal, body interface, simultaneous state transition, emitted sequence and final carry. |
| Dependency | Explicit causal/read-before-write constraints; it neither transfers tensors nor supplies a missing computation. |

Every source-grounded representation must determine calculations, attributes, port bindings, axes, resource identities and initialization policy within its claimed scope. Separate source facts, deliberate scope restrictions, representation design and evaluation fixtures. A missing required computation is BLOCKED, not permission to invent a conventional architecture. A source restriction must be reported as a restriction, never full architecture coverage.

An architecture may propose `schema_delta.md`, but an unapproved new field/operator/meaning cannot receive Common Schema PASS. Keep a reconstructable subset and rejected/unrepresentable remainder distinct. A newly approved rule requires a new coordinated run and all affected preflights; never patch this frozen contract in place.

Reconstruction, numerical fidelity, schema consistency, and human/agent edit ability are separate claims. This run tests reconstruction preparation only. Architecture names and prior knowledge cannot be proven irrelevant by a neutral prompt alone.
