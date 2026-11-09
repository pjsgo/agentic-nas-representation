You reconstruct a computation described by a hierarchical tensor representation.
Use only the supplied YAML and supplied frozen representation semantics/operator profile.
Treat architecture labels and factor names as descriptive identifiers. Do not fill gaps with a familiar named architecture. Interpret membership, explicit edges, boundary/call bindings, scan carries, operator attributes, Module identity, Parameter identity, buffer state and mutation exactly as defined.
Generate standalone PyTorch code implementing the declared entry interface and the declared scope. Original source, reference implementations, validator code, fixture values, expected outputs and hidden tests are unavailable and must not be imported, fetched or inferred as an external oracle. Do not access files, network, subprocesses or environment variables. Use only Python standard math utilities and the installed torch package; do not import an architecture implementation or use an opaque architecture operator absent from the profile. No package installation, extra projection, implicit initial state, clone, detach, guard or dtype conversion unless specified by the representation/profile.
Public class/factory and Python variable names are free. Supply entrypoint metadata with factory name, constructor arguments and ordered named forward ports. Preserve hierarchical traceability using semantic IDs in a separate mapping, without adding computation. Required external initial carries/parameter injection must be exposed; fixture values are not to be hard-coded. Return sequence and final state ports exactly as specified.
If the specification does not determine required computation, return needs_clarification with precise unresolved paths. Do not repair its schema or invent numerical formulas. A clarification response is not reconstruction success.
Return one strict JSON object, without Markdown fences, with keys:
- status: "ok" or "needs_clarification"
- code: complete Python source string (empty if blocked)
- entrypoint: {"factory": string, "constructor_kwargs": object, "forward_inputs": array of strings, "forward_outputs": array of strings}
- semantic_mapping: object mapping representation IDs to code locations or exposed tensor roles
- missing_information: array of strings
- assumptions: array of strings (any computational assumption makes the result nonconforming)
No other messages or additional requests are available. The supplied semantics define what may be generated; the architecture label supplies no extra rules.
