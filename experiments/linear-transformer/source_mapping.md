# Source mapping — pinned causal recurrent attention core

Source: Idiap fast-transformers, commit `2ad36b97e64cb93862937bd21fcc9568d989561f`, package metadata version `0.4.0`, MIT. Exact downloaded byte hashes and primary URLs are in `source/provenance.json`. The source was absent from the searched local workspace/site-packages/cache. This search is bounded evidence, not proof of universal absence. No package was installed. Existing ResNet and historical generated artifacts are not oracles.

## Source facts and YAML mapping

Let `R` denote `source/upstream/fast_transformers/recurrent/attention/self_attention/linear_attention.py`; `F` denotes `source/upstream/fast_transformers/feature_maps/base.py`. Machine-readable mappings: `source/mapping.json`.

| Source fact | Exact source span | YAML path / interpretation |
|---|---|---|
| Class and constructor | R:19–45 | Entry is the selected computation core, not the complete Python package. Constructor `query_dimensions=3, feature_map=None, eps=1e-6, event_dispatcher=""`. |
| Default feature map | R:40–43,55–57; F:19–23,30–36,57–73 | `factors.f_q_elu`, `f_k_elu`, `f_q_phi`, `f_k_phi`, `f_one`; **ELU is unapproved**, not substituted. Same source feature_map Module is invoked for q/k; functional YAML does not preserve its wrapper identity. |
| Per-step layouts | R:59–68 | `interface.body`, `state.carry_s`, `state.carry_z`: Q/K [B,H,D], V [B,H,M], S [B,H,D,M], Z [B,H,D]. D=3 and M=5 are explicit scope restrictions. |
| External state unpacking | R:47–49,63–68; `_utils.py`:9–17 | `state.*.initial: external`, `factors.iterate.scan.carries`. Deprecated memory path and implicit-zero branch are excluded. |
| Inference state update | R:79–84 | `f_z_next` adds phi(K) into Z; `f_key_expand`, `f_value_expand`, `f_outer`, `f_s_next` add the K⊗V outer product into S. Adds are in place on the incoming carry; next carries alias input objects. |
| Denominator and epsilon | R:86–87 | `f_den_product`, `f_den_sum`, `f_eps`, `f_den_eps`, `f_reciprocal`: epsilon is added after key-dimension contraction and before reciprocal. |
| Query readout | R:88–90 | `f_query_expand`, `f_read_product`, `f_read_sum`, `f_scale_expand`, `f_y`; uses updated S and Z. Source uses a three-input einsum; reference/YAML decompose it. Finite-precision equivalence is **pending actual numerical comparison**, not assumed by associativity. |
| Causal recurrence | R:7,19–21,47–90 | `factors.iterate.scan` forward axis 1, three sequence ports, two explicit carries, output sequence and final S/Z. No future keys, mask approximation or softmax. Sequence wrapper is evaluator design around the one-step source API. |
| Projection/Parameter/Buffer absence | R:37–45; F:21–23,60–68 | Q/K/V are already projected external ports. No output projection in selected class. No parameter slots declared. Runtime complete slot enumeration is pending; no projection-sharing PASS is claimed by absence. |

## Deliberate scope restrictions

Only the deterministic default feature map, CPU float32, eval plus no-grad inference, external nonzero state, T>=1, D=3 and M=5 are frozen. B/H are symbolic positive dimensions. Four fixtures cover B=1/2, H=2, T=1/4/7, signed q/k/v/S, positive nonzero Z, denominator-sensitive inputs, prefixes and chunk continuation. q/k/v have no gradient history. The no-grad branch mutates externally supplied S and Z; input mutation must be tested. No training/autograd, memory alias API, state=None initialization, random feature maps, complete transformer, mask/padding protocols, batching guard behavior, projections or pretrained weights are claimed. T=0 is excluded even though the shared scan syntax defines empty behavior.

## Representation and reference design choices

`reference.py` was independently authored from the pinned spans before any generated model exists. It explicitly forms broadcast outer products and reductions rather than copying the source einsum expressions. Source comparison uses an unchanged class AST node loaded with exact reviewed feature-map/state/event dependencies. `source_adapter.py` excludes only package initialization, registry side effects and builder imports; it replaces no numerical dependency. `setup.py` is preserved for provenance and **never executed**. No source/reference import or forward is authorized while the global compute slot is unavailable.

The draft `representation.yaml` uses neutral identifiers and complete explicit ports/edges/carries, but contains the proposed `elu` primitive and records missing stateless-wrapper identity. It is an inspectable rejected full graph, **not** common-schema-compliant input. `schema_delta.md` separates these gaps. All proposed fields/operators remain unapproved; common files are unchanged.

## Information loss and claims

- Default feature computation cannot be expressed by an existing exact ELU primitive. Replacing it by exp/clamp arithmetic would require demonstrating finite-precision behavior and would alter the explicit operation identity; no such replacement is silently used.
- A functional graph cannot encode identity of the shared stateless ActivationFunctionFeatureMap Module under the current `resource` rule. Runtime source hooks are prepared to observe it, but do not repair missing representation semantics.
- Source numerical grounding, reference acceptance, full identity/state mapping, non-vacuity and executable negatives are BLOCKED by compute scheduling. Their code and fixtures exist; there are no numeric results yet.
- The validator's static checks are architecture-scoped and are not an executable universal schema validator. Candidate import/evaluation is blocked while the schema gate is unresolved. Reconstruction, initialization, training/gradient and Agent Edit success remain NOT_RUN.
