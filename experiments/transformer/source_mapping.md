# Source mapping: frozen eval functional encoder block

## Source facts and provenance

Source is the installed PyTorch **2.2.1**, git `6c8c5ad5eaf47a62fafbb4a2747198cbffbf1ff0`. Exact original paths, full file SHA-256, AST-derived class/function spans, package license metadata and byte-identical local snapshots are in `source/provenance.json`. Originals remain read-only. No historical YAML/generated model was used to author the oracle.

Constructor: `TransformerEncoderLayer(d_model=8,nhead=2,dim_feedforward=12,dropout=0.1,activation="relu",layer_norm_eps=1e-5,batch_first=True,norm_first=False,bias=True,device="cpu",dtype=torch.float32).eval()`.

| Source fact / local snapshot span | Representation paths |
|---|---|
| transformer.py:554–590: constructor and source children; activation string maps to F.relu | dimensions and biases in `interface`, all `parameters`, `factors.activate_a`; affine/eps in `factors.normalize_a`, `normalize_b` |
| activation.py:973–1034: same embed dimension, packed [3D,D] weight and [3D] bias; 2 heads, no extra bias_k/v or zero attention | `parameters.p0,p1`, `project_a`, `partition_a`, `shape_q/k/v`, `order_q/k/v` |
| functional.py:4819–4865: packed Q,K,V row order and self-attention linear | `project_a` weight p0 and bias p1; `partition_a.sections=[8,8,8]`, dim=2; output0→Q, output1→K, output2→V |
| functional.py:5380–5396,5472–5481: head dimension=4, B/H/time rearrangement, head concatenate before out projection | `shape_q/k/v`, `order_q/k/v`, `order_mix`, `join_a`, `project_b` p2/p3 |
| functional.py:5432–5447: Q / sqrt(head_dim), K transpose, logits, key-axis softmax and values | `scale_value`=0.5, `scale_q`, `turn_k`, `score_a`, `prob_a.dim=3`, `mix_a` |
| functional.py:5092–5121: bool True becomes negative infinity; activation.py:1068–1082: 2D mask broadcasts batch/head | external `unit.mask`, `stage_a.mask`, `mask_shape_a/b`, `apply_mask.value=-inf`; entry shape predicates tie query/key T |
| functional.py:5425–5426 and transformer.py:720–731: eval suppresses attention and residual/FF dropout | selected eval scope; no dropout operation in this tensor graph; source dropout=0.1 is retained in the source comparison |
| transformer.py:710–717: post-norm residual then norm, twice | unit→merge_a→normalize_a→stage_b→merge_b→normalize_b→unit.y; both shortcut edges explicit |
| transformer.py:729–731: linear1→relu→linear2, with biases | `stage_b.contains`, `expand_a` p6/p7, `activate_a`, `contract_a` p8/p9 |
| normalization.py:LayerNorm forward; functional.py:2531–2557 | `normalize_a` p4/p5, `normalize_b` p10/p11, normalized_shape=[8], eps=1e-5, affine=true |
| source Parameter enumeration, original slot identity | `source/parameter_roles.json` maps exactly 12 distinct source Parameters to p0–p11; no source buffers in this scope |
| source returns the final x; intermediate values are not additional model outputs | `interface.unit.outputs.y`; internal interfaces expose all leaves for trace validation |

`dataflow` explicitly supplies every input and boundary. Membership in the three regions (`unit`, `stage_a`, `stage_b`) never implies order or Module creation. All computations use functional profile operators; no Module `resource` entries are invented. Intermediate values are mapped to semantic roles in `graph_engine.TRACE_ROLES` and `fixtures/plan.json`.

## Deliberate restrictions, distinct from source facts

This is **one functional encoder block**, not a whole Transformer, source-object API clone, or training implementation. B and T are positive symbolic dimensions; D=8, head count=2, head dimension=4 and FFN width=12 are fixed architecture settings. External mask is always 2D bool with True meaning exclusion, including all-False unmasked fixtures. No padding mask or causal hint is used; a causal mask is an explicit input. All-masked rows and empty sequences are excluded. Parameters are fixture-supplied, not original initializer/RNG equivalents. Gradients are not evaluated.

Original Module hierarchy/identity, constructor initialization and train mode behavior are information deliberately outside this selected tensor scope; see `schema_delta.md`. Parameter object identity and packed structure are inside scope. Q/K/V are distinct row regions of one source Parameter, not three source Parameters; sharing a Q result with K is a wrong computation, tested explicitly.

## Independent implementation and actual source checks

`reference.py` implements equations directly from the pinned spans, imports neither YAML nor generated code, and accepts all state from the evaluator. `graph_engine.py` independently executes the submitted graph. Source comparison uses the actual installed encoder with original state slots and dropout=0.1 in eval, the same nontrivial fixtures, and a frozen rtol=1e-5/atol=1e-6 for all outputs and intermediate boundaries.

Slow source run keeps grad mode enabled and trainable Parameters to avoid optimized encoder/MHA paths **without** switching eval mode. Hooks record both norm input/output boundaries, projection and FFN boundaries. An additional actual source MHA `need_weights=True,average_attn_weights=False` run records packed Q/K/V, scaled Q, masked logits and weights using temporary observation wrappers; no source code or calculation is changed. Main source forward still uses its normal SDPA path. An unmodified no_grad eval run separately checks the optimized encoder result against the same reference/tolerance and records observed optimized entry calls. Instrumentation and numerical evidence live in `validation/source_reference.json`.

## Design and limits of the claims

Neutral IDs carry no architectural defaults. There are no custom computational fields/operators. `fixture_supplied` is a supported initialization policy. Region structure and non-mutating dataflow replace original object hierarchy only within the declared functional scope. Parameter identities are bijective, non-aliasing and fixed by semantic roles rather than source/candidate attribute spelling. There is no recurrent state, scan, temporal cache or buffer to reconstruct; this axis is reported as BLOCKED/not applicable rather than vacuously PASS. A successful reconstruction would provide no Agent Edit evidence. Neutral names cannot establish absence of architecture prior knowledge.
