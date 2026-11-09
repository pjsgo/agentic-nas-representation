# Source mapping — revision 02

The first SOURCE_MISSING report and all previous root files remain byte-for-byte
unchanged. This independent revision resolves that gate using public upstream
commit `b3cd54c72cd6e4e63f672d334c795b4fe744ef92` (MIT). Exact notebook SHA-256:
`4ad8dbbc56d4fe05b2b61ebfde35be10b90942a1c5a12e399af3aa35522407d1`.
Raw source, GitHub commit metadata, license, source extraction, cell spans and
installed operator hashes are frozen in source/provenance.json.

## Source facts

Notebook code cells 77, 79, 81 (zero-based) define Encoder, Decoder and Seq2Seq.
Encoder returns LSTM final hidden/cell. Seq2Seq requires encoder/decoder hidden
dimensions and layer counts to match; forwards both final tensors without a
learned transformation. Decoder consumes the previous target token, its own
embedding/LSTM Parameters and fc_out. There is no attention or extra context
injection. Teacher forcing initially uses trg[0], then trg[t] for the next step.
The leading outputs[0] is an unused zero placeholder. The pinned PyTorch 2.2.1
rnn.py equations fix gate order i,f,g,o and simultaneous hidden/cell transitions.

## Selected scope and restrictions

One layer, unidirectional, bias=True, dropout=0, CPU float32 eval, complete teacher
forcing (ratio=1), batch=2. Hidden size=4 for both networks (source assertion),
encoder embedding=3 and decoder embedding=5 (allowed distinct dimensions), output
vocabulary=7. Source and target lengths are separate positive symbols S and T.
This is the pre-embedded recurrent core, preserving all computations between the
embedding outputs and vocabulary logits. It is not full token-model coverage.

Source full forward is executed as an oracle at the same constructor settings.
Fixtures provide complete token/embedding values outside the request. Set core
source_embeddings=encoder.embedding(src), target_prev_embeddings=
decoder.embedding(trg[:-1]); compare core logits[j] to original outputs[j+1].
Only the dummy leading zero slot is omitted. Exposed encoder sequence, final
states, bridge states and decoder sequence are source-hook observations, not new
calculations. The unit layer axis is removed at the core boundary.

The source encoder starts at zero: YAML contains explicit zero constants. Decoder
initial states are nonzero source final h/c. Supplemental nonzero encoder and
decoder state tests call the actual source recurrent modules with explicit hx;
these do not claim a changed original Seq2Seq forward interface. No data-dependent
EOS, argmax feedback, training, multiple layers, arbitrary batch or empty sequence
claim is made. At ratio=1 the original random decision/argmax results never enter
the computed logits; RNG-state side effects are outside this numerical scope.

## Identity and information loss

Encoder and decoder have distinct recurrent Parameters; every selected tensor is
mapped bijectively to one declared Parameter ID and reused at every scan step.
The identity handoff must preserve actual Tensor object identity without clone,
projection or hidden/cell swap. All ten core Parameters and zero buffers are
aligned exactly. Two embedding Parameters remain outside the core and are used
only in source fixtures. Opaque LSTM and Linear Module objects are lowered to
functional primitives; source class/container names and opaque Module identities
are not reconstructed. Static regions retain hierarchy and source traceability,
but containers introduce no Module identity under the frozen contract. This
explicit scope loss must not be interpreted as full-source object identity PASS.
Fixture-supplied initialization does not reproduce the notebook's training weight
initialization policy; initializer fidelity is NOT_RUN.

## Source to YAML paths

| Source fact | YAML paths |
|---|---|
| Distinct time scans | factors.read_a.scan, factors.read_b.scan; interface.unit.symbols S/T |
| Zero source initial states | factors.zero_h, factors.zero_c; read_a.h0/c0 dataflow |
| Final-state selection | read_a.scan.carries a_h/a_c; outputs hf/cf |
| Identity h/c handoff | factors.handoff; dataflow read_a.hf/cf -> handoff.h/c -> read_b.h0/c0 |
| Teacher forcing shift | interface.unit.inputs.target_prev; external adapter trg[:-1]; no feedback edge |
| Gate order and simultaneous update | step_a_parts and cell_b_parts; scan carries, step_a/cell_b next_h/next_c |
| Vocabulary projection | factors.readout; parameters.out_w/out_b; step_b.logits -> read_b emits |
| No tied encoder/decoder weights | parameters.a_* versus b_* and all explicit parameter edges |
| Required ancestry | dependency.unit.logits from unit.source and unit.target_prev |

source/mapping.json additionally enumerates every factor and every core Parameter
slot. The independent reference never executes YAML or imports a historical model.
The YAML runtime interpreter provides a third implementation checked against both
source and reference. Candidate checks use parameter roles and normalized runtime
DAGs, not generated class names or exact Python AST statements. They accept only
this declared operator subset; they are not a universal program-equivalence proof.

## Evaluation frozen before execution

Three cases: (S,T,B)=(3,2,2),(4,3,2),(2,4,2), seeds 1701/1702/1703.
Signed nonzero core Parameters, nontrivial embeddings/tokens and nonzero carried
states. All nine exposed outputs use rtol=1e-5, atol=1e-6. Negative controls cover
hidden/cell swaps, earlier source state, omitted bridge, wrong target shift/time,
reverse traversal, lost cell carry, gate/operator/update errors, bias omission,
accidental sharing and malformed-schema errors. Extra context omission is not
applicable because the source declares no such port; bridge omission is tested.
