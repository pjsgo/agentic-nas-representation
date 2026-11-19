# Tensor trigonometric extension v1

Profile: `tensor_pytorch_2_2_1_v0_3_r1_trig_v1`.

Extends [the tensor profile](tensor-pytorch-2.2.1-v0.3-r1.txt) without changing its operators, factor/interface/dataflow format, region rules or state semantics. Used by [Transformer](../../architectures/transformer/representation.yaml). This local extension is not a certification of the frozen research schema.

| Operator | Inputs and output | Semantics |
|---|---|---|
| cast | input → output | `input.to(dtype=torch.float32)`; required dtype attribute `float32`, same shape and device, no mutation. Selected input is CPU int64. |
| sin | input → output | `torch.sin(input)`, elementwise radians, same shape and floating dtype, no mutation. |
| cos | input → output | `torch.cos(input)`, elementwise radians, same shape and floating dtype, no mutation. |

The selected graph uses width 8, so the sinusoid denominators are the explicit constant `[1, 10, 100, 1000]`. Stack sine then cosine on the final axis and reshape to interleave even/odd features. Token scaling uses the float32 constant sqrt(8). Positions are supplied, not inferred from padding. No additional parameters or buffers are registered. Numerical and graph execution evidence is recorded in the architecture provenance.
