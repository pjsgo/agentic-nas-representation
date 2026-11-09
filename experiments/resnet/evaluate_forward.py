"""Evaluate the unmodified reviewed code against the installed original on CPU."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "validation"
OUT.mkdir(exist_ok=False)
plan = json.loads((ROOT / "evaluation_plan.json").read_text())
static = json.loads((ROOT / "structural_review.json").read_text())
format_review = json.loads((ROOT / "response_review/format_review.json").read_text())
code_path = ROOT / "response_review/generated_model.py"
assert static["status"] == "passed_static_review"
assert hashlib.sha256(code_path.read_bytes()).hexdigest() == static["generated_source_sha256"]
result = {
    "format_pass": format_review["strict_format_pass"],
    "syntax_pass": format_review["python_syntax"] == "passed",
    "static_structure_pass": True,
    "instantiation": "not_run", "runtime_structure": "not_run",
    "state_mapping": "not_run", "forward_comparison": "not_run",
    "failures": [], "cases": [],
    "generated_code_modified": False,
    "original_source_provided_to_llm": False,
    "evaluation_plan_fixed_before_request": plan["fixed_before_api_call"]
}
def save():
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
save()
stage = "instantiation"
try:
    import torch
    from torchvision.models.resnet import BasicBlock as OriginalBasicBlock
    torch.set_num_threads(1)
    assert plan["device"] == "cpu" and plan["dtype"] == "float32" and plan["mode"] == "eval"
    spec = importlib.util.spec_from_file_location("reviewed_generated_basicblock", code_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    candidate = mod.BasicBlock().to(device="cpu", dtype=torch.float32)
    torch.manual_seed(plan["reference_seed"])
    reference = OriginalBasicBlock(**plan["reference_constructor"]).to(device="cpu", dtype=torch.float32)
    result["instantiation"] = "passed"
    result["entrypoint"] = "BasicBlock()"
    save()

    stage = "runtime_structure"
    roles = static["module_role_mapping"]
    generated_modules = dict(candidate.named_modules())
    original_modules = dict(reference.named_modules())
    assert set(generated_modules) == {"", "conv1", "bn1", "relu", "conv2", "bn2"}
    assert set(original_modules) == set(generated_modules)
    assert candidate.activation is candidate.relu
    assert len({id(getattr(candidate, n)) for n in roles}) == 5
    constructors = static["constructor_review"]
    attributes = {}
    for original_name, generated_name in roles.items():
        a = original_modules[original_name]
        b = generated_modules[generated_name]
        assert type(a) is type(b), (original_name, type(a), type(b))
        if isinstance(a, torch.nn.Conv2d):
            keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding",
                    "dilation", "groups", "padding_mode"]
            assert a.bias is None and b.bias is None
        elif isinstance(a, torch.nn.BatchNorm2d):
            keys = ["num_features", "eps", "momentum", "affine", "track_running_stats"]
        else:
            assert isinstance(a, torch.nn.ReLU)
            keys = ["inplace"]
        av = {k: getattr(a, k) for k in keys}
        bv = {k: getattr(b, k) for k in keys}
        assert av == bv, (original_name, av, bv)
        attributes[original_name] = {"type": type(a).__name__, "original": av, "generated": bv}
    result["runtime_structure"] = "passed"
    result["runtime_attributes"] = attributes
    result["resource_identity"] = {
        "generated_activation_is_relu": True, "module_call_factors": 6,
        "unique_leaf_module_objects": 5, "independent_conv_bn_modules": True
    }
    save()

    stage = "state_mapping"
    # Non-default BN state is a predeclared evaluator fixture, not a code repair.
    rng = torch.Generator(device="cpu").manual_seed(plan["bn_fixture_seed"])
    with torch.no_grad():
        for i, bn in enumerate([reference.bn1, reference.bn2]):
            bn.weight.copy_(0.5 + torch.rand(64, generator=rng))
            bn.bias.copy_(0.1 * torch.randn(64, generator=rng))
            bn.running_mean.copy_(0.2 * torch.randn(64, generator=rng))
            bn.running_var.copy_(0.5 + torch.rand(64, generator=rng))
            bn.num_batches_tracked.fill_([7, 11][i])
    ref_state = reference.state_dict()
    gen_state = candidate.state_dict()
    mapping = []
    transferred = {}
    for original_name, generated_name in roles.items():
        a = original_modules[original_name]
        b = generated_modules[generated_name]
        a_params = dict(a.named_parameters(recurse=False))
        b_params = dict(b.named_parameters(recurse=False))
        a_buffers = dict(a.named_buffers(recurse=False))
        b_buffers = dict(b.named_buffers(recurse=False))
        assert set(a_params) == set(b_params)
        assert set(a_buffers) == set(b_buffers)
        for kind, aa, bb in [("parameter", a_params, b_params), ("buffer", a_buffers, b_buffers)]:
            for name, tensor in aa.items():
                src = original_name + "." + name
                dst = generated_name + "." + name
                assert src in ref_state and dst in gen_state
                assert tensor.shape == bb[name].shape and tensor.dtype == bb[name].dtype
                assert dst not in transferred
                transferred[dst] = ref_state[src].detach().clone()
                mapping.append({"kind": kind, "source": src, "destination": dst,
                                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                "basis": "manually reviewed computational role and module-local field name"})
    assert set(transferred) == set(gen_state)
    assert {m["source"] for m in mapping} == set(ref_state)
    assert len(mapping) == len(ref_state) == len(gen_state) == 12
    loaded = candidate.load_state_dict(transferred, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    for entry in mapping:
        torch.testing.assert_close(ref_state[entry["source"]],
                                   candidate.state_dict()[entry["destination"]],
                                   rtol=0, atol=0)
    assert len({p.data_ptr() for p in candidate.parameters()}) == 6
    reference.eval()
    candidate.eval()
    assert all(not m.training for m in reference.modules())
    assert all(not m.training for m in candidate.modules())
    torch.save({"reference_state_dict": ref_state, "input_case_plan": plan["cases"]},
               OUT / "reference_fixture.pt")
    (OUT / "state_mapping.json").write_text(json.dumps({
        "status": "passed", "mapping": mapping, "strict_load": True,
        "mapping_is_bijective": True, "parameters": 6, "buffers": 6,
        "all_values_exactly_equal_after_transfer": True,
        "no_shape_only_matching_or_manual_code_repair": True
    }, indent=2) + "\n")
    result["state_mapping"] = "passed"
    result["mapped_parameters"] = 6
    result["mapped_buffers"] = 6
    save()

    stage = "forward_comparison"
    class Recorder:
        def __init__(self, model):
            self.events = []
            self.handles = []
            self.pending = {}
            for role in ["conv1", "bn1", "relu", "conv2", "bn2"]:
                layer = getattr(model, role)
                def pre(m, args, role=role):
                    t = args[0]
                    event = {"role": role, "module_id": id(m), "input": t,
                             "input_version": t._version,
                             "input_pointer": t.data_ptr()}
                    self.pending[role] = event
                    self.events.append(event)
                def post(m, args, out, role=role):
                    event = self.pending.pop(role)
                    event.update(output=out, output_version=out._version,
                                 output_pointer=out.data_ptr())
                self.handles.append(layer.register_forward_pre_hook(pre))
                self.handles.append(layer.register_forward_hook(post))
        def close(self):
            for handle in self.handles:
                handle.remove()
        def verify(self, x, y):
            e = self.events
            assert [v["role"] for v in e] == ["conv1", "bn1", "relu", "conv2", "bn2", "relu"]
            assert e[0]["input"] is x
            for i in range(1, 5):
                assert e[i]["input"] is e[i-1]["output"]
            assert e[5]["input"] is e[4]["output"]  # Add keeps the main tensor object.
            assert e[5]["input_version"] == e[4]["output_version"] + 1  # in-place Add
            assert e[2]["module_id"] == e[5]["module_id"]
            for i in (2, 5):
                assert e[i]["output"] is e[i]["input"]
                assert e[i]["output_version"] == e[i]["input_version"] + 1
            assert y is e[5]["output"]
            return {
                "call_order": [v["role"] for v in e],
                "direct_dataflow_preserved": True,
                "same_relu_object_called_twice": True,
                "main_tensor_updated_inplace_by_add": True,
                "both_relus_update_input_inplace": True,
                "events": [
                    {k: v for k, v in item.items() if k not in {"input", "output"}}
                    for item in e
                ]
            }
    saved_cases = {}
    state_before = {k: v.clone() for k, v in candidate.state_dict().items()}
    for case in plan["cases"]:
        shape = case["shape"]
        kind = case["kind"]
        g = torch.Generator(device="cpu").manual_seed(case.get("seed", 0))
        if kind == "normal":
            x = torch.randn(shape, generator=g, dtype=torch.float32)
        elif kind == "noncontiguous_normal":
            n, c, h, w = shape
            x = torch.randn((n, c, w, h), generator=g, dtype=torch.float32).transpose(-1, -2)
            assert not x.is_contiguous()
        elif kind == "zeros":
            x = torch.zeros(shape, dtype=torch.float32)
        elif kind == "ones":
            x = torch.ones(shape, dtype=torch.float32)
        elif kind == "ramp":
            x = torch.linspace(-2.0, 2.0, math.prod(shape), dtype=torch.float32).reshape(shape)
        else:
            raise AssertionError("Unknown predefined case.")
        x_ref, x_gen = x.clone(), x.clone()
        expected_input = x.clone()
        rc, gc = Recorder(reference), Recorder(candidate)
        item = {"name": case["name"], "shape": shape, "kind": kind,
                "input_stride": list(x_gen.stride()),
                "input_contiguous": x_gen.is_contiguous(), "status": "started"}
        try:
            with torch.no_grad():
                expected = reference(x_ref)
                actual = candidate(x_gen)
            assert list(expected.shape) == list(actual.shape) == shape
            assert expected.device.type == actual.device.type == "cpu"
            assert expected.dtype == actual.dtype == torch.float32
            torch.testing.assert_close(x_ref, expected_input, rtol=0, atol=0)
            torch.testing.assert_close(x_gen, expected_input, rtol=0, atol=0)
            item["original_trace"] = rc.verify(x_ref, expected)
            item["generated_trace"] = gc.verify(x_gen, actual)
            diff = (actual - expected).abs()
            item["max_absolute_error"] = diff.max().item()
            item["mean_absolute_error"] = diff.mean().item()
            try:
                torch.testing.assert_close(
                    actual, expected, rtol=plan["comparison"]["rtol"],
                    atol=plan["comparison"]["atol"], check_dtype=True, check_device=True)
            except AssertionError as exc:
                item["status"] = "numerical_mismatch"
                item["error"] = str(exc)
                result["failures"].append({"type": "numerical_mismatch", "case": case["name"]})
            else:
                item["status"] = "passed"
            saved_cases[case["name"]] = {
                "input": x, "reference_output": expected.detach().clone(),
                "generated_output": actual.detach().clone()
            }
        except Exception as exc:
            item["status"] = "forward_execution_failure"
            item["error_type"] = type(exc).__name__
            item["error"] = str(exc)
            result["failures"].append({"type": "forward_execution_failure", "case": case["name"]})
        finally:
            rc.close()
            gc.close()
        result["cases"].append(item)
        save()
    for name, before in state_before.items():
        torch.testing.assert_close(before, candidate.state_dict()[name], rtol=0, atol=0)
    for entry in mapping:
        torch.testing.assert_close(reference.state_dict()[entry["source"]],
                                   candidate.state_dict()[entry["destination"]], rtol=0, atol=0)
    torch.save(saved_cases, OUT / "forward_cases.pt")
    result["eval_state_unchanged_and_original_candidate_still_equal"] = True
    result["forward_comparison"] = "passed" if all(v["status"] == "passed" for v in result["cases"]) else "failed"
    result["numerical_comparison"] = dict(plan["comparison"])
    result["overall_pipeline_pass"] = (
        result["format_pass"] and result["syntax_pass"] and result["static_structure_pass"]
        and result["runtime_structure"] == result["state_mapping"] == result["forward_comparison"] == "passed"
        and not result["failures"]
    )
except Exception as exc:
    error_type = {
        "instantiation": "instantiation_failure",
        "runtime_structure": "structural_mismatch",
        "state_mapping": "state_mapping_failure",
        "forward_comparison": "forward_execution_failure"
    }[stage]
    result[stage] = "failed"
    result["overall_pipeline_pass"] = False
    result["failures"].append({"type": error_type, "error_type": type(exc).__name__, "error": str(exc)})
    (OUT / "failure_traceback.txt").write_text(traceback.format_exc())
finally:
    result["generated_code_sha256_after_evaluation"] = hashlib.sha256(code_path.read_bytes()).hexdigest()
    result["generated_code_unchanged"] = result["generated_code_sha256_after_evaluation"] == static["generated_source_sha256"]
    save()
print(json.dumps({
    "overall_pipeline_pass": result.get("overall_pipeline_pass", False),
    "instantiation": result["instantiation"], "runtime_structure": result["runtime_structure"],
    "state_mapping": result["state_mapping"], "forward_comparison": result["forward_comparison"],
    "cases": [{k: v for k, v in item.items() if k not in {"original_trace", "generated_trace"}}
              for item in result["cases"]],
    "failures": result["failures"], "generated_code_unchanged": result["generated_code_unchanged"]
}, ensure_ascii=False, indent=2))
