"""Edit-specific validation: semantic call roles, not candidate names or AST text."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "validation"

class StructuralMismatch(Exception):
    pass
class MappingFailure(Exception):
    pass
def require(condition, message, kind=StructuralMismatch):
    if not condition:
        raise kind(message)
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    import torch
    import torchvision
    import yaml
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    plan = json.loads((ROOT / "evaluation_plan.json").read_text())
    decision = json.loads((ROOT / "candidate_review.json").read_text())
    fmt = json.loads((ROOT / "response_review/format_review.json").read_text())
    code_path = ROOT / "response_review/generated_model.py"
    state = torch.load(ROOT / "preflight/reference_fixture.pt", map_location="cpu", weights_only=True)
    inputs = torch.load(ROOT / "preflight/inputs.pt", map_location="cpu", weights_only=True)
    oracle_outputs = torch.load(ROOT / "preflight/oracle_outputs.pt", map_location="cpu", weights_only=True)
    data = yaml.safe_load((ROOT / "generator/input.yaml").read_text())
    result = {"format_pass": fmt["strict_format_pass"], "syntax_pass": fmt["python_syntax"] == "passed",
              "source_review_pass": decision["safe_to_execute"], "stage": "instantiation",
              "failures": [], "cases": [], "overall_pass": False}
    def save():
        (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    save()
    stage = "instantiation"
    try:
        require(decision["safe_to_execute"], "Source review forbids execution.")
        require(hashlib.sha256(code_path.read_bytes()).hexdigest() == decision["code_sha256"], "Reviewed code changed.")
        mod = load_module(code_path, "unedited_qwen_candidate")
        entry = decision["entrypoint"]
        candidate = getattr(mod, entry["name"])(**entry["kwargs"]).to(device="cpu", dtype=torch.float32).eval()
        require(isinstance(candidate, torch.nn.Module), "Entrypoint must produce Module.")
        ref_mod = load_module(ROOT / "reference/edited_reference.py", "independent_edit_reference")
        reference = ref_mod.EditedBasicBlockReference().to(device="cpu", dtype=torch.float32).eval()
        baseline = torchvision.models.resnet.BasicBlock(**plan["reference_constructor"]).to(device="cpu", dtype=torch.float32).eval()
        reference.load_state_dict(state, strict=True)
        baseline.load_state_dict(state, strict=True)
        result["instantiation"] = "passed"
        save()

        # Follow explicit edges through main, ignoring contains array order.
        successors = {}
        for a, b in data["dataflow"]:
            successors.setdefault(a, []).append(b)
        main_roles = []
        endpoint = "main.input"
        seen = set()
        while True:
            require(endpoint not in seen, "Cycle in main dataflow.")
            seen.add(endpoint)
            require(len(successors.get(endpoint, [])) == 1, "Ambiguous main edge.")
            destination = successors[endpoint][0]
            if destination == "main.output":
                break
            role, port = destination.rsplit(".", 1)
            require(port == "input" and role in data["factors"], "Invalid operator input.")
            main_roles.append(role)
            endpoint = role + ".output"
        terminal = successors["merge.output"][0].rsplit(".", 1)[0]
        roles = main_roles + [terminal]
        types = {"conv2d": torch.nn.Conv2d, "batch_norm2d": torch.nn.BatchNorm2d,
                 "relu": torch.nn.ReLU, "gelu": torch.nn.GELU}
        ref_paths = {"conv1": "conv1", "bn1": "bn1", "relu": "relu",
                     "conv2": "conv2", "bn2": "bn2", "activation": "post_add_activation"}

        class Trace:
            def __init__(self, model):
                self.events = []
                self.handles = []
                for name, layer in model.named_modules():
                    if name and isinstance(layer, tuple(types.values())):
                        def pre(m, args, name=name):
                            t = args[0]
                            self.events.append({"path": name, "module": m, "input": t,
                                "input_version": t._version, "input_snapshot": t.detach().clone()})
                        def post(m, args, output):
                            e = self.events[-1]
                            require(e["module"] is m, "Unexpected nested tracked operator.")
                            e.update(output=output, output_version=output._version,
                                     input_version_after=e["input"]._version,
                                     output_snapshot=output.detach().clone())
                        self.handles += [layer.register_forward_pre_hook(pre), layer.register_forward_hook(post)]
            def close(self):
                for h in self.handles:
                    h.remove()
            def verify(self, x, y):
                e = self.events
                require(len(e) == len(roles) and len(e) > 0, "Wrong operator call count.")
                for role, event in zip(roles, e):
                    require(isinstance(event["module"], types[data["factors"][role]["operator"]]), "Wrong operator type for " + role)
                require(e[0]["input"] is x, "Main does not start from x.")
                for i in range(1, len(main_roles)):
                    require(e[i]["input"] is e[i-1]["output"], "Main dataflow diverged.")
                last_main = e[len(main_roles)-1]
                last = e[-1]
                require(last["input"] is last_main["output"], "Add must retain main tensor object.")
                require(last["input_version"] == last_main["output_version"] + 1, "Add did not update main once.")
                for role, event in zip(roles, e):
                    op = data["factors"][role]["operator"]
                    if op == "relu":
                        require(event["module"].inplace and event["output"] is event["input"], "Inner ReLU is not inplace.")
                        require(event["input_version_after"] == event["input_version"]+1, "ReLU mutation changed.")
                    if op == "gelu":
                        require(event["module"].approximate == "none", "Wrong GELU approximation.")
                        require(event["output"] is not event["input"] and event["output"].data_ptr() != event["input"].data_ptr(), "GELU must return separate output.")
                        require(event["input_version_after"] == event["input_version"], "GELU mutated input.")
                for i, a in enumerate(roles):
                    for j, b in enumerate(roles):
                        require((e[i]["module"] is e[j]["module"]) == (data["resource"][a] == data["resource"][b]), "Resource identity inconsistent.")
                require(y is e[-1]["output"], "Unexpected post-activation computation.")
                return {role: event["path"] for role, event in zip(roles, e)}
            def summary(self):
                return [{"path": e["path"], "type": type(e["module"]).__name__,
                         "module_id": id(e["module"]), "input_pointer": e["input"].data_ptr(),
                         "output_pointer": e["output"].data_ptr(), "input_version": e["input_version"],
                         "input_version_after": e["input_version_after"], "output_version": e["output_version"]}
                        for e in self.events]

        def execute(model, x):
            trace = Trace(model)
            before = x.clone()
            try:
                with torch.no_grad():
                    y = model(x)
                torch.testing.assert_close(x, before, rtol=0, atol=0)
                return y, trace
            finally:
                trace.close()

        stage = "structural_fidelity"
        require(len(plan["cases"]) > 0 and len(inputs) == len(plan["cases"]), "Empty/incomplete input plan.")
        probe_x = inputs[plan["cases"][0]["name"]].clone()
        probe_y, probe = execute(candidate, probe_x)
        role_map = probe.verify(probe_x, probe_y)
        modules = dict(candidate.named_modules())
        source_modules = dict(reference.named_modules())
        attr_rows = []
        for role, path in role_map.items():
            a, b = source_modules[ref_paths[role]], modules[path]
            require(type(a) is type(b), "Different module class at " + role)
            if isinstance(a, torch.nn.Conv2d):
                attrs = ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "padding_mode"]
                require(a.bias is None and b.bias is None, "Conv bias changed.")
            elif isinstance(a, torch.nn.BatchNorm2d):
                attrs = ["num_features", "eps", "momentum", "affine", "track_running_stats"]
            elif isinstance(a, torch.nn.ReLU):
                attrs = ["inplace"]
            else:
                attrs = ["approximate"]
            aa, bb = {k: getattr(a,k) for k in attrs}, {k: getattr(b,k) for k in attrs}
            require(aa == bb, "Operator attributes changed at " + role)
            attr_rows.append({"role": role, "candidate_path": path, "reference_attributes": aa, "candidate_attributes": bb})
        result["structural_fidelity"] = "passed"
        result["candidate_role_map"] = role_map
        result["operator_attributes"] = attr_rows
        result["probe_trace"] = probe.summary()
        save()

        stage = "state_mapping"
        ref_state, gen_state = reference.state_dict(), candidate.state_dict()
        transfer, rows = {}, []
        for role, path in role_map.items():
            a, b = source_modules[ref_paths[role]], modules[path]
            for kind, left, right in [
                ("parameter", dict(a.named_parameters(recurse=False)), dict(b.named_parameters(recurse=False))),
                ("buffer", dict(a.named_buffers(recurse=False)), dict(b.named_buffers(recurse=False)))]:
                require(set(left) == set(right), "Module state fields differ at " + role, MappingFailure)
                for field, tensor in left.items():
                    src, dst = ref_paths[role]+"."+field, path+"."+field
                    require(dst not in transfer and src in ref_state and dst in gen_state, "Ambiguous state mapping.", MappingFailure)
                    require(tensor.shape == right[field].shape and tensor.dtype == right[field].dtype, "State shape/dtype differs.", MappingFailure)
                    transfer[dst] = tensor.detach().clone()
                    rows.append({"role": role, "kind": kind, "reference": src, "candidate": dst,
                                 "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
        require(set(transfer) == set(gen_state) and {v["reference"] for v in rows} == set(ref_state), "Incomplete state mapping.", MappingFailure)
        candidate.load_state_dict(transfer, strict=True)
        for v in rows:
            torch.testing.assert_close(candidate.state_dict()[v["candidate"]], ref_state[v["reference"]], rtol=0, atol=0)
        require(all(not m.training for model in [candidate,reference,baseline] for m in model.modules()), "eval mode not applied.")
        (OUT / "state_mapping.json").write_text(json.dumps({"status":"passed","mapping":rows,
            "parameters":sum(v["kind"]=="parameter" for v in rows),
            "buffers":sum(v["kind"]=="buffer" for v in rows),"basis":"runtime semantic roles + module-local state names; no candidate name or shape-only assumptions"},indent=2)+"\n")
        result["state_mapping"] = "passed"
        state_before = {k:v.clone() for k,v in candidate.state_dict().items()}
        save()

        stage = "forward"
        saved = {}
        for case in plan["cases"]:
            row = {"name":case["name"],"shape":case["shape"],"status":"started"}
            try:
                x = inputs[case["name"]]
                xr, xg, xb = x.clone(),x.clone(),x.clone()
                yr, tr = execute(reference,xr)
                yg, tg = execute(candidate,xg)
                yb, tb = execute(baseline,xb)
                tr.verify(xr,yr); tg.verify(xg,yg)
                for i in range(len(main_roles)):
                    torch.testing.assert_close(tr.events[i]["output_snapshot"],tg.events[i]["output_snapshot"],rtol=plan["comparison"]["rtol"],atol=plan["comparison"]["atol"])
                    torch.testing.assert_close(tr.events[i]["output_snapshot"],tb.events[i]["output_snapshot"],rtol=0,atol=0)
                torch.testing.assert_close(tr.events[-1]["input_snapshot"],tb.events[-1]["input_snapshot"],rtol=0,atol=0)
                torch.testing.assert_close(yr,oracle_outputs[case["name"]]["edited_reference"],rtol=0,atol=0)
                torch.testing.assert_close(yb,oracle_outputs[case["name"]]["baseline"],rtol=0,atol=0)
                require(list(yr.shape)==list(yg.shape)==case["shape"],"Output shape mismatch.")
                require(yr.dtype==yg.dtype==torch.float32 and yr.device.type==yg.device.type=="cpu","Output dtype/device mismatch.")
                delta=(yg-yr).abs()
                torch.testing.assert_close(yg,yr,rtol=plan["comparison"]["rtol"],atol=plan["comparison"]["atol"],check_dtype=True,check_device=True)
                row.update(status="passed",max_absolute_error=delta.max().item(),
                    baseline_vs_edited_max_difference=(yb-yr).abs().max().item(),
                    edit_observed=not torch.allclose(yb,yr,rtol=plan["comparison"]["rtol"],atol=plan["comparison"]["atol"]),
                    negative_edited_outputs=int((yr<0).sum().item()),
                    input_contiguous=xg.is_contiguous(),input_stride=list(xg.stride()),
                    main_prefix_preserved=True,reference_trace=tr.summary(),candidate_trace=tg.summary())
                saved[case["name"]]={"input":x,"reference":yr,"candidate":yg,"baseline":yb}
            except StructuralMismatch as exc:
                row.update(status="structural_mismatch",error=str(exc))
                result["failures"].append({"type":"structural_mismatch","case":case["name"],"error":str(exc)})
            except AssertionError as exc:
                row.update(status="numerical_mismatch",error=str(exc))
                result["failures"].append({"type":"numerical_mismatch","case":case["name"],"error":str(exc)})
            except Exception as exc:
                row.update(status="forward_execution_failure",error=str(exc))
                result["failures"].append({"type":"forward_execution_failure","case":case["name"],"error":str(exc)})
            result["cases"].append(row); save()
        torch.save(saved, OUT/"forward_cases.pt")
        torch.save(candidate.state_dict(), OUT/"candidate_state_dict.pt")
        require(len(result["cases"])==len(plan["cases"]) and len(result["cases"])>0,"Missing numerical cases.")
        require(any(c.get("edit_observed") for c in result["cases"]),"Edit never observable.")
        for k,v in state_before.items():
            torch.testing.assert_close(candidate.state_dict()[k],v,rtol=0,atol=0)
        result["eval_state_unchanged"]=True
        result["overall_pass"] = (result["format_pass"] and result["syntax_pass"] and result["source_review_pass"]
            and not result["failures"] and all(c["status"]=="passed" for c in result["cases"]))
    except Exception as exc:
        kind = "structural_mismatch" if isinstance(exc,StructuralMismatch) else "state_mapping_failure" if isinstance(exc,MappingFailure) else {
            "instantiation":"instantiation_failure","structural_fidelity":"structural_mismatch",
            "state_mapping":"state_mapping_failure","forward":"forward_execution_failure"}[stage]
        result["failures"].append({"type":kind,"stage":stage,"error_type":type(exc).__name__,"error":str(exc)})
        (OUT/"failure_traceback.txt").write_text(traceback.format_exc())
    finally:
        result["stage"]=stage
        result["generated_code_unchanged"]=hashlib.sha256(code_path.read_bytes()).hexdigest()==decision["code_sha256"]
        save()
    print(json.dumps({"overall_pass":result["overall_pass"],
        "cases":[{k:v for k,v in c.items() if k not in ["reference_trace","candidate_trace"]} for c in result["cases"]],
        "failures":result["failures"],"candidate_role_map":result.get("candidate_role_map")},ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
